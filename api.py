"""Local HTTP interface. Start from the repository root with one worker."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import pickle
import threading
import time
from typing import Annotated, Callable

import anyio
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from utils.telemetry import Telemetry
from main.uploaded_documents import UploadedDocuments, MAX_FILE_BYTES

logger = logging.getLogger(__name__)
DocumentToken = Annotated[str | None, Header(alias="X-Document-Token", min_length=43,
    max_length=43, pattern=r"^[A-Za-z0-9_-]+$")]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    use_web_fallback: StrictBool = False

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class Source(BaseModel):
    id: str
    title: str
    law_id: str
    article_id: str
    excerpt: str
    score: float
    source_type: str = "legal_corpus"
    filename: str | None = None
    page: int | None = None


class WebSource(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    question: str
    sources: list[Source]
    web_sources: list[WebSource]
    fallback_used: bool
    rejected_non_legal: bool
    elapsed_seconds: float
    trace_id: str | None = None


def close_pipeline(rag):
    store = getattr(rag, "vector_store", None)
    client = getattr(store, "client", None)
    if client is not None:
        client.close()


def check_dependencies(rag):
    """Check services without generating an answer or calculating index statistics."""
    from utils.inference import check_inference_ready

    check_inference_ready()
    store = rag.vector_store
    info = store.client.get_collection(store.collection_name)
    if not info.points_count or info.points_count != len(rag.bm25_retriever.documents):
        raise RuntimeError("Qdrant and BM25 indexes must have matching nonzero document counts")


def load_pipeline():
    from config import Config
    from main.chatbot import VietnameseLegalRAG

    if not Config.ENABLE_RERANKING:
        raise RuntimeError("Enable BGE reranking before starting the API")
    index = Path("index/bm25_index.pkl")
    if not index.is_file():
        raise RuntimeError("Existing index/bm25_index.pkl is required; API startup never rebuilds indexes")
    rag = VietnameseLegalRAG()
    try:
        if (rag.llm is None or rag.vector_store is None
                or rag.vector_store.client is None or rag.vector_store.embedding_model is None
                or rag.bm25_retriever is None or rag.reranker is None or rag.reranker.model is None):
            raise RuntimeError("RAG components failed to initialize; see startup diagnostics")
        # Load the project's trusted pickle without the legacy loader's deletion-on-error behavior.
        with index.open("rb") as handle:
            data = pickle.load(handle)
        for key in ("bm25", "documents", "tokenized_corpus"):
            setattr(rag.bm25_retriever, key, data[key])
        if rag.bm25_retriever.bm25 is None:
            raise RuntimeError("BM25 index is empty")
        count = len(rag.bm25_retriever.documents)
        if (not count or len(rag.bm25_retriever.tokenized_corpus) != count
                or rag.bm25_retriever.bm25.corpus_size != count):
            raise RuntimeError("BM25 index contains inconsistent document counts")
        for component in (rag, rag.vector_store, rag.reranker):
            component.raise_errors = True
        return rag
    except Exception:
        close_pipeline(rag)
        raise


def create_app(rag_factory: Callable = load_pipeline, readiness_probe: Callable = check_dependencies,
               document_factory: Callable = UploadedDocuments):
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        rag = await anyio.to_thread.run_sync(rag_factory)
        telemetry = Telemetry.from_environment()
        try:
            await anyio.to_thread.run_sync(readiness_probe, rag)
            app.state.rag = rag
            app.state.documents = document_factory(rag.vector_store)

            async def expire_documents():
                while True:
                    def expire():
                        if lock.acquire(blocking=False):
                            try:
                                app.state.documents.cleanup()
                            except Exception:
                                logger.warning("Upload expiry cleanup failed; will retry")
                            finally:
                                lock.release()
                    await anyio.to_thread.run_sync(expire)
                    await anyio.sleep(60)
            for component in (rag, getattr(rag, "reranker", None), getattr(rag, "question_refiner", None)):
                if component is not None:
                    component.telemetry = telemetry
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(expire_documents)
                try:
                    yield
                finally:
                    tasks.cancel_scope.cancel()
        finally:
            def cleanup():
                with lock:
                    try:
                        close_pipeline(rag)
                    finally:
                        telemetry.shutdown()
            await anyio.to_thread.run_sync(cleanup)

    app = FastAPI(title="Vietnamese Legal RAG", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def bound_upload(request, call_next):
        if request.method == "POST" and request.url.path == "/documents":
            try:
                length = int(request.headers.get("content-length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                return JSONResponse({"detail": "Upload Content-Length is required"}, status_code=411)
            if length > MAX_FILE_BYTES + 1024*1024:
                return JSONResponse({"detail": "Upload exceeds 10 MiB file limit"}, status_code=413)
        return await call_next(request)

    @app.post("/documents")
    async def upload_document(file: Annotated[UploadFile, File()], document_token: DocumentToken = None,
                              force_ocr: Annotated[bool, Form()] = False):
        try:
            if file.size is not None and file.size > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds 10 MiB")
            content = await file.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds 10 MiB")
        finally:
            await file.close()

        def run():
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "Pipeline is busy; retry later")
            try:
                with app.state.rag.telemetry.span("index-uploaded-document", as_type="span",
                    input={"bytes": len(content)} ) as observation:
                    result = app.state.documents.add(file.filename or "", content, document_token, force_ocr=force_ocr)
                    Telemetry.update(observation, output={"files": len(result['files']),
                        "chunks": sum(f['chunks'] for f in result['files'])})
                    return result
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            except Exception:
                logger.warning("Document upload failed")
                raise HTTPException(502, "Document processing or indexing failed; try again") from None
            finally:
                lock.release()
        return await anyio.to_thread.run_sync(run)

    @app.delete("/documents")
    async def delete_documents(document_token: DocumentToken = None):
        def run():
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "Pipeline is busy; retry later")
            try:
                app.state.documents.delete(document_token)
                return {"status": "deleted"}
            except ValueError:
                raise HTTPException(401, "Document session is missing or expired") from None
            except Exception:
                raise HTTPException(502, "Could not delete documents; retry later") from None
            finally:
                lock.release()
        return await anyio.to_thread.run_sync(run)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        try:
            await anyio.to_thread.run_sync(readiness_probe, app.state.rag)
        except Exception:
            logger.exception("Readiness check failed")
            raise HTTPException(503, "Pipeline dependencies are unavailable") from None
        return {"status": "ready"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest, session_id: Annotated[str | None, Header(
        alias="X-Session-ID", max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )] = None, document_token: DocumentToken = None):
        def run():
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "Pipeline is busy; retry later", headers={"Retry-After": "5"})
            try:
                start = time.monotonic()
                uploaded = None
                if document_token:
                    try:
                        app.state.documents.session(document_token)
                    except ValueError:
                        raise HTTPException(401, "Document session expired; upload again") from None
                    uploaded = lambda query: app.state.documents.search(query, document_token)
                with app.state.rag.telemetry.span(
                    "answer-legal-question", as_type="chain", input=request.question, session_id=session_id,
                    metadata={"top_k": 5, "refinement": False, "web_fallback": request.use_web_fallback},
                ) as observation:
                    trace_id = observation.trace_id if observation is not None else None
                    result = app.state.rag.answer_question(
                        request.question, use_fallback=request.use_web_fallback,
                        refine_question=False, top_k=5,
                        **({"uploaded_documents": uploaded} if uploaded is not None else {}),
                    )
                    Telemetry.update(observation, output=result["answer"], metadata={
                        "article_ids": [doc.get("id") for doc in result["retrieved_documents"][:5]],
                        "fallback_used": result.get("fallback_used", False),
                        "rejected_non_legal": result.get("rejected_non_legal", False),
                    })
                sources = []
                for doc in result["retrieved_documents"][:5]:
                    metadata = doc.get("metadata") or {}
                    sources.append(Source(
                        id=str(doc.get("id", "")), title=str(doc.get("title", "")),
                        law_id=str(metadata.get("law_id", "")),
                        article_id=str(metadata.get("article_id", "")),
                        excerpt=doc.get("content", "")[:2000], score=float(doc.get("score", 0)),
                        source_type=metadata.get("source", "legal_corpus"),
                        filename=metadata.get("filename"), page=metadata.get("page"),
                    ))
                return AskResponse(
                    answer=result["answer"], question=request.question, sources=sources,
                    web_sources=[WebSource(title=str(item.get("title", "")),
                                           url=str(item.get("url", item.get("link", ""))))
                                 for item in result.get("search_results", [])],
                    fallback_used=result.get("fallback_used", False),
                    rejected_non_legal=result.get("rejected_non_legal", False),
                    elapsed_seconds=round(time.monotonic() - start, 3),
                    trace_id=trace_id,
                )
            except HTTPException:
                raise
            except Exception:
                logger.exception("Question processing failed")
                raise HTTPException(502, "Pipeline failed to process the question") from None
            finally:
                lock.release()
        return await anyio.to_thread.run_sync(run)

    return app


app = create_app()
