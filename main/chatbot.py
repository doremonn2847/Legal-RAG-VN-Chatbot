from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from main.vector_store import QdrantVectorStore
from main.bm25_retriever import BM25Retriever
from main.reranker import DocumentReranker
from utils.text_processor import VietnameseTextProcessor
from utils.google_search import GoogleSearchTool
from utils.question_refiner import VietnameseLegalQuestionRefiner
from config import Config
from utils.telemetry import trace_stage, llm_trace_config

class VietnameseLegalRAG:
    """Vietnamese Legal RAG System"""
    
    def __init__(self):
        self.vector_store = None
        self.bm25_retriever = None
        self.reranker = None
        self.llm = None
        self.text_processor = VietnameseTextProcessor()
        self.google_search = GoogleSearchTool()
        self.question_refiner = VietnameseLegalQuestionRefiner()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            self.llm = ChatOllama(
                model=Config.MODEL_GEN, base_url=Config.OLLAMA_BASE_URL,
                temperature=0.1, reasoning=False,
                num_ctx=Config.OLLAMA_CONTEXT, num_predict=Config.GENERATION_MAX_TOKENS
            )
            print("Qwen/Ollama LLM configured")

            # Initialize vector store
            self.vector_store = QdrantVectorStore()
            
            # Initialize BM25 retriever
            self.bm25_retriever = BM25Retriever()
            
            # Initialize reranker if enabled
            if Config.ENABLE_RERANKING:
                self.reranker = DocumentReranker(model_name=Config.RERANKER_MODEL)
            else:
                print("Reranking disabled in configuration")
            
        except Exception as e:
            print(f"Error initializing RAG components: {e}")
    
    def setup_indices(self, documents: List[Dict[str, Any]], force_rebuild: bool = False):
        """Setup both vector and BM25 indices"""
        print("Setting up RAG indices...")
        
        try:
            # Setup vector store
            if self.vector_store:
                # Check if we need to create collection
                try:
                    # First, do a simple existence check
                    collections = self.vector_store.client.get_collections().collections
                    collection_exists = any(col.name == self.vector_store.collection_name for col in collections)
                    print(f"Collection existence check: {collection_exists}")
                    
                    if collection_exists:
                        # Collection exists, try to get detailed info
                        try:
                            collection_info = self.vector_store.get_collection_info()
                            has_documents = collection_info.get('points_count', 0) >= len(documents)
                            
                            if force_rebuild:
                                print("Force rebuild requested - recreating vector store...")
                                if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                                    self.vector_store.create_collection(force_recreate=True, vector_size=768)
                                else:
                                    self.vector_store.create_collection(force_recreate=True)
                                self.vector_store.add_documents(documents)
                            elif not has_documents:
                                print("Collection is incomplete - adding remaining documents...")
                                self.vector_store.add_documents(documents)
                            else:
                                print(f"Vector store collection already exists with {collection_info.get('points_count', 0)} documents")
                        except Exception as info_e:
                            print(f"Could not get collection info: {info_e}")
                            if force_rebuild:
                                print("Force rebuild requested - recreating vector store...")
                                if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                                    self.vector_store.create_collection(force_recreate=True, vector_size=768)
                                else:
                                    self.vector_store.create_collection(force_recreate=True)
                                self.vector_store.add_documents(documents)
                            else:
                                print("Assuming collection has documents - skipping setup")
                    else:
                        # Collection doesn't exist, create it
                        print("Collection does not exist - creating new collection...")
                        if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                            self.vector_store.create_collection(force_recreate=True, vector_size=768)
                        else:
                            self.vector_store.create_collection(force_recreate=True)
                        self.vector_store.add_documents(documents)
                        
                except Exception as e:
                    print(f"Error during vector store setup: {e}")
                    print("Attempting to create collection...")
                    if Config.EMBEDDING_MODEL == "bkai-foundation-models/vietnamese-bi-encoder":
                        self.vector_store.create_collection(force_recreate=True, vector_size=768)
                    else:
                        self.vector_store.create_collection(force_recreate=True)
                    self.vector_store.add_documents(documents)
            
            # Setup BM25 index
            if self.bm25_retriever:
                # Try to load existing index
                if not self.bm25_retriever.load_index() or force_rebuild:
                    self.bm25_retriever.build_index(documents)
                    self.bm25_retriever.save_index()
                else:
                    print("BM25 index loaded from file")
            
            print("RAG indices setup completed")
            
        except Exception as e:
            print(f"Error setting up indices: {e}")
            raise
    
    @trace_stage("retrieve-context", as_type="retriever")
    def retrieve_documents(self, query: str, use_hybrid: bool = True, use_reranking: bool = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid approach with optional reranking"""
        retrieved_docs = []
        top_k = Config.TOP_K_RETRIEVAL if top_k is None else top_k
        if not isinstance(top_k, int) or top_k not in (5, 10):
            raise ValueError("top_k must be 5 or 10.")
        
        # Use config default if not specified
        if use_reranking is None:
            use_reranking = Config.ENABLE_RERANKING
        
        # Adjust retrieval counts if reranking is enabled
        bm25_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else top_k
        vector_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else top_k
        
        try:
            if use_hybrid and self.bm25_retriever and self.vector_store:
                # Hybrid retrieval: BM25 + Vector Search
                
                # BM25 retrieval
                bm25_results = self.bm25_retriever.get_relevant_documents(
                    query, top_k=bm25_top_k
                )
                
                # Vector search
                vector_results = self.vector_store.search_similar_documents(
                    query, top_k=vector_top_k
                )
                
                # Combine and deduplicate results with better scoring
                all_docs = {}
                
                # Add BM25 results with proper scoring
                for doc in bm25_results:
                    doc_id = doc.get('id', '')
                    if doc_id:
                        all_docs[doc_id] = {
                            **doc, 
                            'retrieval_method': 'bm25',
                            'bm25_score': doc.get('score', 0)
                        }
                
                # Add vector results with proper scoring
                for doc in vector_results:
                    doc_id = doc.get('id', '')
                    if doc_id:
                        if doc_id in all_docs:
                            # Combine scores if document found by both methods
                            all_docs[doc_id]['retrieval_method'] = 'hybrid'
                            all_docs[doc_id]['vector_score'] = doc.get('score', 0)
                            # Use higher score as main score for now
                            all_docs[doc_id]['score'] = max(
                                all_docs[doc_id].get('bm25_score', 0),
                                doc.get('score', 0)
                            )
                        else:
                            all_docs[doc_id] = {
                                **doc, 
                                'retrieval_method': 'vector',
                                'vector_score': doc.get('score', 0)
                            }
                
                retrieved_docs = list(all_docs.values())
                
                # Sort by score for better ranking
                retrieved_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                
            elif self.vector_store:
                # Vector search only
                retrieved_docs = self.vector_store.search_similar_documents(query, top_k=vector_top_k)
                
            elif self.bm25_retriever:
                # BM25 only
                retrieved_docs = self.bm25_retriever.get_relevant_documents(query, top_k=bm25_top_k)
            
            # Improved similarity filtering logic
            if retrieved_docs:
                # Apply reranking FIRST if enabled (before similarity filtering)
                if use_reranking and self.reranker and retrieved_docs:
                    print(f"Applying reranking to {len(retrieved_docs)} documents...")
                    
                    if Config.USE_SCORE_FUSION:
                        # Use score fusion for better results
                        retrieved_docs = self.reranker.rerank_with_fusion(
                            query, 
                            retrieved_docs, 
                            alpha=Config.RERANKER_FUSION_ALPHA,
                            top_k=top_k
                        )
                    else:
                        # Use pure reranker scores
                        retrieved_docs = self.reranker.rerank_documents(
                            query, 
                            retrieved_docs, 
                            top_k=top_k
                        )
                    print(f"Reranking completed, returning {len(retrieved_docs)} documents")
                    print([(retrieved_doc['id'], retrieved_doc['score']) for retrieved_doc in retrieved_docs])
                    return retrieved_docs[:top_k]
                
                # Check for high-quality documents first (if no reranking)
                high_quality_docs = []
                moderate_quality_docs = []
                
                for doc in retrieved_docs:
                    score = doc.get('score', 0)
                    if score >= Config.SIMILARITY_THRESHOLD:
                        high_quality_docs.append(doc)
                    elif score >= Config.MIN_SIMILARITY_FOR_LEGAL_DOCS:
                        moderate_quality_docs.append(doc)
                
                # Return high quality docs if available
                if high_quality_docs:
                    print(f"Retrieved {len(high_quality_docs)} high-quality documents")
                    print([(high_quality_doc['id'], high_quality_doc['score']) for high_quality_doc in high_quality_docs])
                    return high_quality_docs[:top_k]
                
                # Return moderate quality docs if no high quality ones
                elif moderate_quality_docs:
                    print(f"Retrieved {len(moderate_quality_docs)} moderate-quality documents")
                    print([(moderate_quality_doc['id'], moderate_quality_doc['score']) for moderate_quality_doc in moderate_quality_docs])
                    return moderate_quality_docs[:top_k]
                
                else:
                    print("No documents found with sufficient similarity scores")
                    # Fallback: return best available documents anyway (with lower threshold)
                    if retrieved_docs:
                        print(f"Fallback: returning top {min(top_k, len(retrieved_docs))} documents with best scores")
                        # Sort by score and return best ones
                        retrieved_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                        fallback_docs = retrieved_docs[:top_k]
                        print([(fallback_doc['id'], fallback_doc['score']) for fallback_doc in fallback_docs])
                        return fallback_docs
                    return []
            else:
                # No documents retrieved
                return []
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            if getattr(self, "raise_errors", False):
                raise
            return []
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for LLM"""
        if not documents:
            return "Không có tài liệu pháp luật liên quan được tìm thấy."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Không có tiêu đề')
            content = doc.get('content', '')
            doc_id = doc.get('id', '')
            metadata = doc.get('metadata', {})
            law_id = metadata.get('law_id', '')
            article_id = metadata.get('article_id', '')
            
            # Limit content length
            if len(content) > Config.MAX_ARTICLE_CHARACTERS:
                content = content[:Config.MAX_ARTICLE_CHARACTERS] + "..."
            
            # Format law and article information
            law_info = f"Luật: {law_id}" if law_id else ""
            article_info = f"Điều {article_id}" if article_id else f"ID: {doc_id}"
            
            context_part = f"""
Tài liệu {i}:
{law_info}
{article_info}: {title}
Nội dung: {content}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    @trace_stage("generate-response", as_type="chain")
    def generate_answer(self, query: str, context: str, is_fallback: bool = False) -> str:
        """Generate answer using LLM with context"""
        if not self.llm:
            if getattr(self, "raise_errors", False):
                raise RuntimeError("Language model is unavailable")
            return "Lỗi: Không thể kết nối với mô hình ngôn ngữ."
        
        try:
            # Create prompt based on scenario
            if is_fallback:
                prompt = Config.FALLBACK_SYSTEM_PROMPT.format(
                    context=context,
                    question=query
                )
            else:
                prompt = Config.SYSTEM_PROMPT.format(
                    context=context,
                    question=query
                )
            
            # Generate response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages, **llm_trace_config(self))
            
            answer = response.content
            
            # Add legal disclaimer if enabled and it's a fallback response
            if Config.SHOW_LEGAL_DISCLAIMER and is_fallback:
                disclaimer = "\n\nLưu ý quan trọng: Để đảm bảo quyền lợi của mình, người lao động nên tìm đến các chuyên gia pháp lý hoặc cơ quan chức năng có thẩm quyền để được tư vấn cụ thể và chính xác nhất dựa trên tình hình thực tế của mình."
                answer += disclaimer
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            if getattr(self, "raise_errors", False):
                raise
            return f"Lỗi khi tạo câu trả lời: {str(e)}"
    
    def _is_negative_response(self, response: str) -> bool:
        """Check if the response is a negative/unable to answer response"""
        negative_indicators = [
            "không thể trả lời",
            "không tìm thấy",
            "không có thông tin",
            "xin lỗi",
            "không thể tìm thấy",
            "không có dữ liệu",
            "không rõ",
            "không biết",
            "không đủ thông tin",
            "thiếu thông tin",
            "không có trong",
            "ngoài phạm vi",
            # Add the specific pattern mentioned by user
            "không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này",
            "cần tham khảo thêm các văn bản pháp luật khác",
            "tìm kiếm thông tin chuyên sâu hơn về",
            "tài liệu tham khảo không chứa thông tin đầy đủ"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in negative_indicators)
    
    def answer_question(self, query: str, use_fallback: bool = True, refine_question: bool = False, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Answer a legal question with original-query retrieval and negative response handling"""
        print(f"Processing question: {query}")
        top_k = 5 if top_k is None else top_k
        if type(top_k) is not int or top_k != 5:
            raise ValueError("Answer generation uses top-5 evidence only.")

        # Initialize legal_check variable
        legal_check = None

        # Step 0: Legal domain filtering (if enabled)
        if Config.ENABLE_LEGAL_DOMAIN_FILTER:
            print("🔍 Checking if question is legal-related...")
            legal_check = self.question_refiner.is_legal_question(
                query, 
                use_llm=Config.USE_LLM_FOR_LEGAL_DETECTION
            )
            
            print(f"Legal detection result: {legal_check}")
            
            # If question is not legal-related, return early
            if not legal_check["is_legal"] or legal_check["confidence"] < Config.LEGAL_DOMAIN_CONFIDENCE_THRESHOLD:
                print(f"Question rejected: Not legal-related (confidence: {legal_check['confidence']:.2f})")
                return {
                    'answer': Config.NON_LEGAL_RESPONSE_MESSAGE,
                    'retrieved_documents': [],
                    'fallback_used': False,
                    'context': "",
                    'search_results': [],
                    'search_results_html': "",
                    'original_question': query,
                    'refined_question': query,
                    'question_refinement': None,
                    'legal_domain_check': legal_check,
                    'rejected_non_legal': True
                }
            else:
                print(f"Question accepted: Legal-related (confidence: {legal_check['confidence']:.2f})")

        # The original question is used for both retrieval and generation.
        # Keep legacy response fields and the refine_question argument for callers.
        original_query = query
        refinement_result = None

        retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        
        # Check if we have relevant documents
        if not retrieved_docs and Config.ENABLE_GOOGLE_SEARCH and use_fallback:
            print("No relevant legal documents found, using Google search fallback")
            
            # Use Google search as fallback
            search_results = self.google_search.search_legal_info(query)
            
            if search_results:
                fallback_context = self.google_search.format_search_results(search_results)
                
                # Generate answer with fallback context
                fallback_answer = self.generate_answer(original_query, fallback_context, True)
                
                return {
                    'answer': fallback_answer,
                    'retrieved_documents': [],
                    'fallback_used': True,
                    'search_results': search_results,
                    'context': fallback_context,
                    'search_results_html': self.google_search.format_search_results_for_display(search_results),
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                    'rejected_non_legal': False
                }
            else:
                return {
                    'answer': "Xin lỗi, tôi không tìm thấy thông tin pháp luật liên quan đến câu hỏi của bạn trong cơ sở dữ liệu nội bộ và cũng không thể tìm kiếm thông tin trên web. Vui lòng thử lại với câu hỏi khác hoặc liên hệ với chuyên gia pháp lý.",
                    'retrieved_documents': [],
                    'fallback_used': True,
                    'search_results': [],
                    'context': "",
                    'search_results_html': "",
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                    'rejected_non_legal': False
                }
        elif not retrieved_docs:
            return {
                'answer': "Xin lỗi, tôi không tìm thấy thông tin pháp luật liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.",
                'retrieved_documents': [],
                'fallback_used': False,
                'context': "",
                'search_results': [],
                'search_results_html': "",
                'original_question': original_query,
                'refined_question': query,
                'question_refinement': refinement_result,
                'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                'rejected_non_legal': False
            }

        # Format context
        context = self.format_context(retrieved_docs)

        # Generate answer
        answer = self.generate_answer(original_query, context, False)

        # Check if the generated answer is negative and retry with Google search
        if self._is_negative_response(answer) and use_fallback:
            print("🔍 Detected insufficient information response, activating search tools...")
            
            # Inform user that search is being performed
            search_notification = f"\n\n*🔍 Đang tìm kiếm thông tin bổ sung để trả lời câu hỏi của bạn...*"
            
            # Try Google search if enabled
            if Config.ENABLE_GOOGLE_SEARCH:
                print("📡 Trying web search...")
                search_results = self.google_search.search_legal_info(query)
                
                if search_results:
                    # Generate enhanced response with web information
                    web_context = self.google_search.format_search_results(search_results)
                    combined_context = context + "\n\nThông tin bổ sung từ web:\n" + web_context
                    enhanced_answer = self.generate_answer(original_query, combined_context, True)
                    
                    return {
                        'answer': enhanced_answer,
                        'retrieved_documents': retrieved_docs,
                        'fallback_used': True,
                        'context': combined_context,
                        'search_results': search_results,
                        'search_results_html': self.google_search.format_search_results_for_display(search_results),
                        'search_triggered': True,
                        'original_question': original_query,
                        'refined_question': query,
                        'question_refinement': refinement_result,
                        'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                        'rejected_non_legal': False
                    }
                else:
                    # Google search found nothing useful
                    return {
                        'answer': answer + "\n\n*⚠️ Tôi đã cố gắng tìm kiếm thêm thông tin trên web nhưng không tìm thấy kết quả phù hợp. Để có câu trả lời chính xác hơn, bạn có thể tham khảo ý kiến chuyên gia pháp lý.*",
                        'retrieved_documents': retrieved_docs,
                        'fallback_used': True,
                        'context': context,
                        'search_results': [],
                        'search_results_html': "",
                        'search_triggered': True,
                        'original_question': original_query,
                        'refined_question': query,
                        'question_refinement': refinement_result,
                        'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                        'rejected_non_legal': False
                    }
            else:
                # Google search disabled
                return {
                    'answer': answer + "\n\n*⚠️ Để có câu trả lời chính xác hơn, bạn có thể tham khảo ý kiến chuyên gia pháp lý.*",
                    'retrieved_documents': retrieved_docs,
                    'fallback_used': False,
                    'context': context,
                    'search_results': [],
                    'search_results_html': "",
                    'original_question': original_query,
                    'refined_question': query,
                    'question_refinement': refinement_result,
                    'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
                    'rejected_non_legal': False
                }

        # Return successful result
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'fallback_used': False,
            'context': context,
            'search_results': [],
            'search_results_html': "",
            'original_question': original_query,
            'refined_question': query,
            'question_refinement': refinement_result,
            'legal_domain_check': legal_check if Config.ENABLE_LEGAL_DOMAIN_FILTER else None,
            'rejected_non_legal': False
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of RAG system components"""
        status = {
            'llm_available': self.llm is not None,
            'vector_store_available': self.vector_store is not None,
            'bm25_available': self.bm25_retriever is not None,
            'reranker_available': self.reranker is not None and self.reranker.model is not None,
            'reranking_enabled': Config.ENABLE_RERANKING,
            'ollama_configured': bool(Config.OLLAMA_BASE_URL),
            'inference_model': Config.MODEL_GEN,
            'qdrant_configured': bool(Config.QDRANT_URL and Config.QDRANT_API_KEY)
        }
        
        # Get collection info if available
        if self.vector_store:
            try:
                status['vector_store_info'] = self.vector_store.get_collection_info()
            except:
                status['vector_store_info'] = {}
        
        # Get BM25 stats if available
        if self.bm25_retriever:
            status['bm25_stats'] = self.bm25_retriever.get_index_stats()
        
        # Get reranker info if available
        if self.reranker:
            status['reranker_info'] = self.reranker.get_model_info()
        
        return status
