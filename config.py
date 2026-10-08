import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Local inference through LangChain/Ollama; no provider API key.
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_GEN = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    MODEL_REFINE = MODEL_GEN  # Legacy helpers use the same local model.
    GENERATION_MAX_TOKENS = 2048
    OLLAMA_CONTEXT = 8192

    # QDrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_PATH = os.getenv("QDRANT_PATH")
    
    # Embedding configuration
    # EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    
    COLLECTION_NAME = "final_vietnamese_legal_corpus" if EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" else "bkai_biencoder_vietnamese_legal_corpus"

    # Text Processing Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MAX_ARTICLE_CHARACTERS = 1000
    INDEX_BATCH_SIZE = 128
    EMBEDDING_BATCH_SIZE = 16

    # Data Paths
    DATA_DIR = "data"
    CORPUS_PATH = "data/corpus/legal_corpus.json"
    STOPWORDS_PATH = "data/utils/stopwords.txt"

    # RAG Configuration
    TOP_K_RETRIEVAL = 5
    BM25_TOP_K = TOP_K_RETRIEVAL
    BM25_B = 0.65
    BM25_K1 = 1.2
    SIMILARITY_THRESHOLD = 0.25

    # Reranker Configuration
    ENABLE_RERANKING = True
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K = TOP_K_RETRIEVAL
    RERANK_BEFORE_RETRIEVAL_TOP_K = 25
    USE_SCORE_FUSION = True
    RERANKER_FUSION_ALPHA = 0.8

    # Google Search Configuration
    ENABLE_GOOGLE_SEARCH = True
    GOOGLE_SEARCH_RESULTS_COUNT = 10
    MIN_SIMILARITY_FOR_LEGAL_DOCS = 0.15

    # Question Refinement Configuration
    ENABLE_QUESTION_REFINEMENT = False
    USE_LLM_FOR_REFINEMENT = True

    # Advanced LLM Refinement Settings
    ENABLE_CHAIN_OF_THOUGHT = True
    ENABLE_ITERATIVE_REFINEMENT = True
    ENABLE_LLM_VALIDATION = True
    MAX_REFINEMENT_ITERATIONS = 3
    MIN_CONFIDENCE_SCORE = 0.7

    # UI Display Settings - Control what information to show in responses
    SHOW_REFINEMENT_INFO = False  # 🔧 Câu hỏi đã được tối ưu
    SHOW_SEARCH_TRIGGER_INFO = False  # 🔍➡️🌐 Tự động tìm kiếm
    SHOW_SOURCE_INFO = False  # 📚 Dựa trên X tài liệu, 🌐 Thông tin từ web
    SHOW_LEGAL_DISCLAIMER = False  # Lưu ý về tìm chuyên gia pháp lý

    # Legal Domain Filtering Settings
    ENABLE_LEGAL_DOMAIN_FILTER = True  # Only answer legal questions
    LEGAL_DOMAIN_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to consider a question legal
    USE_LLM_FOR_LEGAL_DETECTION = True  # Use LLM for more accurate legal domain detection
    NON_LEGAL_RESPONSE_MESSAGE = """Xin lỗi, tôi là trợ lý pháp lý chuyên biệt và chỉ có thể trả lời các câu hỏi liên quan đến pháp luật Việt Nam.

Tôi có thể giúp bạn với:
• Quyền và nghĩa vụ theo pháp luật
• Thủ tục pháp lý và hành chính
• Quy định về doanh nghiệp, lao động, thuế
• Luật gia đình, hôn nhân, thừa kế
• Bất động sản và quyền sử dụng đất
• Hợp đồng và giao dịch dân sự
• Tranh chấp và giải quyết khiếu nại

Vui lòng đặt câu hỏi về các vấn đề pháp lý và tôi sẽ cố gắng hỗ trợ bạn!"""

    # System Prompt
    SYSTEM_PROMPT = """Bạn là trợ lý pháp lý thông minh chuyên sâu về luật pháp Việt Nam. Nhiệm vụ của bạn là cung cấp các câu trả lời chính xác và dễ hiểu cho các câu hỏi pháp lý, dựa trên các tài liệu luật được cung cấp.

Khi trả lời:
1.  **Chỉ sử dụng thông tin** trực tiếp từ các điều luật và văn bản được cung cấp trong phần "Tài liệu tham khảo". Tuyệt đối không suy diễn hoặc thêm thông tin bên ngoài.
2.  **Trích dẫn chính xác** tên luật (ví dụ: Luật Doanh nghiệp 2020), số hiệu văn bản (nếu có), và điều khoản cụ thể (ví dụ: Điều 3, Khoản 2).
3.  **Giải thích rõ ràng, ngắn gọn và khách quan**, tập trung vào việc làm sáng tỏ nội dung của điều luật liên quan đến câu hỏi.
4.  **Nếu tài liệu tham khảo không chứa thông tin đầy đủ hoặc trực tiếp để trả lời câu hỏi**, hãy thông báo rõ ràng rằng "Không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này."
5.  **Trình bày bằng tiếng Việt chuẩn xác.**

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""

    # Fallback System Prompt for Google Search
    FALLBACK_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý thông minh chuyên sâu về luật pháp Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Khi trả lời:
1.  **Tóm tắt và trình bày thông tin liên quan** một cách tự nhiên và mạch lạc.
2.  **Cung cấp các liên kết (URLs)** của các nguồn đã được tham khảo ở cuối câu trả lời để người dùng có thể kiểm tra thêm.
3.  **Giải thích rõ ràng và dễ hiểu**.
4.  **Trình bày bằng tiếng Việt chuẩn xác.**

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""
