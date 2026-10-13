import os
import logging
import time
import uuid
from html import escape
from urllib.parse import urlsplit
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()


class VietnameseLegalChatbot:
    """Gradio client; inference and indexes belong to the FastAPI service."""

    def __init__(self, api_base_url=None):
        self.api_base_url = (api_base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")

    def process_message(self, message, history, session_id=None, document_token=None):
        history = list(history or [])
        question = message.strip()
        if not question:
            return history, "", "💬 Nhập câu hỏi để bắt đầu"
        if len(question) > 2000:
            return history, "", "⚠️ Câu hỏi tối đa 2000 ký tự"
        start = time.monotonic()
        headers = {}
        if session_id:
            headers['X-Session-ID'] = session_id
        if document_token:
            headers['X-Document-Token'] = document_token
        try:
            response = requests.post(
                self.api_base_url + "/ask",
                json={"question": question, "use_web_fallback": False},
                timeout=(5, 600),
                **({"headers": headers} if headers else {}),
            )
            response.raise_for_status()
            result = response.json()
            answer = result["answer"]
            if not isinstance(answer, str):
                raise TypeError("Invalid answer")
            docs = (self._format_legal_guidance() if result.get("rejected_non_legal")
                    else self._format_retrieved_documents(result.get("sources", []), result.get("web_sources", [])))
            status = f"✅ Hoàn thành ({time.monotonic() - start:.1f}s)"
        except requests.Timeout:
            answer = "⚠️ Hết thời gian chờ. Hệ thống có thể vẫn đang xử lý; vui lòng chờ trước khi thử lại."
            docs, status = "", "❌ Hết thời gian chờ"
        except requests.ConnectionError:
            answer = "⚠️ Không kết nối được dịch vụ trả lời. Vui lòng khởi động FastAPI và thử lại."
            docs, status = "", "❌ Dịch vụ chưa sẵn sàng"
        except requests.HTTPError as error:
            code = error.response.status_code if error.response is not None else None
            if code == 429:
                answer = "⚠️ Hệ thống đang xử lý câu hỏi khác. Vui lòng thử lại sau."
            elif code == 422:
                answer = "⚠️ Câu hỏi không hợp lệ. Vui lòng nhập từ 1 đến 2000 ký tự."
            elif code == 401:
                answer = "⚠️ Phiên tài liệu đã hết hạn. Vui lòng xóa tài liệu cũ và tải lên lại."
            else:
                answer = "⚠️ Dịch vụ trả lời đang gặp lỗi. Vui lòng thử lại sau."
            docs, status = "", "❌ Lỗi dịch vụ"
        except (ValueError, KeyError, TypeError):
            logging.getLogger(__name__).exception("Invalid API response")
            answer = "⚠️ Dịch vụ trả về dữ liệu không hợp lệ."
            docs, status = "", "❌ Lỗi phản hồi"
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        return history, docs, status

    def upload_document(self, filepath, token=None, force_ocr=False):
        if not filepath:
            return "⚠️ Chọn tệp PDF/DOCX trước.", token
        try:
            path = Path(filepath)
            if path.suffix.lower() not in ('.pdf', '.docx') or path.stat().st_size > 10*1024*1024:
                return "⚠️ Chỉ nhận PDF/DOCX tối đa 10 MiB.", token
            with path.open('rb') as handle:
                response = requests.post(self.api_base_url + '/documents',
                    files={'file': (path.name, handle)},
                    data={'force_ocr': str(bool(force_ocr)).lower()},
                    headers={'X-Document-Token': token} if token else {}, timeout=(5, 660 if force_ocr else 210))
            response.raise_for_status()
            result = response.json()
            files = ', '.join(escape(f['filename']) for f in result['files'])
            return f"✅ **Xử lý xong — tài liệu sẵn sàng để hỏi.**\n\nĐã lập chỉ mục: {files}. Tài liệu chỉ dùng trong phiên này (24 giờ).", result['token']
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            return "⚠️ **Xử lý không thành công.** Kiểm tra giới hạn (3 tệp, 50 trang PDF), thử tệp có văn bản hoặc thử lại khi dịch vụ rảnh.", token

    def clear_documents(self, token):
        if not token:
            return
        response = requests.delete(self.api_base_url + '/documents',
            headers={'X-Document-Token': token}, timeout=(5, 30))
        if response.status_code != 401:
            response.raise_for_status()

    def _format_retrieved_documents(self, sources, web_sources=None):
        # Escape corpus text before putting it into Markdown/HTML.
        if not sources and not web_sources:
            return "📄 **Không tìm thấy tài liệu tham khảo**"
        blocks = [f"## 📄 Tài liệu tham khảo ({len(sources)} tài liệu)"]
        for i, source in enumerate(sources, 1):
            law = escape(str(source.get("law_id", "")))
            article = escape(str(source.get("article_id", "")))
            title = escape(str(source.get("title", "")))
            identifier = escape(str(source.get("id", "")))
            excerpt = escape(str(source.get("excerpt", "")))
            if source.get('source_type') == 'upload':
                filename = escape(str(source.get('filename', title)))
                page = source.get('page')
                origin = f"Tài liệu tải lên: {filename}" + (f" · Trang {escape(str(page))}" if page else " · Đoạn văn bản DOCX")
            else:
                origin = f"Luật: {law} · Điều: {article}"
            blocks.append(f"<h3>{i}. {title}</h3><p>{origin}</p>"
                          f"<p>ID: {identifier}</p><p>{excerpt}</p>")
        for source in web_sources or []:
            url = str(source.get("url", ""))
            if urlsplit(url).scheme in ("http", "https"):
                blocks.append(f'<p><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                              f'{escape(str(source.get("title", url)))}</a></p>')
        return "\n\n".join(blocks)

    def _format_legal_guidance(self):
        """Format legal guidance for rejected non-legal questions"""
        return """## 📚 Hướng dẫn sử dụng trợ lý pháp lý

Câu hỏi của bạn không thuộc lĩnh vực pháp luật mà tôi có thể hỗ trợ.

### ⚖️ Tôi có thể giúp bạn với:
- **Doanh nghiệp**: Thành lập, giải thể, vốn điều lệ, giấy phép kinh doanh
- **Lao động**: Hợp đồng lao động, lương, nghỉ phép, sa thải, bảo hiểm
- **Thuế**: Kê khai thuế, miễn thuế, thuế thu nhập cá nhân/doanh nghiệp
- **Bất động sản**: Mua bán nhà đất, chuyển nhượng, sổ đỏ, quyền sử dụng đất
- **Gia đình**: Hôn nhân, ly hôn, thừa kế, nuôi con, quyền con cái
- **Dân sự**: Hợp đồng, tranh chấp, bồi thường, quyền sở hữu
- **Hành chính**: Thủ tục pháp lý, giấy tờ, cơ quan nhà nước

### 💡 Gợi ý:
Hãy đặt câu hỏi cụ thể về các vấn đề pháp lý trên để nhận được hỗ trợ tốt nhất!"""
    
    def get_sample_questions(self):
        """Get categorized sample questions"""
        return {
            "🏢 Doanh nghiệp": [
                "Thủ tục thành lập doanh nghiệp như thế nào?",
                "Quy định về vốn điều lệ tối thiểu?",
                "Thủ tục giải thể doanh nghiệp?"
            ],
            "⚖️ Lao động": [
                "Quyền lợi của người lao động khi bị sa thải?",
                "Quy định về thời gian làm việc?",
                "Chế độ nghỉ phép hàng năm?"
            ],
            "💰 Thuế": [
                "Điều kiện miễn thuế thu nhập cá nhân?",
                "Cách tính thuế giá trị gia tăng?",
                "Thủ tục kê khai thuế?"
            ],
            "🏠 Bất động sản": [
                "Hợp đồng mua bán nhà đất cần giấy tờ gì?",
                "Quy trình chuyển nhượng quyền sử dụng đất?",
                "Thủ tục cấp sổ đỏ?"
            ],
            "👨‍👩‍👧‍👦 Gia đình": [
                "Thủ tục ly hôn thuận tình?",
                "Quyền thừa kế của con cái?",
                "Quy định về nuôi con nuôi?"
            ]
        }

def load_css():
    """Load CSS from external file"""
    try:
        with open('css/style.css', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open('css/app/style.css', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print("⚠️ Warning: CSS file not found. Using default styles.")
            return ""

def create_chatbot_interface():
    """Create the local Gradio interface backed by FastAPI"""
    
    # Initialize chatbot
    chatbot = VietnameseLegalChatbot()
    
    # Load CSS from external file
    css = load_css()
    
    with gr.Blocks(
        css=css, 
        title="Trợ lý Pháp lý Việt Nam", 
        theme=gr.themes.Default(),
        analytics_enabled=False,
        delete_cache=(3600, 86400)
    ) as interface:
        
        # Enhanced header with simple styling
        gr.HTML("""
        <div class="main-header">
            <h1>⚖️ Trợ lý Pháp lý Việt Nam</h1>
            <p>Hệ thống tư vấn pháp luật thông minh</p>
        </div>
        """)
        
        with gr.Row(elem_classes="main-container"):
            # Left sidebar - Sample questions with dropdowns
            with gr.Column(scale=2, min_width=280):
                gr.HTML('<div class="sidebar-header">💡 Câu hỏi mẫu</div>')
                
                # Sample questions as simple buttons instead of dropdowns
                sample_categories = chatbot.get_sample_questions()
                sample_buttons = []
                
                for category, questions in sample_categories.items():
                    gr.HTML(f'<div style="margin: 10px 0; font-weight: bold; color: #4285f4;">{category}</div>')
                    for question in questions[:2]:  # Limit to 2 questions per category
                        btn = gr.Button(
                            question[:40] + "..." if len(question) > 40 else question,
                            size="sm",
                            variant="secondary",
                            elem_classes="sample-question-btn"
                        )
                        sample_buttons.append((btn, question))
            
            # Center - Main chat interface (expanded)
            with gr.Column(scale=5, min_width=500):
                # Simplified chat interface
                chatbot_component = gr.Chatbot(
                    label="💬 Trợ lý Pháp lý",
                    type="messages",
                    elem_classes="chat-container-main",
                    height=500,
                    show_copy_button=True
                )
                
                # Enhanced input area
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Hỏi tôi về pháp luật Việt Nam...",
                        container=False,
                        scale=5,
                        lines=1,
                        elem_classes="main-input"
                    )
                    send_button = gr.Button("📤 Gửi", variant="primary", scale=1, elem_classes="send-button")
                
                # Control buttons
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Xóa cuộc trò chuyện", size="sm", variant="secondary")
            
            # Right sidebar - Reference documents (expanded)
            with gr.Column(scale=3, min_width=350):
                gr.Markdown("### 📎 Thêm tài liệu pháp lý\nPDF/DOCX: tối đa 10 MiB/tệp, 50 trang PDF, 3 tệp/phiên. PDF scan hoặc lỗi dấu: bật OCR bên dưới.\n\nKhi bấm xử lý, nội dung sẽ được gửi tới dịch vụ embedding, reranking và mô hình trả lời. Tài liệu riêng cho phiên này, tự hết hạn sau 24 giờ.", elem_classes="upload-help")
                upload_status = gr.Markdown("Chưa có tài liệu tải lên.", elem_classes="upload-status")
                upload_file = gr.File(label="Chọn tài liệu", file_types=['.pdf', '.docx'], type='filepath')
                upload_ocr = gr.Checkbox(label="PDF scan / sửa văn bản lỗi bằng OCR tiếng Việt", value=False)
                gr.Markdown("OCR đọc lại ảnh trang PDF, có thể mất đến 10 phút. Hãy kiểm tra số điều, ngày và nội dung với bản gốc. DOCX không dùng OCR.", elem_classes="upload-help")
                upload_button = gr.Button("📥 Xử lý tài liệu", size='sm')
                # Enhanced documents display
                docs_display = gr.Markdown(
                    value="📄 **Tài liệu tham khảo sẽ hiển thị ở đây**",
                    label="📚 Cơ sở pháp lý",
                    elem_classes="docs-display"
                )
        
        status_display = gr.Markdown("💬 Nhập câu hỏi để bắt đầu")
        session_id = gr.State(value=None)
        document_token = gr.State(value=None, time_to_live=86400)

        def handle_message(message, history, conversation_id, token=None):
            if not isinstance(conversation_id, str):
                conversation_id = uuid.uuid4().hex
            new_history, docs, status = chatbot.process_message(message, history, conversation_id, token)
            return new_history, "", docs, status, conversation_id

        def handle_clear_chat(token):
            try:
                chatbot.clear_documents(token)
            except requests.RequestException:
                raise gr.Error("Không xóa được tài liệu. Vui lòng thử lại; phiên vẫn được giữ.") from None
            return [], "", "💬 Nhập câu hỏi để bắt đầu", uuid.uuid4().hex, None, "Chưa có tài liệu tải lên.", None

        def handle_upload(filepath, token, force_ocr=False):
            if filepath:
                detail = "Đang OCR tiếng Việt từ ảnh trang và lập chỉ mục; có thể mất đến 10 phút." if force_ocr else "Đang đọc văn bản và lập chỉ mục. PDF đầu tiên có thể mất vài phút."
                yield ("⏳ **Đang xử lý tài liệu…**\n\n" + detail + " Chờ thông báo hoàn tất trước khi hỏi.",
                       token, gr.update(value="⏳ Đang xử lý…", interactive=False))
            status, next_token = chatbot.upload_document(filepath, token, force_ocr=force_ocr)
            yield status, next_token, gr.update(value="📥 Xử lý tài liệu", interactive=True)

        upload_button.click(handle_upload, inputs=[upload_file, document_token, upload_ocr],
                            outputs=[upload_status, document_token, upload_button], show_progress="hidden")

        # Wire up sample question buttons
        for btn, question in sample_buttons:
            btn.click(
                lambda q=question: q,
                outputs=[message_input]
            )
        
        # Wire up main events
        send_button.click(
            handle_message,
            inputs=[message_input, chatbot_component, session_id, document_token],
            outputs=[chatbot_component, message_input, docs_display, status_display, session_id]
        )
        
        message_input.submit(
            handle_message,
            inputs=[message_input, chatbot_component, session_id, document_token],
            outputs=[chatbot_component, message_input, docs_display, status_display, session_id]
        )
        
        clear_chat_btn.click(
            handle_clear_chat,
            inputs=[document_token],
            outputs=[chatbot_component, docs_display, status_display, session_id, document_token, upload_status, upload_file]
        )
    
    return interface

def main():
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    username = os.getenv("DEMO_USERNAME", "").strip()
    password = os.getenv("DEMO_PASSWORD", "")
    if (share or username or password) and (not username or len(password) < 12):
        raise ValueError("Public demos require DEMO_USERNAME and DEMO_PASSWORD (at least 12 characters).")
    interface = create_chatbot_interface()
    interface.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                     server_port=7860, share=share, show_api=False,
                     auth=(username, password) if username and password else None,
                     max_file_size="10mb")


if __name__ == "__main__":
    main()
