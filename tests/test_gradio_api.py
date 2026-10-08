import unittest
from unittest.mock import MagicMock, patch

import requests
from fastapi.testclient import TestClient

from api import create_app
from app import VietnameseLegalChatbot, create_chatbot_interface


class GradioApiTests(unittest.TestCase):
    def setUp(self):
        self.chatbot = VietnameseLegalChatbot("http://api.test:8000/")
        self.result = {
            "answer": "Supported answer", "sources": [{
                "id": "law_1", "title": "Title", "law_id": "law", "article_id": "1",
                "excerpt": "Evidence", "score": 0.8}],
            "web_sources": [], "rejected_non_legal": False,
        }

    def response(self, status=200):
        response = MagicMock()
        response.status_code = status
        response.json.return_value = self.result
        if status >= 400:
            response.raise_for_status.side_effect = requests.HTTPError(response=response)
        return response

    def test_request_and_history_preserved(self):
        history = [{"role": "user", "content": "Earlier"},
                   {"role": "assistant", "content": "Earlier answer"}]
        with patch("app.requests.post", return_value=self.response()) as post:
            updated, docs, status = self.chatbot.process_message("  Question  ", history)
        post.assert_called_once_with("http://api.test:8000/ask",
                                     json={"question": "Question", "use_web_fallback": False},
                                     timeout=(5, 600))
        self.assertEqual(len(history), 2)
        self.assertEqual(updated[:2], history)
        self.assertEqual(updated[-1]["content"], "Supported answer")
        self.assertIn("Evidence", docs)
        self.assertIn("law_1", docs)
        self.assertIn("Hoàn thành", status)

    def test_empty_or_oversized_input_does_not_call_api(self):
        with patch("app.requests.post") as post:
            for question in ("  ", "x" * 2001):
                history, _, _ = self.chatbot.process_message(question, [])
                self.assertEqual(history, [])
        post.assert_not_called()

    def test_conversation_id_sent_as_header(self):
        with patch("app.requests.post", return_value=self.response()) as post:
            self.chatbot.process_message("q", [], "conversation-1")
        self.assertEqual(post.call_args.kwargs["headers"], {"X-Session-ID": "conversation-1"})

    def test_http_errors_are_friendly_and_not_retried(self):
        for code, expected in ((429, "câu hỏi khác"), (422, "không hợp lệ"), (502, "gặp lỗi")):
            with self.subTest(code=code), patch("app.requests.post", return_value=self.response(code)) as post:
                history, docs, _ = self.chatbot.process_message("q", [])
                self.assertIn(expected, history[-1]["content"])
                self.assertEqual(docs, "")
                post.assert_called_once()

    def test_connection_and_timeout_errors(self):
        for error, expected in ((requests.ConnectionError("secret"), "khởi động FastAPI"),
                                (requests.Timeout("secret"), "vẫn đang xử lý")):
            with patch("app.requests.post", side_effect=error):
                history, _, _ = self.chatbot.process_message("q", [])
                self.assertIn(expected, history[-1]["content"])
                self.assertNotIn("secret", history[-1]["content"])

    def test_bad_response_is_sanitized(self):
        response = self.response()
        response.json.side_effect = ValueError("private backend details")
        with patch("app.requests.post", return_value=response):
            history, _, _ = self.chatbot.process_message("q", [])
            self.assertIn("không hợp lệ", history[-1]["content"])
            self.assertNotIn("private", history[-1]["content"])

    def test_nonlegal_guidance(self):
        self.result["rejected_non_legal"] = True
        with patch("app.requests.post", return_value=self.response()):
            _, docs, _ = self.chatbot.process_message("q", [])
        self.assertIn("Hướng dẫn", docs)

    def test_source_html_is_escaped_and_web_urls_filtered(self):
        docs = self.chatbot._format_retrieved_documents(
            [{"title": '<script>alert(1)</script>', "excerpt": '<img src=x onerror="alert(1)">'}],
            [{"url": "javascript:alert(1)", "title": "bad"},
             {"url": "https://example.com", "title": "Web evidence"}])
        self.assertNotIn("<script>", docs)
        self.assertNotIn("<img", docs)
        self.assertNotIn("javascript:", docs)
        self.assertIn("&lt;script&gt;", docs)
        self.assertIn("https://example.com", docs)

    def test_real_api_contract_with_fake_inference(self):
        rag = MagicMock()
        rag.answer_question.return_value = {
            "answer": "API answer", "retrieved_documents": [{"id": "1", "content": "API evidence"}],
        }
        with TestClient(create_app(lambda: rag, lambda _: None)) as api_client:
            def post(url, json, timeout):
                return api_client.post("/ask", json=json)
            with patch("app.requests.post", side_effect=post):
                history, docs, _ = self.chatbot.process_message("q", [])
        self.assertEqual(history[-1]["content"], "API answer")
        self.assertIn("API evidence", docs)
        rag.answer_question.assert_called_once_with("q", use_fallback=False, refine_question=False, top_k=5)

    def test_interface_uses_messages_and_builds_without_network(self):
        with patch("app.requests.post") as post:
            interface = create_chatbot_interface()
        chat = next(c for c in interface.config["components"] if c["type"] == "chatbot")
        self.assertEqual(chat["props"]["type"], "messages")
        post.assert_not_called()
        interface.close()


if __name__ == "__main__":
    unittest.main()
