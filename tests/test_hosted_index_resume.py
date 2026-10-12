import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from index_hosted_corpus import pending_documents


class IndexResumeTests(unittest.TestCase):
    def test_reads_all_pages_and_excludes_only_saved_article_ids(self):
        store = SimpleNamespace(collection_name="hosted_test", client=MagicMock())
        saved_id = uuid.uuid5(uuid.NAMESPACE_URL, "hosted_test/saved")
        store.client.scroll.side_effect = [
            ([SimpleNamespace(id=saved_id)], "next-page"),
            ([SimpleNamespace(id=uuid.uuid4())], None),
        ]
        docs = [{"id": "saved", "content": "first"}, {"id": "pending", "content": "second"}]
        self.assertEqual(pending_documents(store, docs), [docs[1]])
        self.assertEqual(store.client.scroll.call_args_list[1].kwargs["offset"], "next-page")
        for call in store.client.scroll.call_args_list:
            self.assertFalse(call.kwargs["with_payload"])
            self.assertFalse(call.kwargs["with_vectors"])
        self.assertEqual(len(docs), 2)

    def test_empty_collection_keeps_all_articles(self):
        store = SimpleNamespace(collection_name="hosted_test", client=MagicMock())
        store.client.scroll.return_value = ([], None)
        docs = [{"id": "first", "content": "text"}]
        self.assertEqual(pending_documents(store, docs), docs)
