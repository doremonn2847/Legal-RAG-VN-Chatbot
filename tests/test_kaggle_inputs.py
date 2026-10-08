import hashlib
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from tools.kaggle_input_restore import input_digest, restore_kaggle_inputs


class KaggleInputTests(unittest.TestCase):
    def make_input(self, root):
        dataset = root / 'inputs/dataset'
        vectors = dataset / 'colab_vector_index'
        files = {'build_manifest.json': b'{"article_count":61068}',
                 'qdrant/meta.json': b'{}',
                 'qdrant/collection/test/storage.sqlite': b'vector database'}
        for name, data in files.items():
            path = vectors / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        bm25 = dataset / 'bm25_index.pkl'
        bm25.write_bytes(b'BM25 index')
        expected = {'bm25_sha256': input_digest(bm25), 'bundle_sha256': '',
                    'vector_files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        return dataset, vectors, expected

    def test_kaggle_unpacks_outer_and_inner_zips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, vectors, expected = self.make_input(root)
            result = restore_kaggle_inputs(root / 'inputs', root / 'repo', expected)
            self.assertEqual(input_digest(root / 'repo/index/bm25_index.pkl'), expected['bm25_sha256'])
            with ZipFile(result) as archive:
                self.assertEqual(set(archive.namelist()), set(expected['vector_files']))
                self.assertEqual(archive.read('qdrant/collection/test/storage.sqlite'), b'vector database')

    def test_changed_vectors_do_not_replace_previous_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, vectors, expected = self.make_input(root)
            (vectors / 'qdrant/meta.json').write_text('changed')
            old = root / 'repo/index/bm25_index.pkl'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'previous')
            with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
                restore_kaggle_inputs(root / 'inputs', root / 'repo', expected)
            self.assertEqual(old.read_bytes(), b'previous')

    def test_original_zip_still_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, vectors, expected = self.make_input(root)
            bundle = dataset / 'colab_benchmark_inputs.zip'
            with ZipFile(bundle, 'w') as archive:
                archive.writestr('bm25_index.pkl', b'BM25 index')
                archive.writestr('colab_vector_index.zip', b'portable archive')
            expected['bundle_sha256'] = input_digest(bundle)
            result = restore_kaggle_inputs(root / 'inputs', root / 'repo', expected)
            self.assertEqual(result.read_bytes(), b'portable archive')

    def test_incomplete_extracted_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, vectors, expected = self.make_input(root)
            (vectors / 'qdrant/meta.json').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
                restore_kaggle_inputs(root / 'inputs', root / 'repo', expected)
