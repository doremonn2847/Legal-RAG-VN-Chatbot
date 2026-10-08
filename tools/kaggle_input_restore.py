"""Accept Kaggle's ZIP or automatically unpacked Dataset layout."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile, ZIP_STORED


def input_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def restore_kaggle_inputs(input_root, repo, expected):
    input_root, repo = Path(input_root), Path(repo)
    bundle = next(input_root.rglob('colab_benchmark_inputs.zip'), None)
    destination = repo / 'colab_vector_index.zip'
    if bundle:
        if input_digest(bundle) != expected['bundle_sha256']:
            raise ValueError('Index bundle checksum differs.')
        with ZipFile(bundle) as archive:
            if set(archive.namelist()) != {'bm25_index.pkl', 'colab_vector_index.zip'}:
                raise ValueError('Unexpected bundle contents.')
            (repo / 'index').mkdir(parents=True, exist_ok=True)
            with archive.open('bm25_index.pkl') as source, (repo / 'index/bm25_index.pkl').open('wb') as target:
                shutil.copyfileobj(source, target)
            archive.extract('colab_vector_index.zip', repo)
        print('Verified original ZIP input.')
        return destination

    bm25 = next((p for p in input_root.rglob('bm25_index.pkl') if input_digest(p) == expected['bm25_sha256']), None)
    if bm25 is None:
        raise ValueError('Matching BM25 index not found in the attached Kaggle Dataset.')
    vector_files = expected['vector_files']
    for name in vector_files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe expected vector path.')
    manifests = [p for p in input_root.rglob('build_manifest.json')
                 if input_digest(p) == vector_files['build_manifest.json']]
    vector_root = None
    for manifest in manifests:
        root = manifest.parent
        if all((root / name).is_file() and input_digest(root / name) == checksum
               for name, checksum in vector_files.items()):
            vector_root = root
            break
    if vector_root is None:
        raise ValueError('Complete matching vector files not found; the extracted index is incomplete or changed.')
    # Verify all input hashes before changing any destination files.
    (repo / 'index').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bm25, repo / 'index/bm25_index.pkl')
    with ZipFile(destination, 'w', compression=ZIP_STORED) as archive:
        for name in vector_files:
            archive.write(vector_root / name, name)
    print('Verified automatically extracted Kaggle input and rebuilt the portable archive.')
    return destination
