"""Build a full-corpus GPU benchmark notebook and a public-index upload bundle."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
bundle = ROOT / 'results/colab_benchmark_inputs.zip'
bundle.parent.mkdir(exist_ok=True)
with ZipFile(bundle, 'w', compression=ZIP_DEFLATED, compresslevel=3) as archive:
    archive.write(ROOT / 'colab_vector_index.zip', 'colab_vector_index.zip', compress_type=0)
    archive.write(ROOT / 'index/bm25_index.pkl', 'bm25_index.pkl')
bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
buffer = io.BytesIO()
with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
    for name in sorted(set(paths)):
        if name.startswith(('notebooks/', 'artifacts/')) or name == '.env' or name.endswith(('.zip', '.ipynb')):
            continue
        path = ROOT / name
        if path.is_file():
            archive.writestr(name, path.read_bytes())
source = buffer.getvalue()
source_sha = hashlib.sha256(source).hexdigest()
encoded = base64.b64encode(source).decode()
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text, hidden=False):
    cells.append({'cell_type': 'code', 'execution_count': None, 'outputs': [],
                  'metadata': {'jupyter': {'source_hidden': True}} if hidden else {},
                  'source': text.strip().splitlines(keepends=True)})


markdown('''# Full retrieval benchmark on Colab GPU — pending README review

Choose the existing Colab **GPU** kernel in VS Code, or open this notebook in Colab and choose a GPU runtime. Run the cells in order. This evaluates **all 3,196 labeled training questions**, four methods, and **final top 5 and top 10**, including nDCG. BGE runs once per question; both cutoffs share that ranking.

The source snapshot includes our uncommitted changes. No credentials are embedded. Upload the local file `results/colab_benchmark_inputs.zip` when prompted; it contains only the existing public-corpus vectors and BM25 index. There is no re-embedding. No query refinement or answer generation is performed; no API key is needed.

The README Results table is never edited. Download the final ZIP and return it to the local working directory for review. This is a full training-set evaluation, not a held-out accuracy claim.
''')
code(f'''
from pathlib import Path
import base64, hashlib, io, os, subprocess, zipfile, sys
SOURCE_SHA256 = {source_sha!r}
SOURCE_BASE64 = {encoded!r}
source = base64.b64decode(SOURCE_BASE64)
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
REPO = Path('/content/vietnamese-legal-chatbot')
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(REPO)
os.chdir(REPO)
print('Current source snapshot restored:', SOURCE_SHA256)
''', hidden=True)
markdown('## Install dependencies and prepare the same dataset')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, 'download_dataset.py'], check=True)
import torch
assert torch.cuda.is_available(), 'Connect a GPU runtime before continuing.'
print('GPU:', torch.cuda.get_device_name(0))
''')
markdown('## Upload the existing indexes\nUpload `results/colab_benchmark_inputs.zip` from the local project. In VS Code, you can instead upload it to `/content/colab_benchmark_inputs.zip` through the remote file browser before running this cell.')
code(f'''
INPUT = Path('/content/colab_benchmark_inputs.zip')
if not INPUT.exists():
    from google.colab import files
    uploaded = files.upload()
    name = 'colab_benchmark_inputs.zip'
    assert name in uploaded, 'Upload the exact benchmark input bundle.'
    candidate = REPO / name
    if candidate.exists():
        candidate.replace(INPUT)
    del uploaded
assert INPUT.exists(), 'Input bundle is not available on the runtime.'
with INPUT.open('rb') as handle:
    checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
assert checksum == {bundle_sha!r}, 'Input bundle checksum mismatch.'
with zipfile.ZipFile(INPUT) as archive:
    assert set(archive.namelist()) == {{'bm25_index.pkl', 'colab_vector_index.zip'}}
    Path('index').mkdir(exist_ok=True)
    with archive.open('bm25_index.pkl') as source, Path('index/bm25_index.pkl').open('wb') as target:
        import shutil
        shutil.copyfileobj(source, target)
    archive.extract('colab_vector_index.zip', REPO)
subprocess.run([sys.executable, 'import_vector_index.py', 'colab_vector_index.zip'], check=True,
               env={{**os.environ, 'QDRANT_PATH': 'index/qdrant', 'QDRANT_URL': '', 'QDRANT_API_KEY': ''}})
print('Verified existing vectors and BM25 index. No embedding rebuild.')
''')
markdown('## Start Qdrant 1.19.0 and transfer the existing vectors\nColab does not need Docker for this cell. It runs the same Qdrant server version as the local Docker container, using the verified official Linux binary. The embedded index is used only for transfer, avoiding large-collection local-mode search.')
code(r'''
import requests, tarfile, time, urllib.request
RUNTIME = REPO / 'qdrant-runtime'
RUNTIME.mkdir(exist_ok=True)
BINARY_DIR = Path('/tmp/legal-rag-qdrant-bin')
BINARY_DIR.mkdir(parents=True, exist_ok=True)
binary = BINARY_DIR / 'qdrant'
if not binary.exists():
    package = RUNTIME / 'qdrant.tar.gz'
    urllib.request.urlretrieve('https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-unknown-linux-gnu.tar.gz', package)
    assert hashlib.sha256(package.read_bytes()).hexdigest() == 'e4405091f67d02f96fb941695ef8a6974e677632507ff7b04a3fcbb332ad9c19'
    with tarfile.open(package) as archive:
        member = next(m for m in archive.getmembers() if m.isfile() and Path(m.name).name == 'qdrant')
        with archive.extractfile(member) as source, binary.open('wb') as target:
            shutil.copyfileobj(source, target)
binary.chmod(0o755)
config = RUNTIME / 'config.yaml'
config.write_text('storage:\n  storage_path: ' + str(RUNTIME / 'storage') + '\nservice:\n  host: 127.0.0.1\n  http_port: 6333\n  grpc_port: 6334\n', encoding='utf-8')
try:
    running = requests.get('http://localhost:6333', timeout=2).json()
except requests.RequestException:
    running = None
if running is None:
    server_log = (RUNTIME / 'server.log').open('ab')
    server = subprocess.Popen([str(binary), '--config-path', str(config)], cwd=RUNTIME,
                              stdout=server_log, stderr=subprocess.STDOUT)
    for _ in range(60):
        if server.poll() is not None:
            raise RuntimeError((RUNTIME / 'server.log').read_text()[-2000:])
        try:
            running = requests.get('http://localhost:6333', timeout=2).json()
            break
        except requests.RequestException:
            time.sleep(1)
assert running and running['version'] == '1.19.0', 'Qdrant server did not start with the expected version.'
os.environ['QDRANT_PATH'] = ''
os.environ['QDRANT_URL'] = 'http://localhost:6333'
os.environ['QDRANT_API_KEY'] = ''
subprocess.run([sys.executable, '-u', 'upload_vector_index.py'], check=True)
print('Qdrant server ready with the existing 61,068 article vectors.')
''')
markdown('## Choose checkpoint storage\nNo inference API key is needed. For a long run, set USE_GOOGLE_DRIVE = True to preserve checkpoints.')
code('''
USE_GOOGLE_DRIVE = False  # Set True to preserve checkpoints across runtime resets.
OUTPUT = REPO / 'results/benchmark-full-colab'
OUTPUT.parent.mkdir(exist_ok=True)
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    persistent = Path('/content/drive/MyDrive/vietnamese-legal-benchmarks/full-5-10')
    persistent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT.exists():
        OUTPUT.symlink_to(persistent, target_is_directory=True)
    assert OUTPUT.is_symlink(), 'Choose Drive storage before starting a local run; do not overwrite existing results.'
print('Checkpoint storage:', OUTPUT)
''')
markdown('## Run all questions and all methods\nThis cell uses GPU for embeddings and BGE, a BGE inference batch size of 8, the existing 25+25 candidate pool, and final cutoffs of 5/10. Original queries are used without refinement. It excludes answer generation and web search. A failed retrieval stops the run; it is not scored as a relevance miss. Rerun this cell to resume completed checkpoints.\n\nFull dataset runs can take hours even with GPU depending on corpus and GPU speed. Keep the runtime connected and use persistent checkpoints for longer runs.')
code('''
command = [sys.executable, '-u', 'benchmark_retrieval.py', '--limit', '0', '--seed', '42',
           '--threads', '2', '--reranker-batch-size', '8', '--require-gpu', '--no-refine',
           '--output', 'results/benchmark-full-colab']
subprocess.run(command, cwd=REPO, check=True)
summary = json.loads((OUTPUT / 'summary.json').read_text())
assert all(row['questions'] == 3196 for row in summary['metrics'])
assert len(summary['metrics']) == 8
from IPython.display import Markdown, display
display(Markdown((REPO / 'docs/benchmark-comparison.md').read_text()))
''')
markdown('## Download the report and raw results\nYou can run this cell after interrupting to download partial checkpoints too. Only a complete 3,196-question run should be considered for the README table. The ZIP contains no credentials, index weights, or environment files.')
code('''
EXPORT = REPO / 'colab_benchmark_results.zip'
with zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ('manifest.json', 'summary.json', 'questions.jsonl', 'queries.jsonl'):
        path = OUTPUT / name
        if path.exists():
            archive.write(path, 'benchmark-full-colab/' + name)
    report = REPO / 'docs/benchmark-comparison.md'
    if report.exists():
        archive.write(report, 'benchmark-comparison.md')
print('Download:', EXPORT, 'MB:', round(EXPORT.stat().st_size / 1024**2, 1))
from google.colab import files
files.download(str(EXPORT))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'benchmark-cell-{i}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'colab': {'name': 'colab_benchmark.ipynb'}, 'accelerator': 'GPU'}}
destination = ROOT / 'notebooks/colab_benchmark.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Notebook:', destination)
print('Upload bundle:', bundle, 'MB:', round(bundle.stat().st_size / 1024**2, 1))
print('Source snapshot SHA256:', source_sha)
