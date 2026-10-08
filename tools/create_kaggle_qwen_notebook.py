"""Package the current source for a Kaggle Qwen-only benchmark."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
bundle = ROOT / 'results/colab_benchmark_inputs.zip'
with bundle.open('rb') as handle:
    bundle_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
with ZipFile(ROOT / 'colab_vector_index.zip') as archive:
    vector_hashes = {}
    for name in archive.namelist():
        with archive.open(name) as handle:
            vector_hashes[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
with (ROOT / 'index/bm25_index.pkl').open('rb') as handle:
    bm25_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
expected_inputs = {'bundle_sha256': bundle_sha, 'bm25_sha256': bm25_sha, 'vector_files': vector_hashes}
restore_cell = (ROOT / 'tools/kaggle_input_restore.py').read_text(encoding='utf-8') + f'''

import os, subprocess, sys
EXPECTED_INPUTS = {expected_inputs!r}
archive_path = restore_kaggle_inputs('/kaggle/input', REPO, EXPECTED_INPUTS)
subprocess.run([sys.executable, 'import_vector_index.py', str(archive_path)], cwd=REPO, check=True,
               env={{**os.environ, 'QDRANT_PATH': 'index/qdrant', 'QDRANT_URL': '', 'QDRANT_API_KEY': ''}})
print('Existing indexes restored; no re-embedding.')
'''
(ROOT / 'tools/kaggle_restore_cell.py').write_text(restore_cell, encoding='utf-8')
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
buffer = io.BytesIO()
with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
    for name in sorted(set(paths)):
        if name.startswith(('notebooks/', 'artifacts/', 'results/', 'data/', 'index/')) or name == '.env' or name.endswith(('.zip', '.ipynb')):
            continue
        path = ROOT / name
        if path.is_file():
            archive.writestr(name, path.read_bytes())
payload = buffer.getvalue()
encoded = base64.b64encode(payload).decode()
checksum = hashlib.sha256(payload).hexdigest()
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text, hidden=False):
    cells.append({'cell_type': 'code', 'metadata': {'jupyter': {'source_hidden': True}} if hidden else {},
                  'source': text.strip().splitlines(keepends=True), 'execution_count': None, 'outputs': []})


markdown('''# Qwen 3.5 9B — Qwen-only Kaggle benchmark

Enable **GPU and internet** in Kaggle. Attach `colab_benchmark_inputs.zip` as a private Kaggle Dataset through **Add Input**. The source snapshot includes our current local changes, without credentials, indexes, or model weights.

This runs a new **full-dataset Qwen-only benchmark**. Retrieval uses the original question, 25 dense + 25 sparse candidates, BGE, and **top-5 only**. There are no refinement calls. Qwen receives top-5 evidence and a **2,048-token output cap**. Results include nDCG, token usage, finish reasons, latency, and retries. README results and the previous experiment stay unchanged; Qwen is now the application default. Set --limit to 50 in the run cell for a smaller sample, or keep 0 for all questions.

Expect model setup/download time before inference. Qwen weights are approximately 6.55 GB. On two T4 GPUs, two independent workers process separate questions concurrently. Each GPU hosts Qwen, embeddings, and BGE. A single GPU must have enough memory for both.
''')
code(f'''
from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, zipfile, shutil
REPO = Path('/kaggle/working/vietnamese-legal-chatbot')
REPO.mkdir(parents=True, exist_ok=True)
source = base64.b64decode({encoded!r})
assert hashlib.sha256(source).hexdigest() == {checksum!r}
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(REPO)
os.chdir(REPO)
print('Current source snapshot restored.')
''', hidden=True)
markdown('## Verify GPUs, install dependencies, and prepare the dataset')
code('''
gpu_probe = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, check=True)
GPUS = gpu_probe.stdout.strip().splitlines()
assert GPUS, 'Enable a Kaggle GPU accelerator.'
print(gpu_probe.stdout)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Notebook checks only; child workers pin their own GPUs.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', '-r', 'requirements-trial.txt'], check=True)
subprocess.run([sys.executable, 'download_dataset.py'], check=True)
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU.'
print('PyTorch GPU:', torch.cuda.get_device_name(0))
''')
markdown('## Restore the existing indexes from the attached input Dataset')
code(restore_cell)
markdown('## Start Qdrant 1.19.0 and load the same vectors')
colab = json.loads((ROOT / 'notebooks/colab_benchmark.ipynb').read_text(encoding='utf-8'))
server_cell = next(''.join(c['source']) for c in colab['cells'] if c['cell_type'] == 'code' and 'import requests, tarfile, time, urllib.request' in ''.join(c['source']))
code(server_cell)
markdown('## Start one Ollama server per GPU\nOn two T4 GPUs, each GPU runs its own Qwen, embeddings and BGE worker. Model weights are downloaded once and shared on disk. A single-GPU runtime runs one worker. The startup check rejects substantial CPU offloading or insufficient memory.')
code((ROOT / 'tools/kaggle_ollama_workers.py').read_text(encoding='utf-8'))
markdown('## Optionally restore Qwen-only checkpoints\nIf resuming, attach a previously exported `qwen_gpu_workers_top5_results.zip` through Add Input. Only this worker-protocol ZIP is restored; sequential and earlier refinement trials are kept separate. Worker count and all model/runtime settings must match to resume. Matching source, versions, model digest, and GPU are required. Checkpoints under `/kaggle/working` must be preserved through saved outputs or downloading; they do not automatically survive a discarded runtime.')
code('''
OUTPUT = REPO / 'results/qwen-gpu-workers-top5'
checkpoint_zip = next(Path('/kaggle/input').rglob('qwen_gpu_workers_top5_results.zip'), None)
if checkpoint_zip and not OUTPUT.exists():
    with zipfile.ZipFile(checkpoint_zip) as archive:
        for name in archive.namelist():
            path = Path(name)
            assert name.startswith('qwen-gpu-workers-top5/') and not path.is_absolute() and '..' not in path.parts and '\\\\' not in name
        archive.extractall(REPO / 'results')
    print('Trial checkpoint restored.')
print('Trial output:', OUTPUT)
''')
markdown('## Run all questions with Qwen only\nQwen makes one local answer call per question; no hosted provider or API key is used. Each worker runs BGE on the original question, returns five documents, then generates an answer. Workers process disjoint questions concurrently and keep isolated checkpoints. Qwen thinking is disabled, context is 8192, and the output cap is 2048. The runner stops on inference or retrieval failure instead of scoring failed calls. Rerun to resume cached model calls and completed questions.')
code('''
subprocess.run([sys.executable, '-u', 'benchmark_multi_gpu.py', '--workers', str(WORKERS), '--limit', '0', '--seed', '42',
                '--output', 'results/qwen-gpu-workers-top5'], cwd=REPO, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT / 'comparison.md').read_text()))
print('Review answer_review.csv for citation fidelity and completeness.')
''')
markdown('## Export complete or partial results\nRun this after completion, or after interrupting to preserve partial work. The exporter merges completed worker records before packaging both merged reports and per-worker checkpoints. Download the ZIP from the Kaggle Files panel and save a notebook version with outputs. Import that ZIP as a Dataset to resume later.')
code('''
subprocess.run([sys.executable, 'benchmark_multi_gpu.py', '--workers', str(WORKERS), '--merge-only',
                '--output', 'results/qwen-gpu-workers-top5'], cwd=REPO, check=True)
EXPORT = Path('/kaggle/working/qwen_gpu_workers_top5_results.zip')
with zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT.rglob('*'):
        if path.is_file() and path.suffix != '.tmp':
            archive.write(path, 'qwen-gpu-workers-top5/' + path.relative_to(OUTPUT).as_posix())
print('Download from Kaggle Files:', EXPORT)
''')
for i, cell in enumerate(cells):
    cell['id'] = f'qwen-trial-{i}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}}}
destination = ROOT / 'notebooks/kaggle_qwen_trial.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Created Kaggle pilot:', destination)
