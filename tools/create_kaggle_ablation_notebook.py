"""Package lightweight ablations and verified baseline rankings for Kaggle."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
baseline = ROOT / 'results/qwen-gpu-workers-top5'
manifest = json.loads((baseline / 'manifest.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for line in (baseline / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows) == manifest['questions'] == 3196
# Preserve every retrieval record, question and label, without bundling answers.
compact = [{'question_id': r['question_id'], 'question': r['question'], 'expected_ids': r['expected_ids'],
            'arms': {'original_query': r['arms']['original_query']}} for r in rows]
payload = gzip.compress(json.dumps({'manifest': manifest, 'rows': compact,
                                   'original_questions_sha256': hashlib.sha256((baseline / 'questions.jsonl').read_bytes()).hexdigest()},
                                  ensure_ascii=False).encode(), mtime=0)
paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'], cwd=ROOT, text=True).splitlines()
buffer = io.BytesIO()
with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
    for name in sorted(set(paths)):
        if name == '.env' or name.startswith(('notebooks/','results/','index/','data/','artifacts/')) or name.endswith(('.zip','.ipynb')):
            continue
        path = ROOT / name
        if path.is_file():
            archive.writestr(name, path.read_bytes())
source = buffer.getvalue()
reference = json.loads((ROOT / 'notebooks/kaggle_qwen_trial.ipynb').read_text(encoding='utf-8'))
restore = next(''.join(c['source']) for c in reference['cells'] if c['cell_type'] == 'code' and 'EXPECTED_INPUTS =' in ''.join(c['source']))
server = next(''.join(c['source']) for c in reference['cells'] if c['cell_type'] == 'code' and 'import requests, tarfile, time, urllib.request' in ''.join(c['source']))
cells = []


def markdown(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})


def code(text, hidden=False):
    cells.append({'cell_type':'code','metadata':{'jupyter':{'source_hidden':True}} if hidden else {},
                  'source':text.strip().splitlines(keepends=True),'execution_count':None,'outputs':[]})


markdown('''# Matched top-5 retrieval ablations

Enable GPU and internet. Attach the existing `colab_benchmark_inputs.zip` Dataset, then Run All.
The verified 3,196-question baseline rankings are embedded; no result ZIP or API key is needed.
Runs BM25, dense, and hybrid without BGE. No Qwen, BGE loading, refinement, generation or re-embedding.
A single GPU is sufficient for the query embedding model. CPU BM25 searches are shared across methods.
README results stay unchanged. Exports include raw checkpoint records and a matched comparison table.
''')
code(f'''
from pathlib import Path
import base64, gzip, hashlib, io, json, os, shutil, subprocess, sys, zipfile
REPO = Path('/kaggle/working/vietnamese-legal-chatbot')
REPO.mkdir(parents=True, exist_ok=True)
source = base64.b64decode({base64.b64encode(source).decode()!r})
assert hashlib.sha256(source).hexdigest() == {hashlib.sha256(source).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(REPO)
os.chdir(REPO)
baseline_bytes = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(baseline_bytes).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
baseline_data = json.loads(gzip.decompress(baseline_bytes))
BASELINE = REPO / 'results/ablation-baseline'
BASELINE.mkdir(parents=True, exist_ok=True)
(BASELINE / 'manifest.json').write_text(json.dumps(baseline_data['manifest']), encoding='utf-8')
(BASELINE / 'questions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\\n' for r in baseline_data['rows']), encoding='utf-8')
(BASELINE / 'origin.json').write_text(json.dumps({{'original_questions_sha256': baseline_data['original_questions_sha256']}}), encoding='utf-8')
print('Current source and verified retrieval baseline restored.')
''', hidden=True)
markdown('## Dependencies and dataset')
code('''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, 'download_dataset.py'], check=True)
''')
markdown('## Restore existing indexes')
code(restore)
markdown('## Start Qdrant with the existing vectors')
code(server)
markdown('## Optionally restore ablation checkpoints\nAttach `ablation_top5_results.zip` through Add Input to resume. Kaggle may automatically unpack it; both layouts are supported. Existing working checkpoints take precedence.')
code('''
OUTPUT = REPO / 'results/ablation-top5'
checkpoint_zip = next(Path('/kaggle/input').rglob('ablation_top5_results.zip'), None)
if not OUTPUT.exists():
    if checkpoint_zip:
        with zipfile.ZipFile(checkpoint_zip) as archive:
            for name in archive.namelist():
                path = Path(name)
                assert name.startswith('ablation-top5/') and not path.is_absolute() and '..' not in path.parts and '\\\\' not in name and ':' not in name
            archive.extractall(REPO / 'results')
    else:
        checkpoint = next((p.parent for p in Path('/kaggle/input').rglob('records.jsonl')
                           if (p.parent / 'manifest.json').is_file()), None)
        if checkpoint:
            saved = json.loads((checkpoint / 'manifest.json').read_text())
            assert saved['protocol'] == 'fixed-candidates-top5-ablation-v1'
            OUTPUT.mkdir(parents=True)
            for name in ('manifest.json','records.jsonl','summary.json','comparison.md'):
                if (checkpoint / name).is_file():
                    shutil.copyfile(checkpoint / name, OUTPUT / name)
print('Ablation output:', OUTPUT)
''')
markdown('## Run three ablations\nDefault: all 3,196 saved questions at top-5. Set `--limit` to a smaller number for a smoke run and choose a separate output folder. Every completed method record is checkpointed. Reports refresh every 25 questions and at completion.')
code('''
subprocess.run([sys.executable, '-u', 'benchmark_ablation.py', '--baseline', 'results/ablation-baseline',
                '--methods', 'bm25', 'dense', 'hybrid', '--require-gpu', '--limit', '0',
                '--output', 'results/ablation-top5'], cwd=REPO, check=True)
from IPython.display import Markdown, display
display(Markdown((OUTPUT / 'comparison.md').read_text()))
''')
markdown('## Export full or partial checkpoints\nRun after completion or after stopping inference. Partial raw records remain resumable even if the last report has not refreshed. Download the ZIP before discarding the runtime.')
code('''
EXPORT = Path('/kaggle/working/ablation_top5_results.zip')
assert OUTPUT.is_dir(), 'No ablation checkpoint directory yet.'
with zipfile.ZipFile(EXPORT, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT.rglob('*'):
        if path.is_file():
            archive.write(path, 'ablation-top5/' + path.relative_to(OUTPUT).as_posix())
print('Download:', EXPORT)
''')
for index, cell in enumerate(cells):
    cell['id'] = f'ablation-{index}'
notebook = {'nbformat':4,'nbformat_minor':5,'cells':cells,
            'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}}
path = ROOT / 'notebooks/kaggle_retrieval_ablation.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print('Created:', path)
