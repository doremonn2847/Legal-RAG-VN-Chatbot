"""Read-only localhost view of checkpointed benchmark progress."""
import argparse
from datetime import datetime, timedelta, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from string import Template
import subprocess


def snapshot(output, docker):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    files = list((output / 'questions').glob('*.json'))
    total, count = len(manifest['question_ids']), len(files)
    try:
        state = subprocess.check_output([docker, 'inspect', '--format', '{{.State.Status}}',
                                         'legal-hosted-benchmark'], timeout=5, text=True,
                                        stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        state = 'unknown'
    status = 'Complete' if count == total else {'running': 'Running', 'exited': 'Interrupted / stopped'}.get(state, 'Status unavailable')
    times = sorted(p.stat().st_mtime for p in files)[-50:]
    seconds = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
    eta = '—'
    if status == 'Running' and seconds:
        eta = f'{max(1, round((total-count)*seconds/60)):,} minutes'
    elif count == total:
        eta = 'Finished'
    try:
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        summary = {'completed_questions': 0, 'metrics': []}
    return dict(count=count, total=total, status=status, eta=eta, summary=summary)


def render(data):
    template = Path(__file__).with_name('indexing_progress.html').read_text(encoding='utf-8')
    template = template.replace('Legal corpus indexing', 'Retrieval benchmark').replace(
        'Remote BGE-M3 embeddings saved to Qdrant', 'BM25 · Hosted BGE-M3 · Hybrid · Hybrid + BGE reranking').replace(
        'articles', 'questions').replace('Articles remaining', 'Questions remaining')
    template = template.replace('<footer>', '<section aria-label="Results">$table</section><footer>')
    template = template.replace('</style>', 'main{max-width:1200px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}caption{text-align:left;padding:16px 0}</style>')
    summary = data['summary']
    columns = [('precision','P@5'),('recall','R@5'),('f1','F1@5'),('hit_rate','Hit@5'),
               ('map','MAP@5'),('mrr','MRR@5'),('ndcg','nDCG@5'),('mean_seconds','Mean s'),('p95_seconds','P95 s')]
    names = dict(bm25='BM25', dense='Dense', hybrid='Hybrid', hybrid_bge='Hybrid + BGE')
    table = '<div class="table-wrap"><table><caption>Latest results: ' + str(summary['completed_questions']) + ' questions scored</caption><thead><tr><th scope="col">Method</th>'
    table += ''.join(f'<th scope="col">{label}</th>' for _, label in columns) + '</tr></thead><tbody>'
    for metric in summary['metrics']:
        table += '<tr><th scope="row">' + escape(names.get(metric['method'], metric['method'])) + '</th>'
        table += ''.join(f'<td>{metric[key]:.4f}</td>' for key, _ in columns) + '</tr>'
    table += '</tbody></table></div><p>Training-set retrieval only; no answer generation. Final top-5 from 25 candidates per retriever. README results remain unchanged.</p>'
    count, total = data['count'], data['total']
    percent = count / total * 100 if total else 0
    return Template(template).substitute(count=f'{count:,}', total=f'{total:,}', remaining=f'{max(0,total-count):,}',
        percent=f'{percent:.1f}', percent_label=f'{percent:.1f}% complete', status=escape(data['status']),
        eta=escape(data['eta']), table=table,
        checked=datetime.now(timezone(timedelta(hours=7))).strftime('%H:%M:%S'),
        notice='Metrics update every 25 questions. Estimates use the latest 50 saved checkpoints; network interruptions can change them.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docker', required=True)
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/status'):
                self.send_error(404)
                return
            try:
                data = snapshot(args.output, args.docker)
                # Expose metrics only, never the manifest or provider configuration.
                body = (json.dumps(data) if self.path == '/status' else render(data)).encode('utf-8')
            except (OSError, ValueError, KeyError):
                self.send_error(503, 'Benchmark checkpoints are not available yet')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if self.path == '/status' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
