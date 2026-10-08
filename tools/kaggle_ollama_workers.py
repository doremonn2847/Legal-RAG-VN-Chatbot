"""Notebook cell: one owned Ollama server per GPU, shared on-disk weights."""
import time
import requests

if not shutil.which('ollama'):
    if not shutil.which('zstd'):
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'zstd'], check=True)
    subprocess.run(['bash', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'], check=True)

WORKERS = min(2, len(GPUS))
OLLAMA_SERVERS = globals().get('OLLAMA_SERVERS', {})
for index in range(WORKERS):
    host = f'127.0.0.1:{11435 + index}'
    url = f'http://{host}'
    process = OLLAMA_SERVERS.get(index)
    if process is None or process.poll() is not None:
        try:
            requests.get(url + '/api/version', timeout=2).raise_for_status()
        except requests.RequestException:
            pass
        else:
            raise RuntimeError(f'{host} is occupied by an unowned server; restart the runtime.')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index), 'OLLAMA_HOST': host,
               'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_MAX_LOADED_MODELS': '1',
               'OLLAMA_KEEP_ALIVE': '-1'}
        with (REPO / f'ollama-gpu-{index}.log').open('ab') as handle:
            process = subprocess.Popen(['ollama', 'serve'], env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
        OLLAMA_SERVERS[index] = process
    ready = False
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((REPO / f'ollama-gpu-{index}.log').read_text()[-2000:])
        try:
            requests.get(url + '/api/version', timeout=2).raise_for_status()
            ready = True
            break
        except requests.RequestException:
            time.sleep(1)
    assert ready, f'Ollama GPU {index} did not start; inspect its log.'
    print(f'GPU {index}: dedicated Ollama server {url}')

# Both servers read the same model store; download weights only once.
subprocess.run(['ollama', 'pull', 'qwen3.5:9b'], check=True,
               env={**os.environ, 'OLLAMA_HOST': '127.0.0.1:11435'})
for index in range(WORKERS):
    url = f'http://127.0.0.1:{11435 + index}'
    response = requests.get(url + '/api/tags', timeout=15)
    response.raise_for_status()
    assert any(m['name'] == 'qwen3.5:9b' for m in response.json()['models']), 'Shared Qwen weights unavailable'
print(f'{WORKERS} independent GPU worker(s) ready; combined memory checks run before question processing.')
