"""Download the public Apache-2.0 model for a local image build, without credentials."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

REPO = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
ROOT = Path(__file__).resolve().parent / 'models/qwen'


def main():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}', timeout=60) as response:
        info = json.load(response)
    revision = info['sha']
    names = [s['rfilename'] for s in info['siblings'] if '/' not in s['rfilename'] and (
        s['rfilename'].endswith(('.json', '.safetensors', '.txt', '.model', '.jinja')) or
        s['rfilename'] in {'LICENSE', 'README.md'})]
    ROOT.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        target = ROOT / name
        if target.exists():
            return {'file': name, 'bytes': target.stat().st_size, 'existing': True}
        temporary = target.with_suffix(target.suffix + '.part')
        url = f'https://huggingface.co/{REPO}/resolve/{revision}/{name}?download=true'
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
        temporary.replace(target)
        result = {'file': name, 'bytes': target.stat().st_size}
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, names))
    if not (ROOT / 'LICENSE').exists():
        with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt', timeout=45) as response:
            (ROOT / 'LICENSE').write_bytes(response.read())
    (ROOT / 'provenance.json').write_text(json.dumps({'model': REPO, 'revision': revision,
        'files': files}, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'revision': revision,
        'total_bytes': sum(f['bytes'] for f in files)}), flush=True)


if __name__ == '__main__':
    main()
