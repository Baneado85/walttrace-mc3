"""Build-time download only. Resolves and records an immutable model commit."""
import json
import os
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

model = 'Qwen/Qwen2.5-VL-7B-Instruct'
revision = os.environ.get('MC3_MODEL_REVISION', 'cc594898137f460bfe9f0759e9844b3ce807cfb5')
destination = Path(__file__).resolve().parent / 'models' / 'qwen'
snapshot_download(model, revision=revision, local_dir=destination,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'LICENSE', 'README.md'])
if not (destination / 'LICENSE').exists():
    with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt', timeout=45) as response:
        (destination / 'LICENSE').write_bytes(response.read())
(destination / 'provenance.json').write_text(json.dumps({'model': model, 'revision': revision}, indent=2))
print(json.dumps({'model': model, 'revision': revision, 'directory': str(destination)}))
