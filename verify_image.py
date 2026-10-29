"""Verify local image lineage, size and offline CPU checks; no GPU claim."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import threading

BASE = 'rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0'


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], capture_output=True, text=True, check=True, **kwargs)


def history_sizes(image_id):
    # The containerd image store reports compressed descriptor sizes in inspect.
    # History's integer Size fields measure the unpacked layer deltas instead.
    request = (f'GET /images/{image_id}/history HTTP/1.1\r\n'
               'Host: docker\r\nConnection: close\r\n\r\n').encode('ascii')
    process = subprocess.Popen(['docker', 'system', 'dial-stdio'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(60, process.kill)
    timer.start()
    process.stdin.write(request)
    process.stdin.flush()

    class Socket:
        def makefile(self, mode):
            return process.stdout

    try:
        response = http.client.HTTPResponse(Socket())
        response.begin()
        if response.status != 200:
            raise RuntimeError(f'Docker history API: {response.status}: {response.read()!r}')
        return [int(row['Size']) for row in json.loads(response.read())]
    finally:
        timer.cancel()
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', default='walttrace-mc3:review', nargs='?')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = json.loads(docker('image', 'inspect', BASE).stdout)[0]
    image = json.loads(docker('image', 'inspect', args.image).stdout)[0]
    base_layers = base['RootFS']['Layers']
    lineage = image['RootFS']['Layers'][:len(base_layers)] == base_layers
    assert lineage, 'Mandated base layers are not preserved'
    layer_sizes = history_sizes(image['Id'])
    uncompressed_size = sum(layer_sizes)
    assert uncompressed_size > image['Size'], 'Unpacked size measurement is not credible'
    assert uncompressed_size <= 60 * 1024**3, 'Uncompressed image exceeds 60 GiB'
    code = r'''
import hashlib,json,importlib.metadata as m,os
from pathlib import Path
import torch
from transformers import AutoProcessor
from backend import Model
assert torch.version.hip
assert os.environ['HF_HUB_OFFLINE']=='1'
model=Path('/models/qwen')
index=json.loads((model/'model.safetensors.index.json').read_text())
shards=set(index['weight_map'].values())
assert len(shards)==5 and all((model/name).is_file() for name in shards)
assert not list(model.glob('*.part'))
processor=AutoProcessor.from_pretrained(model,local_files_only=True)
guard=None
if not torch.cuda.is_available():
    try:
        Model()
    except RuntimeError as error:
        assert 'Real ROCm GPU is required' in str(error)
        guard=str(error)
    else:
        raise AssertionError('CPU fallback was allowed')
print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,
 'cuda_available':torch.cuda.is_available(),'cpu_guard':guard,
 'versions':{p:m.version(p) for p in ['transformers','tokenizers','openpyxl','pypdf','pypdfium2']},
 'model_shards':len(shards),'model_revision':json.loads((model/'provenance.json').read_text())['revision'],
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/app').glob('*.py')}}))
'''
    result = docker('run', '--rm', '--network', 'none', '--cap-drop', 'DAC_OVERRIDE',
                    args.image, 'python3', '-c', code, timeout=180)
    offline = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'app').glob('*.py')}
    assert offline['source_sha256'] == expected, 'Image source differs from reviewed source'
    gpu = json.loads((root / 'final-gpu-evidence.json').read_text())
    assert all(row['source_sha256'] == expected for row in gpu.values())
    tests = docker('run', '--rm', '--network', 'none', '--cap-drop', 'DAC_OVERRIDE',
                   '-e', 'PYTHONPATH=/app', '-v', f'{root / "test_core.py"}:/verify/test_core.py:ro',
                   args.image, 'python3', '/verify/test_core.py', timeout=90)
    report = {'scope': 'Local container lineage, size and offline CPU checks; GPU tested natively only',
        'image_tag': args.image, 'image_id': image['Id'], 'base_id': base['Id'],
        'base_repo_digests': base['RepoDigests'], 'base_layers_preserved': lineage,
        'compressed_image_bytes_docker_inspect': image['Size'],
        'uncompressed_layer_sizes_history_api': layer_sizes,
        'uncompressed_image_bytes': uncompressed_size, 'uncompressed_image_gib': uncompressed_size / 1024**3,
        'offline_network': 'none', 'offline_checks': offline,
        'unit_tests': tests.stdout + tests.stderr, 'gpu_reports_match_image_source': True,
        'actual_image_gpu_execution': 'NOT VERIFIED: local host lacks AMD GPU; notebook lacks container runtime',
        'published': False, 'competition_submitted': False, 'league_points_awarded': None}
    (root / 'container-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
