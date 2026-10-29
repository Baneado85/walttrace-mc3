"""Create a small reviewable source release, excluding all model weights."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    files = [p for p in root.iterdir() if p.is_file() and (
        p.suffix in {'.py', '.md', '.json', '.txt'} or p.name in {'Dockerfile', '.dockerignore', '.gitignore', '.gitattributes'})
        and p.name != 'SOURCE_MANIFEST.json']
    files += [p for p in (root / 'app').iterdir() if p.is_file() and p.suffix in {'.py', '.txt'}]
    files += [root / 'models/qwen' / n for n in ['provenance.json', 'LICENSE', 'README.md']]
    files = sorted(p for p in files if p.exists())
    assert not any(p.suffix == '.safetensors' for p in files)
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = root.parent / 'mc3-rag-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for p in files + [root / 'SOURCE_MANIFEST.json']:
            output.write(p, 'mc3-rag/' + p.relative_to(root).as_posix())
        synthetic = root.parents[1] / 'work/mc3-holdout'
        for p in synthetic.rglob('*'):
            if p.is_file():
                output.write(p, 'fixtures/synthetic/' + p.relative_to(synthetic).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root.parent / 'mc3-rag-source.sha256').write_text(digest + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest, 'model_weights_included': False}))


if __name__ == '__main__':
    main()
