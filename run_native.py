"""GPU development run without Docker. Not a substitute for container validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluate import norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    parser.add_argument('questions', type=Path)
    parser.add_argument('--report', type=Path, default=Path('native-results.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(os.environ.get('MC3_OUTPUT_DIR', '/app/output'))
    started = time.perf_counter()
    init = subprocess.run([sys.executable, str(root / 'app/app.py'), '--index', str(args.corpus.resolve())],
        capture_output=True, text=True, timeout=600)
    indexing_seconds = time.perf_counter() - started
    if init.returncode:
        args.report.write_text(json.dumps({'status': 'index_failed', 'seconds': indexing_seconds,
            'stdout': init.stdout, 'stderr': init.stderr}, indent=2))
        raise SystemExit(init.stderr)
    results = []
    for question in json.loads(args.questions.read_text())['queries']:
        query_id = f'query_{question["n"]:02d}'
        target = output / f'{query_id}_output.json'
        target.unlink(missing_ok=True)
        t0 = time.perf_counter()
        try:
            process = subprocess.run([sys.executable, str(root / 'app/app.py'), '--corpus',
                str(args.corpus.resolve()), '--query-id', query_id, '--query', question['query']],
                capture_output=True, text=True, timeout=30)
            seconds = time.perf_counter() - t0
            if process.returncode:
                raise RuntimeError(process.stderr)
            answer = json.loads(target.read_text())
            accepted = {norm(question['expected_answer'])} | {norm(a) for a in question.get('answer_aliases', [])}
            good = norm(answer['answer']) in accepted
            cited = set(answer['citations']) == set(question['expected_citations'])
            results.append({'id': query_id, 'seconds': seconds, 'output': answer,
                'answer_correct': good, 'citations_correct': cited, 'points': 20 if good and cited else 0})
        except Exception as error:
            results.append({'id': query_id, 'seconds': time.perf_counter() - t0,
                            'error': str(error), 'points': 0})
    sys.path.insert(0, str(root / 'app'))
    from app import rpc
    report = {'scope': 'native GPU development test; not container validation or awarded League XP',
        'corpus': str(args.corpus.resolve()),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'app').glob('*.py')},
        'indexing_seconds': indexing_seconds, 'index_stdout': init.stdout,
        'score': sum(row['points'] for row in results), 'maximum': 200,
        'metrics': rpc({'op': 'metrics'}), 'results': results}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
