"""Run official-format questions through fresh CLI processes in an existing container.

This evaluates sample accuracy and timing, not full compliance or hidden performance.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def norm(text):
    return re.sub(r'[\s.·_\-]', '', str(text)).upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('container')
    parser.add_argument('questions', type=Path)
    parser.add_argument('--report', type=Path, default=Path('sample-results.json'))
    args = parser.parse_args()
    rows = json.loads(args.questions.read_text(encoding='utf-8'))['queries']
    results = []
    for row in rows:
        query_id = f'query_{row["n"]:02d}'
        target = f'/app/output/{query_id}_output.json'
        # Remove old output so a failed process cannot accidentally reuse a prior answer.
        subprocess.run(['docker', 'exec', args.container, 'rm', '-f', target], check=True)
        started = time.perf_counter()
        try:
            process = subprocess.run(['docker', 'exec', args.container, 'python3', '/app/app.py',
                '--corpus', '/app/corpus', '--query-id', query_id, '--query', row['query']],
                capture_output=True, text=True, timeout=30)
            duration = time.perf_counter() - started
            if process.returncode:
                raise RuntimeError(process.stderr)
            output = subprocess.run(['docker', 'exec', args.container, 'cat', target],
                capture_output=True, text=True, check=True, timeout=5)
            answer = json.loads(output.stdout)
            aliases = [row['expected_answer']] + row.get('answer_aliases', [])
            good_answer = isinstance(answer.get('answer'), str) and norm(answer['answer']) in {norm(a) for a in aliases}
            citations = answer.get('citations')
            good_citations = isinstance(citations, list) and all(isinstance(c, str) for c in citations) and set(citations) == set(row['expected_citations'])
            results.append({'id': query_id, 'seconds': duration, 'output': answer,
                'answer_correct': good_answer, 'citations_correct': good_citations,
                'points': 20 if good_answer and good_citations else 0})
        except Exception as error:
            results.append({'id': query_id, 'seconds': time.perf_counter() - started,
                            'error': str(error), 'points': 0})
    report = {'scope': 'public sample only; not League XP or hidden grading',
              'total': sum(r['points'] for r in results), 'maximum': len(rows) * 20, 'results': results}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'score': report['total'], 'maximum': report['maximum'], 'report': str(args.report)}))


if __name__ == '__main__':
    main()
