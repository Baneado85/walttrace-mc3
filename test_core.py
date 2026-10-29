"""Independent local tests: retrieval, parsing, evidence validation, no GPU claims."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parent / 'app'))
from retrieval import Chunk, Retriever, build_index, validate_answer


class CoreTests(unittest.TestCase):
    def test_copied_source_object_is_still_verified(self):
        source = Chunk('fixes.csv', 'row 2', 'ticket: EVT-7382; fixed_in: 8.7.6')
        payload = {'answer': '8.7.6', 'supports': [{'path': source.path, 'text': source.text}]}
        self.assertEqual(validate_answer(payload, [source])['answer'], '8.7.6')
        payload['supports'][0]['text'] = 'ticket: EVT-7382; fixed_in: 9.9.9'
        self.assertEqual(validate_answer(payload, [source])['answer'], '')

    def test_prose_finds_snake_case_constant(self):
        chunks = [Chunk('queue.py', 'text', 'DEFAULT_QUEUE_TIMEOUT_S = 413')]
        self.assertEqual(Retriever(chunks).retrieve('default queue timeout in seconds')[0].path, 'queue.py')

    def test_exact_evidence_and_unknown_sources(self):
        evidence = [Chunk('specs/current.txt', 'text', 'Maximum voltage: 72 V')]
        result = validate_answer({'answer': '72', 'supports': [
            {'path': 'specs/current.txt', 'quote': 'Maximum voltage: 72 V'}]}, evidence)
        self.assertEqual(result['citations'], ['specs/current.txt'])
        self.assertEqual(validate_answer({'answer': '72', 'supports': [
            {'path': 'missing.txt', 'quote': 'Maximum voltage: 72 V'}]}, evidence)['answer'], '')

    def test_hallucinated_value_is_rejected(self):
        evidence = [Chunk('x.txt', 'text', 'Maximum voltage: 72 V')]
        self.assertEqual(validate_answer({'answer': '99', 'supports': [
            {'path': 'x.txt', 'quote': 'Maximum voltage: 72 V'}]}, evidence)['answer'], '')

    def test_withdrawn_only_answer_is_rejected(self):
        evidence = [Chunk('old_WITHDRAWN.txt', 'text', 'Maximum voltage: 90 V', True)]
        self.assertEqual(validate_answer({'answer': '90', 'supports': [
            {'path': 'old_WITHDRAWN.txt', 'quote': 'Maximum voltage: 90 V'}]}, evidence)['answer'], '')

    def test_multihop_identifier_is_retrieved(self):
        chunks = [Chunk('event.log', 'text', 'braking fault linked to EVT-7382'),
                  Chunk('fixes.csv', 'row 2', 'ticket: EVT-7382; resolution: release 8.7.6'),
                  Chunk('decoy.txt', 'text', 'braking specification and braking conditions')]
        found = Retriever(chunks).retrieve('Which release fixed the braking fault?')
        self.assertIn('fixes.csv', {c.path for c in found})

    def test_file_errors_do_not_abort_the_walk(self):
        scratch = Path.cwd().resolve() / 'work' / 'mc3-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp)
            (root / 'invalid.pdf').write_bytes(b'not a PDF')
            (root / 'unknown.bin').write_bytes(b'\x00\x01')
            (root / 'valid.txt').write_text('A novel part number: XP-827-NEW', encoding='utf-8')
            (root / 'empty').mkdir()
            chunks, skipped = build_index(root)
            self.assertEqual({c.path for c in chunks}, {'valid.txt'})
            self.assertEqual(len(skipped), 2)


if __name__ == '__main__':
    unittest.main()
