#!/usr/bin/env python3
"""MC3 CLI contract plus private Unix-socket model daemon."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import subprocess
import sys
import time
import traceback

from retrieval import Chunk, Retriever, build_index, save_index, validate_answer

INDEX = Path(os.environ.get('MC3_INDEX_DIR', '/app/index'))
OUTPUT = Path(os.environ.get('MC3_OUTPUT_DIR', os.environ.get('MC2_OUTPUT_DIR', '/app/output')))
SOCKET = os.environ.get('MC3_SOCKET', '/tmp/mc3-rag.sock')


def rpc(request: dict, timeout: float = 27) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(SOCKET)
        connection.sendall(json.dumps(request).encode() + b'\n')
        with connection.makefile('rb') as reader:
            result = reader.readline(4 * 1024 * 1024)
        if not result:
            raise RuntimeError('daemon closed without a response')
        response = json.loads(result)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response


def ensure_server():
    try:
        rpc({'op': 'ping'}, 2)
        return
    except (OSError, RuntimeError, ValueError):
        pass
    INDEX.mkdir(parents=True, exist_ok=True)
    log = (INDEX / 'daemon.log').open('a', encoding='utf-8')
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve'],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    started = time.monotonic()
    while time.monotonic() - started < 550:
        if process.poll() is not None:
            raise RuntimeError(f'model daemon failed; inspect {INDEX / "daemon.log"}')
        try:
            rpc({'op': 'ping'}, 2)
            return
        except (OSError, RuntimeError, ValueError):
            time.sleep(.5)
    raise TimeoutError('model initialization exceeded startup budget')


def serve():
    from backend import Model
    model = Model()
    state = {'retriever': None, 'corpus': None}
    if Path(SOCKET).exists():
        Path(SOCKET).unlink()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(4 * 1024 * 1024))
                if request['op'] == 'ping':
                    response = {'ready': True}
                elif request['op'] == 'metrics':
                    response = {'peak_allocated_bytes': model.torch.cuda.max_memory_allocated(),
                                'peak_reserved_bytes': model.torch.cuda.max_memory_reserved(),
                                'gpu': model.torch.cuda.get_device_name(0),
                                'torch': model.torch.__version__, 'hip': model.torch.version.hip}
                elif request['op'] == 'index':
                    corpus = Path(request['corpus']).resolve()
                    chunks, skipped = build_index(corpus, model.read_image)
                    save_index(INDEX, corpus, chunks, skipped)
                    state.update(retriever=Retriever(chunks), corpus=str(corpus))
                    response = {'indexed_chunks': len(chunks), 'skipped': skipped}
                elif request['op'] == 'query':
                    corpus = str(Path(request['corpus']).resolve())
                    if state['retriever'] is None:
                        data = json.loads((INDEX / 'index.json').read_text(encoding='utf-8'))
                        state.update(retriever=Retriever([Chunk(**c) for c in data['chunks']]), corpus=data['corpus'])
                    if corpus != state['corpus']:
                        raise ValueError('query corpus differs from indexed corpus')
                    chunks = state['retriever'].retrieve(request['query'])
                    payload = model.answer(request['query'], chunks) if chunks else {}
                    if os.environ.get('MC3_DEBUG') == '1':
                        print(json.dumps({'query': request['query'], 'payload': payload}, ensure_ascii=False), flush=True)
                    response = validate_answer(payload, chunks)
                else:
                    raise ValueError('unknown operation')
            except Exception as error:
                traceback.print_exc()
                response = {'error': f'{type(error).__name__}: {error}'}
            try:
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode() + b'\n')
            except BrokenPipeError:
                pass

    with socketserver.UnixStreamServer(SOCKET, Handler) as server:
        os.chmod(SOCKET, 0o600)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--query-id')
    parser.add_argument('--query')
    args = parser.parse_args()
    if args.serve:
        serve()
    elif args.index:
        ensure_server()
        print(json.dumps(rpc({'op': 'index', 'corpus': str(args.index.resolve())}, 580)))
    else:
        if args.corpus is None or args.query is None or not args.query_id:
            parser.error('query requires --corpus --query-id --query')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.query_id) or args.query_id in {'.', '..'}:
            parser.error('unsafe query id')
        # No heavyweight imports or model reloads in this per-question process.
        answer = rpc({'op': 'query', 'corpus': str(args.corpus.resolve()), 'query': args.query})
        OUTPUT.mkdir(parents=True, exist_ok=True)
        target = OUTPUT / f'{args.query_id}_output.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(answer, ensure_ascii=False), encoding='utf-8')
        temporary.replace(target)


if __name__ == '__main__':
    main()
