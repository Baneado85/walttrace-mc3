# WaltTrace — Offline Multimodal RAG

Submission candidate for AMD AI League Mini-Challenge 3, prepared for WaltTeam.
This is an actual retrieval and GPU generation implementation, not a question/answer lookup.
No League points or hidden-evaluation result have been awarded or claimed by this project.

## Design

The mandated ROCm PyTorch image contains Qwen2.5-VL-7B-Instruct in bf16, pinned to
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. The model license is Apache-2.0;
see the [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
Weights are downloaded during preparation and copied into the image. Evaluation
requires no outbound network, hosted API, API key or other credentials.

Indexing reads PDF, DOCX, XLSX, CSV, text, log and Python files as data. The GPU
vision model transcribes image files and scanned PDF pages. Per-file failures are
recorded and skipped; encrypted PDFs are never decrypted. Source code is never
executed. Unreadable files and symlinks are skipped. BM25 retrieval follows
discovered identifiers for multi-document questions and splits identifier words
so prose queries can find code constants. Withdrawn documents are penalized in
retrieval and cannot supply accepted support.

The model returns a value and supporting excerpts. Each excerpt must match text
from its stated indexed file, and the answer must occur in the accepted evidence.
Unsupported output becomes an empty answer and citation list. This check reduces
fabricated citations; it does not prove semantic correctness or citation necessity.
Confidence 0.8 is a fixed indicator for accepted evidence, not calibrated probability.

A private Unix socket keeps the model resident after `--index`. Each question is
a fresh lightweight CLI process; there is no TCP server or exposed port. A real
ROCm GPU is mandatory: there is no CPU or dummy-allocation fallback.

## Build

From this directory, with Python and Docker available:

```sh
python fetch_model_stdlib.py
docker build -t walttrace-mc3:review .
```

The download helper only requests public model files, with no login credentials.
The model is about 16.6 GB on disk. A large amount of free disk space is needed
for the mandated base, build layers and weights. Do not squash the image: the
organizer checks base-layer identity. The `models/` directory is intentionally
excluded from the source ZIP; the prepared image includes the complete weights.

## Runtime contract

On an AMD ROCm Linux host, create a container with the GPU devices and the corpus
mounted read-only. Adapt the host paths to the actual machine:

```sh
docker run -d --name mc3-review --network none --cap-drop DAC_OVERRIDE \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --shm-size=8g \
  -v /absolute/path/corpus:/app/corpus:ro walttrace-mc3:review
docker exec mc3-review python3 /app/app.py --index /app/corpus
docker exec mc3-review python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query 'What is the maximum junction temperature?'
docker exec mc3-review cat /app/output/query_01_output.json
```

The output is JSON with `answer`, exact corpus-relative `citations`, and optional
`confidence`. Environment overrides exist for development: `MC3_MODEL_DIR`,
`MC3_INDEX_DIR`, `MC3_OUTPUT_DIR`, and `MC3_SOCKET`. `MC3_DEBUG=1` logs generated
support objects for local debugging; it is disabled in the submission image.

## Verification and limits

`python -m unittest discover -s . -p test_core.py` checks malformed-file isolation,
retrieval of code constants, multi-hop retrieval and evidence rejection. It does
not claim GPU or container compliance.

`run_native.py CORPUS QUESTIONS --report report.json` measures a development run
on a real GPU without Docker, with a fresh CLI process per question. Reports
contain individual outputs, timings, source hashes, and PyTorch peak memory.
PyTorch allocated/reserved peaks are not continuous device-wide VRAM sampling.

`evaluate.py CONTAINER QUESTIONS --report report.json` evaluates an already indexed
Docker container. The official starter kit's `selfcheck.py` must also be used on
an AMD GPU. The public example has ten questions worth 200 technical test points;
these are different from the advertised 1,500 League points.

Initial native public-sample run: 200/200, cold model + index 40.24 seconds,
slowest question 4.25 seconds, peak allocated 15.94 GiB. See `native-results.json`.
The first synthetic development set exposed real failures (140/200); the next
revision scored 160/200. They are retained as `holdout-results.json` and
`holdout-v2.json` for transparency. That set has since been used for debugging,
so subsequent success on it must not be described as unseen-test performance.

The final local image is **44.8043 GiB uncompressed**, preserves the mandated
base layers, and passed seven core tests and processor/model-file checks with
`--network none`. The code hashes inside the image match the final native GPU
reports. See `container-validation.json` and `VALIDATION.md` for exact evidence.

Final revised native public and synthetic regression runs each scored 200/200.
Actual execution of the final container on an AMD GPU remains unverified: the
local Docker host has no AMD GPU, and the free GPU notebook has no container
runtime. Source repository: <https://github.com/Baneado85/walttrace-mc3>.
Public image: [Docker Hub](https://hub.docker.com/r/walternavarrodev/walttrace-mc3).
Pull with `docker pull walternavarrodev/walttrace-mc3:match3`.
Its digest is `sha256:19bf6afb0ad28f88b5f185775f246bcc9527593fa0943c2051965565d2034c93`.
Anonymous access, all 24 content blobs and unchanged official base blob identities
were verified after publication. Lablab confirmed the
[WaltTeam submission](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge/waltteam/walttrace-offline-multimodal-rag).
Hidden-corpus accuracy and official League credit remain unverified until
organizer evaluation; native development scores do not represent awarded XP.
