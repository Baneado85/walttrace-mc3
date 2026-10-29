FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
# Constrain the existing GPU wheels rather than letting pip replace them.
RUN python3 -c "import torch,importlib.metadata as m; assert torch.version.hip; open('/tmp/torch-constraints.txt','w').write('\n'.join(n+'=='+m.version(n) for n in ['torch','torchvision']))" \
 && pip install --no-cache-dir -c /tmp/torch-constraints.txt -r /app/requirements.txt \
 && python3 -c "import torch; assert torch.version.hip, torch.__version__"
COPY app/ /app/
COPY models/qwen/*.json /models/qwen/
COPY models/qwen/merges.txt models/qwen/README.md models/qwen/LICENSE /models/qwen/
# Separate unchanged weight shards allow parallel registry uploads.
COPY models/qwen/model-00001-of-00005.safetensors /models/qwen/
COPY models/qwen/model-00002-of-00005.safetensors /models/qwen/
COPY models/qwen/model-00003-of-00005.safetensors /models/qwen/
COPY models/qwen/model-00004-of-00005.safetensors /models/qwen/
COPY models/qwen/model-00005-of-00005.safetensors /models/qwen/
RUN mkdir -p /app/corpus /app/output /app/index
ENV MC3_MODEL_DIR=/models/qwen HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
CMD ["sleep", "infinity"]
