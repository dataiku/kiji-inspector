# syntax=docker/dockerfile:1
#
# Kiji Inspector + fork-built vLLM (hidden-states connector).
#
# Reproducibility policy: every external input is pinned —
#   * base images by digest (floating tags like uv's `latest` are re-pushed
#     upstream; a new digest invalidates EVERY later layer, including the
#     ~30-minute vLLM compile),
#   * the vLLM fork by commit SHA (a moving branch makes builds
#     non-reproducible across machines and hides what was actually built),
#   * torch by exact version, frozen into a constraints file that all later
#     installs must obey.
# Bump pins deliberately, ideally batched with changes that force a recompile
# anyway (e.g. a fork update).
FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04@sha256:5a480db8cbf90098ca816d7e77f07ce5cb4c43353530b6a84915c2dd99c4e0b6
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 /uv /uvx /bin/

# git for cloning, ccache to speed up (re)builds of the CUDA/C++ extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ccache \
    && rm -rf /var/lib/apt/lists/*

# Create the virtualenv (uv downloads & manages the 3.12 interpreter itself)
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv --python 3.12 --seed --managed-python /opt/venv

# Make the venv the default for all later RUN/CMD and interactive shells
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Fetch the (public) vLLM fork at an exact commit. Fetching the SHA directly
# (GitHub serves arbitrary reachable SHAs) instead of `git clone --branch`
# keeps the build identical even if the branch is later force-pushed or
# deleted. Current pin: tip of `hidden-states-inline-return-squashed`.
ARG VLLM_REPO=https://github.com/Davidnet/vllm.git
ARG VLLM_COMMIT=b6455d43be849d4850bed6ecfb834489ba9f0a08
RUN git init /opt/vllm \
    && git -C /opt/vllm remote add origin "${VLLM_REPO}" \
    && git -C /opt/vllm fetch --depth 1 origin "${VLLM_COMMIT}" \
    && git -C /opt/vllm checkout --detach FETCH_HEAD

WORKDIR /opt/vllm

# PyTorch built against CUDA 12.9.
# Use --index-url (PRIMARY, not --extra-index-url) so torch, torchvision AND
# torchaudio all come from the cu129 index. With --extra-index-url the resolver
# picks PyPI's default (CUDA 13) torch, leaving a torch(cu13)/torchaudio(cu129)
# mismatch. Use the venv's own pip so later steps don't re-resolve/upgrade it.
#
# Pin torch to the EXACT version vLLM requires (bump when VLLM_COMMIT changes) —
# the cu129 index's "latest" is newer than vLLM's pin and would conflict.
# torchvision/torchaudio are left unpinned so pip picks versions compatible
# with this torch.
ARG TORCH_VERSION=2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Freeze the exact torch stack (with +cu129 local versions) into a constraints
# file, so the build-reqs and editable installs below can't upgrade it.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /opt/torch-constraints.txt \
    && echo "Pinned torch stack:" && cat /opt/torch-constraints.txt

# vLLM build requirements, minus torch, constrained so the stack stays put.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^torch==' requirements/build/cuda.txt \
    | pip install -c /opt/torch-constraints.txt -r /dev/stdin

# Compile & install vLLM from source. This step deliberately stays on plain
# pip: `--no-build-isolation` + the constraints file is the recipe vLLM's own
# docs use, and it is the proven path for this fork.
#
# Sizing knobs (defaults fit a 16-core / 142 GiB box; override on smaller
# machines, e.g. --build-arg MAX_JOBS=4 — each CUDA job can peak at several
# GiB of RAM, so MAX_JOBS * NVCC_THREADS should stay well under RAM/4 GiB):
#   MAX_JOBS                   -> parallel compile jobs
#   NVCC_THREADS               -> extra intra-nvcc parallelism for heavy kernels
#   TORCH_CUDA_ARCH_LIST       -> GPU arches to compile kernels for. An empty/wrong value
#                                 causes "no kernel image is available for execution on the
#                                 device" at runtime. Default covers Ampere -> Blackwell:
#                                   8.0  = A100
#                                   8.9  = L4/L40(S)/RTX 4090
#                                   9.0  = H100/H200 (Hopper)
#                                   10.0 = B200 (datacenter Blackwell)
#                                   12.0 = RTX PRO 6000 / consumer Blackwell
#                                 Override for a narrower/wider set (much faster), e.g.
#                                   --build-arg TORCH_CUDA_ARCH_LIST=12.0
#   ccache cache mount         -> near-instant recompiles on subsequent builds
ARG MAX_JOBS=16
ARG NVCC_THREADS=2
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV CMAKE_BUILD_TYPE=Release
ARG TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install -e . --no-build-isolation -c /opt/torch-constraints.txt

# Install Kiji Inspector — runtime deps, the `full` extra (accelerate for the
# HF path), and the dev dependency group (pytest/ruff for in-image
# verification) — into the existing vLLM environment. The constraints file
# keeps the resolver off the torch stack vLLM was compiled against; everything
# else resolves from pyproject.toml, so the image stays in sync with the
# project metadata instead of duplicating a hand-picked package list.
# uv (not pip) here: `uv pip install --group` works on any uv version we pin,
# whereas pip only grew `--group` in 25.1, making the layer hostage to
# whatever pip `uv venv --seed` happened to ship.
WORKDIR /opt/kiji-inspector
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e '.[full]' --group dev -c /opt/torch-constraints.txt \
    && python -c "import kiji_inspector, torch, vllm; print(f'kiji-inspector installed (torch={torch.__version__}, vllm={vllm.__version__})')"

# Fail the build early if the pinned vLLM commit does not ship the
# extract_hidden_states connector, or if the torch stack drifted from the
# constraints frozen before the vLLM compile.
RUN python -c "\
from vllm.config.kv_transfer import KVTransferConfig; \
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as c; \
assert hasattr(c, 'load_hidden_states') and hasattr(c, 'cleanup_hidden_states'); \
print('hidden-states connector present')" \
    && pip check \
    && pip freeze | grep -E '^(torch|torchvision|torchaudio)==' | diff - /opt/torch-constraints.txt \
    && echo "torch stack matches constraints"
