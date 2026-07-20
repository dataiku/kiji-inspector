FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

# Clone the (public) vLLM fork and check out the desired branch
ARG VLLM_BRANCH=hidden-states-inline-return-squashed
RUN git clone --branch "${VLLM_BRANCH}" --single-branch \
    https://github.com/Davidnet/vllm.git /opt/vllm

WORKDIR /opt/vllm

# PyTorch built against CUDA 12.9.
# Use --index-url (PRIMARY, not --extra-index-url) so torch, torchvision AND
# torchaudio all come from the cu129 index. With --extra-index-url the resolver
# picks PyPI's default (CUDA 13) torch, leaving a torch(cu13)/torchaudio(cu129)
# mismatch. Use the venv's own pip so later steps don't re-resolve/upgrade it.
#
# Pin torch to the EXACT version vLLM requires (bump when the branch changes) —
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

# Compile & install vLLM from source.
# Fast build on this box (nproc=16, 142 GiB RAM):
#   MAX_JOBS=16                -> use all cores; plenty of RAM headroom for the CUDA jobs
#   NVCC_THREADS=2             -> extra intra-nvcc parallelism for the heavy kernels
#   TORCH_CUDA_ARCH_LIST       -> GPU arches to compile kernels for. An empty/wrong value
#                                 causes "no kernel image is available for execution on the
#                                 device" at runtime. Default covers Ampere -> Blackwell:
#                                   8.0  = A100
#                                   8.9  = L4/L40(S)/RTX 4090
#                                   9.0  = H100/H200 (Hopper)
#                                   10.0 = B200 (datacenter Blackwell)
#                                   12.0 = RTX PRO 6000 / consumer Blackwell
#                                 Override for a narrower/wider set, e.g.
#                                   --build-arg TORCH_CUDA_ARCH_LIST=12.0
#   ccache cache mount         -> near-instant recompiles on subsequent builds
ENV MAX_JOBS=16
ENV NVCC_THREADS=2
ENV CMAKE_BUILD_TYPE=Release
ARG TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install -e . --no-build-isolation -c /opt/torch-constraints.txt

# Install Kiji Inspector from this checkout into the existing vLLM
# environment without changing its dependency versions. In particular, vLLM
# was compiled against the Torch version installed above.
WORKDIR /opt/kiji-inspector
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-deps \
    && python -c "import kiji_inspector, torch, vllm; print(f'kiji-inspector installed (torch={torch.__version__}, vllm={vllm.__version__})')"

# Development tooling. The plan's verification steps run `pytest`/`ruff` inside
# this image; install them explicitly (constrained so they can't touch the
# pinned torch stack) rather than relying on tools incidentally present from
# vLLM's build environment. Mirrors the [dependency-groups].dev group.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/torch-constraints.txt \
    "pytest>=7.0" "pytest-cov>=4.0" "ruff>=0.1.0"

# Pipeline runtime dependencies that kiji needs but that `pip install . --no-deps`
# above skips. Without these the full pipeline fails: pyarrow/pandas for pairs
# parquet I/O (steps 1/3/5), scipy for SAE feature-health stats (step 2).
#
# TODO(maintainer): this is a stopgap. Replace the `--no-deps` kiji install +
# this hand-picked list with a proper install of kiji's dependency groups (e.g.
# `uv pip install --group runtime` or drop `--no-deps` once pyproject pins are
# guaranteed compatible with the pinned torch/vLLM stack), so runtime deps stay
# in sync with pyproject.toml instead of being duplicated here.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/torch-constraints.txt \
    "pyarrow" "pandas" "scipy"

# Import check for the native hidden-states connector: fail the build early if
# the pinned vLLM revision does not ship the extract_hidden_states connector.
RUN python -c "\
from vllm.config.kv_transfer import KVTransferConfig; \
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as c; \
assert hasattr(c, 'load_hidden_states') and hasattr(c, 'cleanup_hidden_states'); \
print('hidden-states connector present')"
