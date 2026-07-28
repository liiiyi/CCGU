#!/usr/bin/env bash
# Create the isolated, pinned CCGU/CGE reproduction environment (no sudo needed).
#
#   bash reproduction/scripts/setup_env.sh
#
# Creates a conda prefix at <repo>/.conda-env (git-ignored) and installs the
# pinned wheels from reproduction/requirements-lock.txt.  Idempotent: re-running
# it on an existing prefix only re-checks the pins.
#
# See README.md -> "Installation".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${CCGU_ENV_PREFIX:-${REPO_ROOT}/.conda-env}"

# `conda` is a shell *function* in interactive shells, so resolve a real binary.
# Override with CONDA_BIN=/path/to/conda if your installation lives elsewhere.
if [ -z "${CONDA_BIN:-}" ]; then
  for candidate in /opt/miniconda3/bin/conda "${HOME}/miniconda3/bin/conda" \
                   "${HOME}/anaconda3/bin/conda"; do
    if [ -x "${candidate}" ]; then CONDA_BIN="${candidate}"; break; fi
  done
fi
if [ ! -x "${CONDA_BIN:-}" ]; then
  echo "[setup] ERROR: no executable conda binary found; set CONDA_BIN=/path/to/conda" >&2
  exit 1
fi

echo "[setup] repo      : ${REPO_ROOT}"
echo "[setup] env prefix: ${ENV_PREFIX}"
echo "[setup] conda     : ${CONDA_BIN}"

# Python 3.8.19 matches the version reported in the paper's appendix.
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.8.19 pip
fi

PY="${ENV_PREFIX}/bin/python"
"${PY}" -m pip install --quiet --upgrade "pip==24.0" "wheel==0.43.0" "setuptools==69.5.1"

# torch first, from the CUDA 12.1 index, so DGL resolves against it.
"${PY}" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.2+cu121"

# DGL CUDA 12.1 build.  NOTE: the CUDA wheels for DGL 2.1.0 (the version named in
# the paper) are no longer published on data.dgl.ai -- that index now starts at
# 2.2.0 -- so we pin the nearest published CUDA build, 2.2.1+cu121.
"${PY}" -m pip install "dgl==2.2.1+cu121" \
  -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

"${PY}" -m pip install -r "${REPO_ROOT}/reproduction/requirements-lock.txt"

# ---------------------------------------------------------------------------
# DGL >= 2.2 dlopens libcudart/libcublas/libcusparse .so.12.  torch 2.1.2 ships
# only some of them inside torch/lib (and libcudart under a hashed filename), so
# `import dgl` fails with "libcusparse.so.12: cannot open shared object file".
# libdgl.so has RUNPATH=$ORIGIN, so symlinking the nvidia-*-cu12 libraries next
# to it resolves them for every interpreter invocation without LD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ -d "${SITE_PACKAGES}/nvidia" ]; then
  echo "[setup] linking CUDA runtime libraries next to libdgl.so"
  find "${SITE_PACKAGES}/nvidia" -name '*.so.12' -o -name '*.so.12.*' | while read -r lib; do
    ln -sfn "${lib}" "${SITE_PACKAGES}/dgl/$(basename "${lib}")"
  done
fi

echo "[setup] verifying"
"${PY}" "${REPO_ROOT}/reproduction/scripts/report_env.py"
echo "[setup] done"
