#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./install_deps.sh --ubuntu
  ./install_deps.sh --rhel
  ./install_deps.sh --build-tools [--tools-dir ./tools]

This script installs system packages only when explicitly run by the user.
It never changes GPU power limits, clocks, ECC mode, or reboots the machine.
USAGE
}

TOOLS_DIR="./tools"
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ubuntu|--rhel|--build-tools)
      MODE="$1"
      shift
      ;;
    --tools-dir)
      TOOLS_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage
  exit 2
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

link_built_binary() {
  local search_dir="$1"
  local binary_name="$2"
  local link_name="$3"
  local found=""
  found="$(find "$search_dir" -type f -name "$binary_name" -perm -111 -print -quit)"
  if [[ -z "$found" ]]; then
    echo "ERROR: $binary_name build finished but executable was not found under $search_dir." >&2
    exit 1
  fi
  found="$(cd "$(dirname "$found")" && pwd -P)/$(basename "$found")"
  ln -sf "$found" "$TOOLS_DIR/$link_name"
  echo "Linked $TOOLS_DIR/$link_name -> $found"
}

install_ubuntu() {
  need_cmd sudo
  sudo apt-get update
  sudo apt-get install -y \
    python3 python3-pip python3-venv git build-essential cmake pkg-config \
    pciutils numactl stress-ng memtester fio smartmontools nvme-cli lm-sensors ipmitool \
    libboost-dev libboost-program-options-dev
}

install_rhel() {
  need_cmd sudo
  local installer="dnf"
  if ! command -v dnf >/dev/null 2>&1; then
    installer="yum"
  fi
  sudo "$installer" install -y \
    python3 python3-pip git gcc gcc-c++ make cmake pciutils numactl \
    stress-ng memtester fio smartmontools nvme-cli lm_sensors ipmitool \
    boost-devel boost-program-options
}

check_cuda_build_env() {
  need_cmd git
  need_cmd make
  need_cmd cmake
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
      export CUDA_HOME=/usr/local/cuda
    else
      echo "ERROR: CUDA_HOME is not set and /usr/local/cuda does not exist." >&2
      exit 1
    fi
  fi
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    exit 1
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
}

build_gpu_burn() {
  mkdir -p "$TOOLS_DIR"
  if [[ ! -d "$TOOLS_DIR/gpu-burn" ]]; then
    git clone https://github.com/wilicc/gpu-burn.git "$TOOLS_DIR/gpu-burn"
  fi
  make -C "$TOOLS_DIR/gpu-burn"
  ln -sf "$PWD/$TOOLS_DIR/gpu-burn/gpu_burn" "$TOOLS_DIR/gpu_burn"
}

build_nccl_tests() {
  mkdir -p "$TOOLS_DIR"
  if [[ ! -d "$TOOLS_DIR/nccl-tests" ]]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "$TOOLS_DIR/nccl-tests"
  fi
  make -C "$TOOLS_DIR/nccl-tests" MPI=0 CUDA_HOME="$CUDA_HOME"
  ln -sf "$PWD/$TOOLS_DIR/nccl-tests/build/all_reduce_perf" "$TOOLS_DIR/all_reduce_perf"
  ln -sf "$PWD/$TOOLS_DIR/nccl-tests/build/all_gather_perf" "$TOOLS_DIR/all_gather_perf"
  ln -sf "$PWD/$TOOLS_DIR/nccl-tests/build/reduce_scatter_perf" "$TOOLS_DIR/reduce_scatter_perf"
}

build_nvbandwidth() {
  mkdir -p "$TOOLS_DIR"
  local src="$TOOLS_DIR/nvbandwidth-src"
  if [[ ! -d "$src" ]]; then
    git clone https://github.com/NVIDIA/nvbandwidth.git "$src"
  fi
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$src/build" -j"$(nproc)"
  link_built_binary "$src/build" nvbandwidth nvbandwidth
}

build_cuda_memtest() {
  mkdir -p "$TOOLS_DIR"
  local src="$TOOLS_DIR/cuda_memtest-src"
  if [[ ! -d "$src" ]]; then
    git clone https://github.com/ComputationalRadiationPhysics/cuda_memtest.git "$src" || {
      echo "ERROR: cuda_memtest clone failed. Build it manually and put cuda_memtest in PATH/tools_dir." >&2
      exit 1
    }
  fi
  if [[ -f "$src/CMakeLists.txt" ]]; then
    local cmake_args=(-S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release)
    if [[ -n "${CUDA_MEMTEST_CUDA_ARCHITECTURES:-}" ]]; then
      cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_MEMTEST_CUDA_ARCHITECTURES}")
    fi
    cmake "${cmake_args[@]}"
    cmake --build "$src/build" -j"$(nproc)"
    link_built_binary "$src/build" cuda_memtest cuda_memtest
  elif [[ -f "$src/Makefile" ]]; then
    make -C "$src"
    link_built_binary "$src" cuda_memtest cuda_memtest
  else
    echo "ERROR: cuda_memtest build file not found; expected CMakeLists.txt or Makefile under $src." >&2
    exit 1
  fi
}

build_tools() {
  check_cuda_build_env
  mkdir -p "$TOOLS_DIR"
  echo "Building optional tools into: $TOOLS_DIR"
  build_gpu_burn
  build_nccl_tests
  build_nvbandwidth
  build_cuda_memtest
  echo "Done. Add this to PATH before running acceptance:"
  echo "  export PATH=\"$PWD/$TOOLS_DIR:\$PATH\""
}

case "$MODE" in
  --ubuntu) install_ubuntu ;;
  --rhel) install_rhel ;;
  --build-tools) build_tools ;;
esac
