# dl_server_acceptance

`dl_server_acceptance` 是一个 Linux 深度学习训练服务器自动化验收工具。它面向 4 张 NVIDIA Pro/RTX PRO 6000 96GB GPU 的默认配置，同时通过 `acceptance.yaml` 支持其他 GPU 型号、数量、阈值、压测时长和测试路径。

工具默认安全：不会对裸块设备写入，不会修改 GPU power limit/clock，不会重启机器，不会 reset GPU，也不会自动安装系统包。

## 快速开始

```bash
git clone https://github.com/sdt201909/dl_server_acceptance.git
cd dl_server_acceptance
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
cp acceptance.yaml.example acceptance.yaml

python acceptance.py preflight --config acceptance.yaml
python acceptance.py run --suite quick --config acceptance.yaml --output ./runs/quick_001
python acceptance.py run --suite standard --config acceptance.yaml --output ./runs/standard_001
```

只看实时监控：

```bash
python acceptance.py monitor --config acceptance.yaml --interval 5
```

从已有 run 目录重新生成报告：

```bash
python acceptance.py report --run-dir ./runs/standard_001
```

查看套件和 dry-run：

```bash
python acceptance.py list-suites
python acceptance.py run --suite standard --config acceptance.yaml --dry-run
```

## 正式验收推荐流程

下面流程面向新到货 4 卡 RTX PRO 6000 / Pro 6000 服务器，目标是做 6 小时左右长稳验收。

### 1. 更新代码

```bash
cd /root/dl_server_acceptance
git pull --ff-only
```

第一次使用：

```bash
git clone https://github.com/sdt201909/dl_server_acceptance.git
cd dl_server_acceptance
```

### 2. 安装基础依赖

Ubuntu：

```bash
sudo ./install_deps.sh --ubuntu
```

RHEL/Rocky/Alma：

```bash
sudo ./install_deps.sh --rhel
```

Python：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. 安装 CUDA 版 PyTorch

`torch_ddp` smoke test 需要 CUDA 版 PyTorch。不要简单运行 `pip install torch`，这可能装到 CPU 版或不匹配的 CUDA wheel。

CUDA 12.8 wheel 示例：

```bash
python3 -m pip install numpy
python3 -m pip install -r requirements-torch-cu128.txt
```

或：

```bash
./scripts/install_torch_cuda.sh cu128
```

确认：

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

说明：`nvidia-smi` 里的 `CUDA Version` 表示驱动最高支持的 CUDA runtime/API 版本；`nvcc --version` 表示已安装的 CUDA Toolkit 编译器版本。两者不完全相同是正常的。

### 4. 安装和验证 DCGM

正式验收建议安装 DCGM。`nvidia-smi` 正常不等于 `dcgmi` 已安装。

```bash
command -v dcgmi
dcgmi --version
dcgmi discovery -l
```

新驱动/CUDA 栈建议使用匹配的 DCGM 4 包。例如 `nvidia-smi` 显示 CUDA 13.x 时，优先查：

```bash
apt-cache search datacenter-gpu-manager-4
apt-cache policy datacenter-gpu-manager-4-cuda13
```

安装示例：

```bash
sudo apt-get update
sudo apt-get install -y --install-recommends datacenter-gpu-manager-4-cuda13
```

如果系统装着旧包，例如 `datacenter-gpu-manager 3.x`，可能会在 `dcgmi diag` 中出现 `Detected unsupported Cuda version`。这是 DCGM/CUDA/driver 软件栈不匹配，不是硬件失败证据。

DCGM r1 还可能因为 persistence mode 未开启而 fail：

```text
Persistence Mode: Persistence mode for GPU 0 is disabled
```

这也是运行配置问题。可开启：

```bash
sudo nvidia-smi -pm 1
dcgmi diag -r 1
```

开启 persistence mode 不会修改 power limit，不会 reset GPU，也不会重启机器。

### 5. 构建可选 GPU 测试工具

```bash
export CUDA_HOME=/usr/local/cuda
./install_deps.sh --build-tools --tools-dir ./tools
export PATH="$PWD/tools:$PATH"
```

常见构建问题：

- `nccl.h: No such file or directory`：缺 NCCL 开发包。Ubuntu 安装 `libnccl2 libnccl-dev`；RHEL 安装 `libnccl libnccl-devel`。自定义路径时设置 `NCCL_HOME` 或 `NCCL_INCLUDE/NCCL_LIB`。
- `Could NOT find Boost ... program_options`：`nvbandwidth` 缺 Boost。Ubuntu 安装 `libboost-dev libboost-program-options-dev`；RHEL 安装 `boost-devel boost-program-options`。
- `cuda_memtest Makefile not found`：当前上游是 CMake 工程；拉取最新脚本后会自动用 CMake 构建。
- 构建成功但找不到二进制：拉取最新脚本，已修复相对软链问题。

构建后检查：

```bash
for t in gpu_burn all_reduce_perf all_gather_perf reduce_scatter_perf nvbandwidth cuda_memtest; do
  command -v "$t" || command -v "./tools/$t" || true
done
```

### 6. 准备 fio 专用目录

不要用 `/`、`/home`、`/var`、`/tmp` 作为 fio 写入目录。建议使用专用挂载点：

```bash
sudo mkdir -p /mnt/acceptance/fio_acceptance
sudo chown -R "$USER:$USER" /mnt/acceptance/fio_acceptance
df -h /mnt/acceptance/fio_acceptance
```

### 7. 创建 6 小时配置

```bash
cp acceptance.yaml.example acceptance.full6h.yaml
```

重点修改：

```yaml
paths:
  fio_test_dir: "/mnt/acceptance/fio_acceptance"
  tools_dir: "./tools"

timeouts:
  gpu_burn_sec_standard: 600
  combined_sec_full: 21600

fio:
  size: "100G"
  runtime_sec: 300
  timeout_padding_sec: 900
  job_name: "fio_acceptance"
  filename_format: "fio_acceptance.$jobnum"

memtester:
  memory_fraction: 0.20
  passes: 1

stress_ng:
  cpu_timeout_sec: 600

torch_ddp:
  nproc_per_node: 4
```

说明：默认 `full` 是 `standard` 全部项目加 6 小时 combined load，总墙钟时间会超过 6 小时。上面的配置把前置项缩短，把核心 CPU/GPU/fio 联合满载保持 6 小时左右。

### 8. preflight

```bash
python acceptance.py preflight --config acceptance.full6h.yaml
```

确认：

- GPU 数量、型号、显存匹配采购配置。
- `nvidia-smi`、`dcgmi`、`fio`、`stress-ng`、`memtester`、`gpu_burn`、`cuda_memtest`、`nccl-tests`、`torch` 可用。
- `fio_test_dir` 可写且空间足够。
- dmesg 没有 `NVRM: Xid`、掉卡、AER uncorrected、真实 MCE hardware error。

注意：`MCE: In-kernel MCE decoding enabled.` 是正常初始化信息，不是硬件错误；最新规则不会把它判成风险。

### 9. dry-run

```bash
python acceptance.py run --suite full --config acceptance.full6h.yaml --dry-run
```

确认命令、路径、时长符合预期。尤其检查：

- `combined_load` 的 timeout 接近 `combined_sec_full + 300`。
- `fio` 目录是专用目录。
- `torch_ddp` 会用 `torchrun` 或 `python3 -m torch.distributed.run`。

### 10. 运行 6 小时长稳

建议放在 `tmux` 中：

```bash
tmux new -s pro6000_full6h
```

启动：

```bash
RUN=./runs/pro6000_full6h_$(date +%Y%m%d_%H%M%S)

python acceptance.py run \
  --suite full \
  --config acceptance.full6h.yaml \
  --output "$RUN" \
  --yes
```

如果 DCGM 仍有明确软件兼容问题、但你希望先收集非 DCGM 的长稳数据，可临时使用：

```bash
python acceptance.py run \
  --suite full \
  --config acceptance.full6h.yaml \
  --output "$RUN" \
  --yes \
  --continue-on-error
```

这种结果只能作为“非 DCGM 长稳参考”，正式签收仍建议修好 DCGM 后重跑。

### 11. 实时观察

```bash
tail -f "$RUN/metrics/risks.jsonl"
tail -f "$RUN/metrics/events.jsonl"
watch -n 5 nvidia-smi
```

如果出现 Xid、掉卡、ECC uncorrectable、温度 critical、gpu-burn/cuda_memtest/NCCL/torch DDP 错误，工具会记录 HIGH/CRITICAL 风险，默认会停止相关高负载测试。

### 12. 看报告和打包

```bash
python acceptance.py report --run-dir "$RUN"
less "$RUN/summary.md"
python3 -m json.tool "$RUN/summary.json" | less
```

打包给供应商：

```bash
tar czf pro6000_acceptance_$(date +%Y%m%d_%H%M%S).tar.gz "$RUN"
```

## 只补 GPU 6 小时测试

如果 CPU、内存、fio/SMART 已经完成，不想重复前面的压测，使用 GPU-only 补测配置：

```bash
cp acceptance.gpu6h.yaml.example acceptance.gpu6h.yaml
```

这个配置默认关闭：

- `stress-ng` CPU 压测
- `memtester` 内存压测
- `fio` 存储读写
- `SMART/NVMe` 健康检查
- `full` 里的 CPU/GPU/fio combined load
- DCGM r1/r2/r3 重复诊断

保留：

- inventory、`nvidia-smi -q`、拓扑
- dmesg、ECC/remapped rows 风险扫描
- 1 分钟短监控
- `gpu_burn` 6 小时
- `cuda_memtest`
- `nvbandwidth`
- NCCL all_reduce/all_gather/reduce_scatter
- PyTorch DDP smoke test

运行前先确认不会重复重项目：

```bash
python acceptance.py run --suite standard --config acceptance.gpu6h.yaml --dry-run
```

确认 dry-run 里 `cpu_stress`、`memtester`、`fio_*`、`smart_health`、`combined_load` 都是 disabled/skipped 后再启动：

```bash
RUN=./runs/pro6000_gpu6h_$(date +%Y%m%d_%H%M%S)

python acceptance.py run \
  --suite standard \
  --config acceptance.gpu6h.yaml \
  --output "$RUN"
```

实时观察：

```bash
tail -f "$RUN/metrics/risks.jsonl"
tail -f "$RUN/metrics/events.jsonl"
watch -n 5 nvidia-smi
```

说明：这里故意不用 `full`。`full` 的额外价值是 CPU/GPU/fio 联合满载，如果前面 CPU、内存、存储已经验收完成，继续跑 `full` 就会重复这些项目。GPU-only 补测应该跑 `standard`，再通过配置关闭非 GPU 项。

如果 DCGM r2/r3 还没有做过，且你希望把它们也纳入这次 GPU 侧验收，把配置里的 `tests.enable_dcgm` 改为 `true`；总耗时会比 6 小时更长，通常会额外增加 1 小时以上。

## 安装系统依赖

Ubuntu：

```bash
./install_deps.sh --ubuntu
```

RHEL/Rocky/Alma：

```bash
./install_deps.sh --rhel
```

Python 依赖：

```bash
pip install -r requirements.txt
```

PyTorch DDP smoke test 需要 CUDA 版 PyTorch。不要在裸机上直接 `pip install torch` 了事，容易装到不匹配的 wheel。按 PyTorch 官方安装页选择 Linux / Pip / Python / CUDA 平台；CUDA 12.8 环境可用：

```bash
python3 -m pip install numpy
python3 -m pip install -r requirements-torch-cu128.txt

# 或显式使用脚本
./scripts/install_torch_cuda.sh cu128
```

本工具只需要 `torch` 本体，不需要 `torchvision` 或 `torchaudio`。

第三方 Python 库缺失时，核心逻辑会尽量降级：没有 `rich` 就使用普通文本状态输出；没有 `psutil` 就读取 `/proc`；没有 `nvidia-ml-py` 就回退到 `nvidia-smi --query-gpu`。

## 构建可选 GPU 测试工具

`gpu-burn`、`nccl-tests`、`nvbandwidth`、`cuda_memtest` 通常需要 CUDA 编译环境：

```bash
export CUDA_HOME=/usr/local/cuda
./install_deps.sh --build-tools --tools-dir ./tools
export PATH="$PWD/tools:$PATH"
```

构建脚本会检查 `CUDA_HOME`、`nvcc`、`git`、`cmake`、`make`。失败时会直接报错，不会吞掉日志。

## 套件说明

- `quick`：到货快速检查，包括 inventory、`nvidia-smi -q`、拓扑、dmesg 风险扫描、DCGM r1、短监控、可选 nvbandwidth/NCCL/PyTorch DDP smoke。
- `standard`：正式验收，包括 quick 全部项目，以及 CPU、内存、fio、SMART/NVMe、DCGM r1/r2/r3、gpu-burn、cuda_memtest、NCCL all_reduce/all_gather/reduce_scatter、torchrun DDP。
- `full`：长稳验收，包括 standard 和 CPU/GPU/fio 联合满载，默认 6 小时。
- `burnin`：24 小时 burn-in，包括 full 并把联合负载延长到 24 小时。

`full` 和 `burnin` 默认要求交互确认；自动化环境可加 `--yes`。

## 风险等级

- `INFO`：记录性信息，不影响验收。
- `WARN`：需要关注或人工核对，但不必然代表硬件失败。
- `HIGH`：高风险，通常不建议直接签收，需要供应商解释或复测。
- `CRITICAL`：严重风险，如掉卡、Xid、ECC uncorrectable、压测错误、DCGM fail，通常建议停止压测并联系供应商/RMA。

退出码：

- `0`：PASS，无 HIGH/CRITICAL。
- `1`：PASS_WITH_WARNINGS，只有 WARN/INFO。
- `2`：FAIL，存在 HIGH 或 CRITICAL。
- `3`：INCOMPLETE，有必要测试未执行或中途失败但无法判断硬件失败。
- `130`：用户中断，仍会生成 partial report。

## 输出目录

每次运行会生成：

- `summary.md`：中文验收报告。
- `summary.json`：机器可读报告。
- `inventory.json`：硬件和工具盘点。
- `environment.txt`：环境命令原始输出。
- `commands.jsonl`：每条命令的 cmd/start/end/returncode/timeout。
- `risks.jsonl`：风险事件。
- `raw_logs/`：每个 stage 的 stdout/stderr。
- `metrics/`：`gpu_metrics.csv`、`system_metrics.csv`、`events.jsonl`、`risks.jsonl`。

把 `summary.md`、`summary.json`、`risks.jsonl`、`commands.jsonl`、`environment.txt` 和整个 `raw_logs/` 打包给供应商，通常足够定位验收失败原因。

## fio 测试目录

`fio` 只会写入 `paths.fio_test_dir`，不会写裸盘。默认拒绝 `/`、`/home`、`/var`、`/tmp` 等危险路径。建议准备专用挂载点，例如：

```yaml
paths:
  fio_test_dir: "/mnt/test/fio_acceptance"
```

如果确实已经人工确认目录安全，可使用 `--force`，但报告中仍会记录强风险提示。

## 常见问题

### 没有 sudo 怎么办

大多数 GPU/NCCL/PyTorch 测试不需要 sudo。`smartctl`、`nvme`、`ipmitool`、`dmesg` 在某些系统上可能需要权限；没有权限时工具会记录 WARN 或 SKIPPED，不会直接崩溃。

### dcgmi 不存在怎么办

`preflight` 会显示 `dcgmi` missing。quick/standard 中对应 DCGM stage 会跳过并记录风险。正式验收建议安装 NVIDIA DCGM 后复测。

### DCGM 报 Detected unsupported Cuda version 怎么办

这是 DCGM 与当前 driver/CUDA 软件栈不兼容，属于软件配置问题，不是硬件失败证据。典型场景是新驱动和 Blackwell GPU 搭配了旧 `datacenter-gpu-manager 3.x`。

处理：

```bash
dcgmi --version
nvidia-smi
nvcc --version
dpkg -l | grep -i datacenter-gpu-manager
apt-cache search datacenter-gpu-manager-4
```

如果 `nvidia-smi` 显示 CUDA 13.x，优先安装匹配的 DCGM 4 CUDA 13 包，例如：

```bash
sudo apt-get install -y --install-recommends datacenter-gpu-manager-4-cuda13
```

然后：

```bash
dcgmi discovery -l
dcgmi diag -r 1
```

### DCGM r1 因 Persistence Mode disabled 失败怎么办

这是运行配置问题，不是硬件故障。开启：

```bash
sudo nvidia-smi -pm 1
dcgmi diag -r 1
```

这不会修改 power limit，不会 reset GPU，也不会重启机器。

### 测试很快就结束了怎么办

先看 `summary.md` 的“测试套件与参数”表，找最后一个执行的 stage。

- 停在 `dcgm_r1` 且 evidence 是 `Detected unsupported Cuda version`：DCGM 软件栈不兼容。
- 停在 `dcgm_r1` 且 evidence 是 persistence mode：开启 persistence mode。
- 停在 `dmesg_scan` 且有 `NVRM: Xid`、`fallen off the bus`、AER uncorrected、真实 MCE hardware error：硬件/驱动风险，需要暂停并收集日志。
- 停在 `fio_*`：检查 fio 测试目录、空间、权限和磁盘健康。
- 停在 `gpu_burn`、`cuda_memtest`、`nccl_*`、`torch_ddp`：优先查看对应 `raw_logs/*.stdout.log` 和 `*.stderr.log`。

如果只是 DCGM 软件兼容问题，修好 DCGM 后重跑。若临时想收集非 DCGM 长稳数据，可以用 `--continue-on-error`，但这种结果不建议直接作为正式签收结论。

### gpu-burn 立刻出现 DIED / read[N] error 怎么办

如果 `gpu_burn` 在 0.x% 就退出，温度还很低，并出现 `DIED!`、`read[0] error`、`read[1] error`，先按软件/工具调用问题排查，不要直接判硬件坏。

常见原因：

- `gpu_burn` 没有从它自己的构建目录启动，导致同目录的 `compare.fatbin` 没有被正确加载。
- `gpu_burn` 是用不匹配的 CUDA Toolkit 或 compute capability 编译的。
- `-tc` Tensor Core 路径和当前 CUDA/driver/gpu-burn 组合不兼容。

先更新工具仓库，最新版本会自动从 `gpu_burn` 真实二进制所在目录启动：

```bash
git pull --ff-only
export PATH="$PWD/tools:$PATH"
```

然后做 30 秒短测：

```bash
cd /root/dl_server_acceptance
python acceptance.py run --suite standard --config acceptance.gpu6h.yaml --dry-run

realpath "$(command -v gpu_burn)"
cd "$(dirname "$(realpath "$(command -v gpu_burn)")")"
./gpu_burn -m 50% 30
./gpu_burn -m 50% -tc 30
```

RTX PRO 6000 Blackwell 是 compute capability 12.0。如果短测仍失败，重新构建：

```bash
cd /root/dl_server_acceptance
CUDA_HOME=/usr/local/cuda-12.8 GPU_BURN_COMPUTE=120 ./install_deps.sh --build-tools --tools-dir ./tools
export PATH="$PWD/tools:$PATH"
```

如果无 `-tc` 能过、加 `-tc` 失败，先把 `acceptance.gpu6h.yaml` 中的 `gpu_burn.use_tensor_cores` 改成 `false` 跑完 GPU burn，再用 PyTorch DDP/NCCL 单独验证 Tensor Core/NCCL 路径。

如果每张卡单独跑都会同样 0.x% 失败，通常更像工具/软件栈问题。如果只有某一张卡失败，或者同时出现 `NVRM: Xid`、掉卡、ECC uncorrectable、AER uncorrected，就按硬件/链路风险处理并联系供应商。

### 旧报告里的误报为什么还在

报告是运行时写入的静态文件。更新代码后，旧 run 目录里的 `summary.md`、`summary.json`、`risks.jsonl` 不会自动重算。需要重新跑验收，或至少重新跑相关 suite，才能得到新规则下的报告。

### MCE: In-kernel MCE decoding enabled 是硬件错误吗

不是。这是内核启用 MCE 解码能力的正常初始化日志。真正危险的是 `mce: [Hardware Error]`、`Machine Check Exception`、`Machine check events logged` 等实际错误事件。

### 构建 nccl-tests 报 nccl.h 找不到怎么办

缺 NCCL 开发包。Ubuntu：

```bash
sudo apt-get install -y libnccl2 libnccl-dev
```

RHEL/Rocky/Alma：

```bash
sudo dnf install -y libnccl libnccl-devel
```

如果 NCCL 在自定义目录，设置 `NCCL_HOME` 或 `NCCL_INCLUDE/NCCL_LIB` 后重新构建。

### 构建 nvbandwidth 报 Boost program_options 缺失怎么办

Ubuntu：

```bash
sudo apt-get install -y libboost-dev libboost-program-options-dev
```

RHEL/Rocky/Alma：

```bash
sudo dnf install -y boost-devel boost-program-options
```

### 构建 cuda_memtest 报 Makefile not found 怎么办

当前上游 `cuda_memtest` 是 CMake 工程。请先拉取最新脚本：

```bash
git pull --ff-only
./install_deps.sh --build-tools --tools-dir ./tools
```

如果需要指定 CUDA 架构：

```bash
CUDA_MEMTEST_CUDA_ARCHITECTURES=90 ./install_deps.sh --build-tools --tools-dir ./tools
```

### nvidia-smi 查询字段 Not Supported 怎么办

解析器会把 `N/A`、`Not Supported`、`[Not Supported]`、`unknown` 当成空值处理。单个字段不支持不会直接失败，但 GPU 数量、型号、显存、温度、掉卡、Xid/ECC 等关键风险仍会判定。

### full/burnin 为什么需要确认

这两个套件会长时间满载 CPU/GPU/存储。默认需要手动输入确认，防止误触发长时间高功耗压测。自动化环境可使用 `--yes`。

### 如何复测单项

先用 dry-run 找到 stage 名和命令：

```bash
python acceptance.py run --suite standard --dry-run
```

然后可以直接执行 raw command，或在 `acceptance.yaml` 的 `commands:` 中覆盖对应 stage 的命令。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
