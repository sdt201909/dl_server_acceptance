# dl_server_acceptance

`dl_server_acceptance` 是一个 Linux 深度学习训练服务器自动化验收工具。它面向 4 张 NVIDIA Pro/RTX PRO 6000 96GB GPU 的默认配置，同时通过 `acceptance.yaml` 支持其他 GPU 型号、数量、阈值、压测时长和测试路径。

工具默认安全：不会对裸块设备写入，不会修改 GPU power limit/clock，不会重启机器，不会 reset GPU，也不会自动安装系统包。

## 快速开始

```bash
cd dl_server_acceptance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
