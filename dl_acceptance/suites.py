from __future__ import annotations

import os
import re
import shlex
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .command import CommandResult, CommandRunner
from .config import AcceptanceConfig
from .risks import RiskEngine
from .utils import (
    JsonlWriter,
    available_memory_mb,
    command_path,
    command_version,
    cpu_model,
    disk_free_gb,
    ensure_dir,
    hostname,
    now_iso,
    read_text_limited,
    run_capture,
    shlex_join,
    strip_ansi,
    total_memory_gb,
    write_json,
)
from .parsers import parse_nvidia_smi_csv


SUITES = {
    "quick": "快速到货检查：inventory、NVIDIA 基础信息、内核风险、DCGM r1、短监控、可选带宽/NCCL/DDP smoke。",
    "standard": "正式验收：quick + CPU/内存/存储/DCGM r1-r3/gpu-burn/cuda_memtest/NCCL/DDP。",
    "full": "长时间稳定性验收：standard + CPU/GPU/fio 联合满载。",
    "burnin": "24 小时 burn-in：full + 24h 联合负载。",
}


EXTERNAL_TOOLS = [
    ("bash", False, False),
    ("dmesg", True, True),
    ("journalctl", True, True),
    ("nvidia-smi", False, False),
    ("dcgmi", True, False),
    ("stress-ng", False, False),
    ("memtester", False, False),
    ("fio", False, False),
    ("smartctl", False, True),
    ("nvme", True, True),
    ("ipmitool", True, True),
    ("gpu_burn", False, False),
    ("cuda_memtest", True, False),
    ("nvbandwidth", True, False),
    ("all_reduce_perf", True, False),
    ("all_gather_perf", True, False),
    ("reduce_scatter_perf", True, False),
    ("torchrun", True, False),
    ("python3", False, False),
    ("numactl", True, False),
    ("lspci", True, False),
    ("lsblk", True, False),
]


@dataclass
class StageSpec:
    name: str
    title: str
    command_type: str = "generic"
    cmd: list[str] | str | None = None
    timeout: int | None = None
    enabled: bool = True
    required: bool = True
    tool: str | None = None
    internal: str | None = None
    shell: bool = False


class MultiJsonlWriter:
    def __init__(self, *paths: Path):
        self.writers = [JsonlWriter(path) for path in paths]

    def write(self, obj: dict[str, Any]) -> None:
        for writer in self.writers:
            writer.write(obj)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()


def parse_size_gb(value: str) -> float:
    text = str(value).strip().lower()
    match = re.match(r"([0-9.]+)\s*([kmgtp]?i?b?)?", text)
    if not match:
        return 0.0
    num = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "": 1 / (1024**3),
        "k": 1 / (1024**2),
        "kb": 1 / (1024**2),
        "m": 1 / 1024,
        "mb": 1 / 1024,
        "g": 1,
        "gb": 1,
        "t": 1024,
        "tb": 1024,
    }
    return num * multipliers.get(unit, 1)


def collect_inventory(config: AcceptanceConfig, run_dir: Path | None = None) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "timestamp": now_iso(),
        "hostname": hostname(),
        "kernel": os.uname().release if hasattr(os, "uname") else "unknown",
        "platform": " ".join(os.uname()) if hasattr(os, "uname") else "unknown",
        "cpu": {"model": cpu_model(), "logical_cores": os.cpu_count()},
        "memory": {"total_gb": total_memory_gb()},
        "paths": {
            "work_dir": str(config.work_dir),
            "fio_test_dir": str(config.fio_test_dir),
            "tools_dir": str(config.tools_dir),
        },
        "fio": {
            "test_dir": str(config.fio_test_dir),
            "dangerous_path": config.is_dangerous_fio_dir(),
            "free_gb": disk_free_gb(config.fio_test_dir),
            "required_free_gb": config.get("thresholds.disk_free_min_gb", 200),
            "writable": _path_writable(config.fio_test_dir),
        },
        "tools": {},
        "commands": {},
        "gpus": [],
        "driver": {},
        "topology": {},
        "sudo": {},
    }

    if Path("/etc/os-release").exists():
        inventory["os_release"] = Path("/etc/os-release").read_text(errors="replace")
    else:
        inventory["os_release"] = ""

    inv_cmds = {
        "hostnamectl": ["hostnamectl"],
        "uname": ["uname", "-a"],
        "lscpu": ["lscpu"],
        "free": ["free", "-h"],
        "numactl": ["numactl", "--hardware"],
        "lspci_nn": ["lspci", "-nn"],
        "lspci_tv": ["lspci", "-tv"],
        "lsblk": ["lsblk", "-o", "NAME,MODEL,SERIAL,SIZE,TYPE,ROTA,MOUNTPOINT,FSTYPE"],
        "nvidia_smi_L": ["nvidia-smi", "-L"],
        "nvidia_smi_q": ["nvidia-smi", "-q"],
        "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
        "nvme_list": ["nvme", "list"],
    }
    for name, cmd in inv_cmds.items():
        code, out, err = run_capture(cmd, timeout=30)
        inventory["commands"][name] = {"cmd": shlex_join(cmd), "returncode": code, "stdout": out, "stderr": err}

    query_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,uuid,serial,pci.bus_id,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,utilization.gpu,utilization.memory,memory.total,memory.used,ecc.mode.current",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_capture(query_cmd, timeout=20)
    if code == 0:
        inventory["gpus"] = parse_nvidia_smi_csv(out)
    else:
        inventory["driver"]["query_failed"] = True
        inventory["driver"]["query_error"] = err

    driver_cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    code, out, err = run_capture(driver_cmd, timeout=15)
    if code == 0 and out.strip():
        inventory["driver"]["nvidia_driver"] = out.strip().splitlines()[0]
    code, out, err = run_capture(["nvcc", "--version"], timeout=15)
    inventory["driver"]["cuda_runtime_nvcc"] = out.strip() if code == 0 else None
    topology = inventory["commands"].get("nvidia_smi_topo", {})
    inventory["topology"]["nvidia_smi_topo_m"] = strip_ansi(topology.get("stdout", ""))

    sudo_path = shutil.which("sudo")
    if sudo_path:
        code, out, err = run_capture(["sudo", "-n", "true"], timeout=5)
        inventory["sudo"] = {"found": True, "non_interactive": code == 0, "path": sudo_path}
    else:
        inventory["sudo"] = {"found": False, "non_interactive": False, "path": None}

    for tool, optional, needs_sudo in EXTERNAL_TOOLS:
        path = command_path(tool)
        inventory["tools"][tool] = {
            "found": bool(path),
            "missing_kind": "optional missing" if optional and not path else ("missing" if not path else "found"),
            "optional": optional,
            "version": command_version(tool) if path else None,
            "path": path,
            "needs_sudo": needs_sudo,
        }

    if run_dir:
        ensure_dir(run_dir)
        write_json(run_dir / "inventory.json", inventory)
        env_lines = [
            f"timestamp: {inventory['timestamp']}",
            f"hostname: {inventory['hostname']}",
            f"kernel: {inventory['kernel']}",
            "",
            "## /etc/os-release",
            inventory.get("os_release", ""),
        ]
        for name, item in inventory["commands"].items():
            env_lines.extend(
                [
                    "",
                    f"## {name}: {item['cmd']} (rc={item['returncode']})",
                    item.get("stdout", ""),
                    item.get("stderr", ""),
                ]
            )
        (run_dir / "environment.txt").write_text("\n".join(env_lines), encoding="utf-8")
    return inventory


def _path_writable(path: Path) -> bool:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return os.access(probe, os.W_OK)


def sudo_readonly_cmd(command: str) -> list[str]:
    quoted = shlex.quote(command)
    return [
        "bash",
        "-lc",
        f"if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then sudo -n bash -lc {quoted}; else bash -lc {quoted}; fi",
    ]


def print_preflight(inventory: dict[str, Any]) -> None:
    print(f"Hostname: {inventory.get('hostname')}")
    print(f"OS/kernel: {inventory.get('kernel')}")
    print(f"CPU: {inventory.get('cpu', {}).get('model')} ({inventory.get('cpu', {}).get('logical_cores')} logical cores)")
    mem = inventory.get("memory", {}).get("total_gb")
    print(f"Memory: {mem:.1f} GB" if isinstance(mem, (int, float)) else "Memory: unknown")
    fio = inventory.get("fio", {})
    print(
        f"FIO dir: {fio.get('test_dir')} free={fio.get('free_gb'):.1f}GB "
        f"writable={fio.get('writable')} dangerous={fio.get('dangerous_path')}"
    )
    print(f"sudo: found={inventory.get('sudo', {}).get('found')} non_interactive={inventory.get('sudo', {}).get('non_interactive')}")
    numa = inventory.get("commands", {}).get("numactl", {}).get("stdout", "").strip()
    if numa:
        print("\nNUMA:")
        print("\n".join(numa.splitlines()[:12]))
    print("\nGPU inventory:")
    for gpu in inventory.get("gpus", []):
        mem_gb = (gpu.get("memory.total") or 0) / 1024 if gpu.get("memory.total") else None
        print(
            f"  GPU{gpu.get('index')}: {gpu.get('name')} mem={mem_gb:.1f}GB uuid={gpu.get('uuid')} bus={gpu.get('pci.bus_id')}"
            if mem_gb
            else f"  GPU{gpu.get('index')}: {gpu.get('name')} uuid={gpu.get('uuid')} bus={gpu.get('pci.bus_id')}"
        )
    if not inventory.get("gpus"):
        print("  No GPU detected by nvidia-smi query.")
    print(f"\nDriver/CUDA: {inventory.get('driver')}")
    topo = inventory.get("topology", {}).get("nvidia_smi_topo_m", "")
    if topo:
        print("\nnvidia-smi topo -m:")
        print(topo.rstrip())
    print("\nExternal tools:")
    for name, info in sorted(inventory.get("tools", {}).items()):
        print(
            f"  {name:22} {info.get('missing_kind', 'found'):16} path={info.get('path')} "
            f"sudo={info.get('needs_sudo')} version={info.get('version')}"
        )


class SuiteRunner:
    def __init__(
        self,
        config: AcceptanceConfig,
        suite: str,
        run_dir: Path,
        risk_engine: RiskEngine,
        dry_run: bool = False,
        continue_on_error: bool = False,
        stop_on_critical_risk: bool | None = None,
        force: bool = False,
    ):
        self.config = config
        self.suite = suite
        self.run_dir = ensure_dir(run_dir)
        ensure_dir(self.run_dir / "raw_logs")
        ensure_dir(self.run_dir / "metrics")
        self.risk_engine = risk_engine
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.stop_on_critical_risk = (
            bool(config.get("safety.stop_on_critical_risk", True)) if stop_on_critical_risk is None else stop_on_critical_risk
        )
        self.force = force
        self.current_stage = ""
        self.active_command: str | None = None
        self.events = MultiJsonlWriter(self.run_dir / "events.jsonl", self.run_dir / "metrics" / "events.jsonl")
        self.stage_writer = JsonlWriter(self.run_dir / "stages.jsonl")
        self.command_runner = CommandRunner(config, self.run_dir, self.events, self._set_active_command)
        self.interrupted = False

    def close(self) -> None:
        self.command_runner.close()
        self.stage_writer.close()
        self.events.close()

    def _set_active_command(self, cmd: str | None) -> None:
        self.active_command = cmd

    def stages(self) -> list[StageSpec]:
        if self.suite not in SUITES:
            raise ValueError(f"unknown suite: {self.suite}")
        stages: list[StageSpec] = [
            StageSpec("inventory", "Collect inventory and baseline risk checks", internal="inventory", command_type="inventory"),
            StageSpec("nvidia_smi_q", "nvidia-smi -q", cmd=["nvidia-smi", "-q"], command_type="generic", tool="nvidia-smi"),
            StageSpec("nvidia_topo", "nvidia-smi topo -m", cmd=["nvidia-smi", "topo", "-m"], command_type="generic", tool="nvidia-smi"),
            StageSpec("dmesg_scan", "Kernel log risk scan", cmd=sudo_readonly_cmd("dmesg --ctime --kernel --color=never"), command_type="dmesg", tool="dmesg", required=False),
            StageSpec(
                "journalctl_kernel",
                "journalctl kernel risk scan",
                cmd=sudo_readonly_cmd("journalctl -k -n 5000 --no-pager"),
                command_type="dmesg",
                tool="journalctl",
                enabled=bool(self.config.get("monitor.capture_journalctl", False)),
                required=False,
            ),
            StageSpec(
                "ecc_retired_pages",
                "NVIDIA retired pages query",
                cmd=["nvidia-smi", "--query-retired-pages=timestamp,gpu_uuid,retired_pages.address,retired_pages.cause", "--format=csv"],
                command_type="ecc",
                tool="nvidia-smi",
                required=False,
            ),
            StageSpec(
                "ecc_remapped_rows",
                "NVIDIA remapped rows query",
                cmd=[
                    "nvidia-smi",
                    "--query-remapped-rows=gpu_uuid,remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure",
                    "--format=csv",
                ],
                command_type="ecc",
                tool="nvidia-smi",
                required=False,
            ),
            StageSpec(
                "ipmi_health",
                "BMC/IPMI sensors and SEL",
                cmd=sudo_readonly_cmd("ipmitool sensor && ipmitool sel list"),
                command_type="ipmi",
                tool="ipmitool",
                required=False,
                enabled=self.config.test_enabled("ipmi"),
            ),
            self._dcgm_stage("dcgm_r1", 1, self.config.get("timeouts.dcgm_r1_sec", 300), required=False),
            StageSpec("monitor_short", "Short passive monitoring window", internal="monitor_short", command_type="monitor", required=False),
            StageSpec(
                "nvbandwidth",
                "nvbandwidth GPU bandwidth test",
                cmd=["nvbandwidth"],
                command_type="nvbandwidth",
                tool="nvbandwidth",
                required=False,
                enabled=self.config.test_enabled("nvbandwidth"),
            ),
            self._nccl_stage("nccl_all_reduce", "all_reduce_perf", required=False),
            self._torch_stage(required=self.suite in {"standard", "full", "burnin"}),
        ]
        if self.suite in {"standard", "full", "burnin"}:
            stages.extend(
                [
                    self._stress_stage(),
                    self._memtester_stage(),
                    self._fio_stage("fio_seqwrite", "write"),
                    self._fio_stage("fio_seqread", "read"),
                    self._fio_stage("fio_randrw", "randrw"),
                    StageSpec(
                        "smart_health",
                        "SMART/NVMe health checks",
                        cmd=sudo_readonly_cmd("for d in /dev/nvme*n1 /dev/sd?; do [ -e \"$d\" ] && smartctl -x \"$d\"; done; nvme list || true"),
                        command_type="smart",
                        tool="smartctl",
                        required=False,
                        enabled=self.config.test_enabled("smart"),
                    ),
                    self._dcgm_stage("dcgm_r2", 2, self.config.get("timeouts.dcgm_r2_sec", 900), required=True),
                    self._dcgm_stage("dcgm_r3", 3, self.config.get("timeouts.dcgm_r3_sec", 3600), required=True),
                    self._dcgm_stage("dcgm_r4", 4, self.config.get("timeouts.dcgm_r4_sec", 7200), required=False, enabled=self.config.test_enabled("dcgm_r4")),
                    self._gpu_burn_stage("gpu_burn", int(self.config.get("timeouts.gpu_burn_sec_standard", 7200)), required=True),
                    StageSpec(
                        "cuda_memtest",
                        "cuda_memtest stress",
                        cmd=["cuda_memtest", "--stress", "--num_iterations", "100", "--num_passes", "1"],
                        command_type="cuda_memtest",
                        tool="cuda_memtest",
                        timeout=None,
                        required=False,
                        enabled=self.config.test_enabled("cuda_memtest"),
                    ),
                    self._nccl_stage("nccl_all_gather", "all_gather_perf", required=True),
                    self._nccl_stage("nccl_reduce_scatter", "reduce_scatter_perf", required=True),
                ]
            )
        if self.suite in {"full", "burnin"}:
            duration = int(self.config.get("timeouts.combined_sec_burnin" if self.suite == "burnin" else "timeouts.combined_sec_full", 21600))
            stages.append(self._combined_stage(duration))
        return [self._apply_override(stage) for stage in stages]

    def _apply_override(self, stage: StageSpec) -> StageSpec:
        override = self.config.command_override(stage.name)
        if override:
            stage.cmd = override
            stage.internal = None
            stage.shell = isinstance(override, str)
        return stage

    def _dcgm_stage(self, name: str, level: int, timeout: int, required: bool = True, enabled: bool | None = None) -> StageSpec:
        return StageSpec(
            name,
            f"DCGM diagnostic r{level}",
            cmd=["dcgmi", "diag", "-r", str(level)],
            command_type="dcgm",
            tool="dcgmi",
            timeout=int(timeout) + 120,
            required=required,
            enabled=self.config.test_enabled("dcgm") if enabled is None else enabled,
        )

    def _stress_stage(self) -> StageSpec:
        sec = int(self.config.get("stress_ng.cpu_timeout_sec", 7200))
        method = str(self.config.get("stress_ng.cpu_method", "matrixprod"))
        return StageSpec(
            "cpu_stress",
            "stress-ng CPU stress",
            cmd=["stress-ng", "--cpu", str(os.cpu_count() or 1), "--cpu-method", method, "--verify", "--metrics-brief", "--timeout", f"{sec}s"],
            command_type="stress_ng",
            tool="stress-ng",
            timeout=sec + 120,
            enabled=self.config.test_enabled("cpu_stress"),
            required=True,
        )

    def _memtester_stage(self) -> StageSpec:
        fraction = float(self.config.get("memtester.memory_fraction", 0.80))
        mb = available_memory_mb(fraction)
        passes = int(self.config.get("memtester.passes", 2))
        return StageSpec(
            "memtester",
            "memtester available memory",
            cmd=["memtester", f"{mb}M", str(passes)],
            command_type="memtester",
            tool="memtester",
            timeout=None,
            enabled=self.config.test_enabled("memtester"),
            required=True,
        )

    def _fio_stage(self, name: str, rw: str) -> StageSpec:
        fio = self.config.get("fio", {})
        runtime = int(fio.get("runtime_sec", 1800))
        timeout_padding = int(fio.get("timeout_padding_sec", 900))
        cmd = [
            "fio",
            f"--name={fio.get('job_name', 'fio_acceptance')}",
            f"--directory={self.config.fio_test_dir}",
            f"--filename_format={fio.get('filename_format', 'fio_acceptance.$jobnum')}",
            f"--size={fio.get('size', '100G')}",
            f"--rw={rw}",
            f"--bs={fio.get('bs', '1M')}",
            f"--direct={1 if fio.get('direct', True) else 0}",
            f"--numjobs={fio.get('numjobs', 4)}",
            f"--iodepth={fio.get('iodepth', 32)}",
            "--time_based",
            f"--runtime={runtime}",
            "--group_reporting",
            "--output-format=json",
        ]
        return StageSpec(name, f"fio {rw}", cmd=cmd, command_type="fio", tool="fio", timeout=runtime + timeout_padding, enabled=self.config.test_enabled("fio"), required=True)

    def _gpu_burn_stage(self, name: str, duration: int, required: bool) -> StageSpec:
        cmd = ["bash", "-lc", self._gpu_burn_shell(duration)]
        return StageSpec(name, f"gpu-burn {duration}s", cmd=cmd, command_type="gpu_burn", tool="gpu_burn", timeout=duration + 180, enabled=self.config.test_enabled("gpu_burn"), required=required)

    def _gpu_burn_shell(self, duration: int) -> str:
        args = ["-m", str(self.config.get("gpu_burn.memory", "90%"))]
        if self.config.get("gpu_burn.use_tensor_cores", True):
            args.append("-tc")
        args.append(str(duration))
        args_text = shlex.join(args)
        return (
            "bin=$(command -v gpu_burn) || exit 127; "
            'real=$(readlink -f "$bin" 2>/dev/null || echo "$bin"); '
            'dir=$(dirname "$real"); '
            'cd "$dir" && exec "$real" '
            f"{args_text}"
        )

    def _nccl_stage(self, name: str, binary: str, required: bool) -> StageSpec:
        nccl = self.config.get("nccl", {})
        gpus = int(nccl.get("gpus", self.config.get("expected.gpu_count", 4)))
        cmd = [binary, "-b", str(nccl.get("min_bytes", "8")), "-e", str(nccl.get("max_bytes", "8G")), "-f", str(nccl.get("factor", 2)), "-g", str(gpus)]
        return StageSpec(name, binary, cmd=cmd, command_type="nccl", tool=binary, timeout=None, enabled=self.config.test_enabled("nccl_tests"), required=required)

    def _torch_stage(self, required: bool) -> StageSpec:
        torch_cfg = self.config.get("torch_ddp", {})
        nproc = int(torch_cfg.get("nproc_per_node", self.config.get("expected.gpu_count", 4)))
        script = Path(__file__).resolve().parents[1] / "scripts" / "torch_ddp_smoke.py"
        args = [
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(script),
            "--matrix-size",
            str(torch_cfg.get("matrix_size", 2048)),
            "--iterations",
            str(torch_cfg.get("iterations", 20)),
            "--dtype",
            str(torch_cfg.get("dtype", "fp16")),
        ]
        torchrun_cmd = shlex.join(["torchrun", *args])
        python_module_cmd = shlex.join(["python3", "-m", "torch.distributed.run", *args])
        cmd = ["bash", "-lc", f"if command -v torchrun >/dev/null 2>&1; then {torchrun_cmd}; else {python_module_cmd}; fi"]
        return StageSpec("torch_ddp", "PyTorch DDP smoke test", cmd=cmd, command_type="torch_ddp", tool=None, timeout=None, enabled=self.config.test_enabled("torch_ddp"), required=required)

    def _combined_stage(self, duration: int) -> StageSpec:
        fio = self.config.get("fio", {})
        stress = f"stress-ng --cpu {os.cpu_count() or 1} --cpu-method {self.config.get('stress_ng.cpu_method', 'matrixprod')} --verify --metrics-brief --timeout {duration}s"
        gpu = self._gpu_burn_shell(duration)
        fio_cmd = (
            f"fio --name=combined_randrw --directory={self.config.fio_test_dir} --size={fio.get('size', '100G')} --rw=randrw "
            f"--bs={fio.get('bs', '1M')} --direct={1 if fio.get('direct', True) else 0} --numjobs={fio.get('numjobs', 4)} "
            f"--iodepth={fio.get('iodepth', 32)} --time_based --runtime={duration} --group_reporting --output-format=json"
        )
        cmd = f"set -o pipefail; ({stress}) & ({gpu}) & ({fio_cmd}) & wait"
        return StageSpec("combined_load", f"combined CPU/GPU/fio load {duration}s", cmd=["bash", "-lc", cmd], command_type="combined", tool="bash", timeout=duration + 300, enabled=self.config.test_enabled("combined"), required=True)

    def plan_rows(self) -> list[dict[str, Any]]:
        rows = []
        for stage in self.stages():
            rows.append(
                {
                    "stage": stage.name,
                    "enabled": stage.enabled,
                    "required": stage.required,
                    "tool": stage.tool,
                    "timeout": stage.timeout,
                    "cmd": shlex_join(stage.cmd) if stage.cmd else f"<internal:{stage.internal}>",
                }
            )
        return rows

    def run(self) -> dict[str, Any]:
        self.events.write({"timestamp": now_iso(), "event": "suite_start", "suite": self.suite, "run_dir": str(self.run_dir)})
        completed = []
        try:
            for stage in self.stages():
                result = self._run_stage(stage)
                completed.append(result)
                if result["status"] == "FAIL" and result.get("required") and not self.continue_on_error:
                    self.events.write({"timestamp": now_iso(), "event": "stop_on_required_stage_failure", "stage": result["stage"]})
                    break
                if self.stop_on_critical_risk and self.risk_engine.has_critical() and stage.command_type in {"gpu_burn", "combined", "torch_ddp", "nccl"}:
                    self.events.write({"timestamp": now_iso(), "event": "stop_on_critical_risk", "stage": stage.name})
                    break
        except KeyboardInterrupt:
            self.interrupted = True
            self.events.write({"timestamp": now_iso(), "event": "user_interrupt", "suite": self.suite, "stage": self.current_stage})
            raise
        finally:
            self.events.write({"timestamp": now_iso(), "event": "suite_end", "suite": self.suite, "interrupted": self.interrupted})
        return {"suite": self.suite, "stages": completed, "interrupted": self.interrupted}

    def _run_stage(self, stage: StageSpec) -> dict[str, Any]:
        self.current_stage = stage.name
        started = now_iso()
        base = {
            "stage": stage.name,
            "title": stage.title,
            "command_type": stage.command_type,
            "required": stage.required,
            "enabled": stage.enabled,
            "cmd": shlex_join(stage.cmd) if stage.cmd else f"<internal:{stage.internal}>",
            "start": started,
            "end": None,
            "returncode": None,
            "status": "PENDING",
            "reason": "",
        }
        self.events.write({"timestamp": started, "event": "stage_start", "suite": self.suite, "stage": stage.name, "title": stage.title})
        if not stage.enabled:
            return self._finish_stage(base, "SKIPPED", "disabled by config")
        if stage.name == "torch_ddp" and not self._torch_ddp_available():
            severity = "HIGH" if stage.required else "INFO"
            self.risk_engine.add(
                severity,
                "SOFTWARE",
                "PyTorch distributed smoke test is unavailable" if stage.required else "Optional PyTorch distributed smoke test is unavailable",
                "Neither torchrun nor python3 -m torch.distributed.run is usable in this environment.",
                evidence="python3 -c 'import torch; import torch.distributed.run'",
                stage=stage.name,
                suggested_action="Install a CUDA-enabled PyTorch build and ensure torch.distributed.run works before formal acceptance.",
                dedupe_key=f"{stage.name}|torch_unavailable",
            )
            return self._finish_stage(base, "SKIPPED", "PyTorch distributed unavailable")
        if stage.tool and shutil.which(stage.tool) is None:
            severity = "WARN" if stage.required else "INFO"
            self.risk_engine.add(
                severity,
                "SOFTWARE",
                "Required test tool is missing" if stage.required else "Optional test tool is missing",
                f"Stage {stage.name} requires command '{stage.tool}', which was not found in PATH.",
                evidence=stage.tool,
                stage=stage.name,
                suggested_action="Install the tool or place it in PATH, then rerun this stage/suite.",
                dedupe_key=f"{stage.name}|missing_tool|{stage.tool}",
            )
            return self._finish_stage(base, "SKIPPED", f"missing tool: {stage.tool}")
        if stage.command_type in {"fio", "combined"}:
            fio_check = self._check_fio_safety(stage.name)
            if fio_check:
                return self._finish_stage(base, "SKIPPED", fio_check)
        if stage.internal:
            try:
                status, reason = self._run_internal(stage)
            except Exception as exc:
                self.risk_engine.add("HIGH", "SOFTWARE", "Internal stage failed", f"{stage.name} raised {exc}", stage=stage.name, suggested_action="Inspect acceptance tool traceback/logs.")
                status, reason = "FAIL", str(exc)
            return self._finish_stage(base, status, reason)
        if not stage.cmd:
            return self._finish_stage(base, "SKIPPED", "no command")
        result = self.command_runner.run(stage.cmd, stage.name, timeout=stage.timeout, dry_run=self.dry_run, shell=stage.shell)
        base["returncode"] = result.returncode
        if self.dry_run:
            return self._finish_stage(base, "SKIPPED", "dry-run")
        output = read_text_limited(Path(result.stdout_path)) + "\n" + read_text_limited(Path(result.stderr_path))
        self._evaluate_stage_output(stage, result, output)
        status = "PASS" if result.returncode == 0 and not result.timed_out else "FAIL"
        reason = "ok" if status == "PASS" else ("timeout" if result.timed_out else f"returncode={result.returncode}")
        if stage.command_type == "dmesg":
            # dmesg often returns non-zero on locked-down kernels; keep the risk text but do not fail hardware on permission alone.
            if "Operation not permitted" in output or "read kernel buffer failed" in output:
                self.risk_engine.add("WARN", "SYSTEM", "dmesg could not be read", "Kernel log access is restricted.", output[-1000:], stage.name, "Run preflight with sufficient permission if kernel risk scan is required.")
                status, reason = "WARN", "permission denied"
        return self._finish_stage(base, status, reason)

    def _finish_stage(self, row: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
        row["end"] = now_iso()
        row["status"] = status
        row["reason"] = reason
        self.stage_writer.write(row)
        self.events.write({"timestamp": row["end"], "event": "stage_end", "suite": self.suite, "stage": row["stage"], "status": status, "reason": reason})
        return row

    def _run_internal(self, stage: StageSpec) -> tuple[str, str]:
        if stage.internal == "inventory":
            inventory = collect_inventory(self.config, self.run_dir)
            self.risk_engine.evaluate_inventory(inventory, stage.name)
            return "PASS", "inventory collected"
        if stage.internal == "monitor_short":
            sec = int(self.config.get("timeouts.monitor_short_sec", 60))
            if self.dry_run:
                return "SKIPPED", f"dry-run sleep {sec}s"
            end = time.time() + sec
            while time.time() < end:
                time.sleep(min(1.0, end - time.time()))
            return "PASS", f"monitored {sec}s"
        return "SKIPPED", f"unknown internal stage {stage.internal}"

    def _evaluate_stage_output(self, stage: StageSpec, result: CommandResult, output: str) -> None:
        if stage.command_type == "dmesg":
            self.risk_engine.evaluate_dmesg(output, stage.name)
            return
        if stage.command_type == "ecc":
            self.risk_engine.evaluate_ecc_remap_text(output, stage.name)
            return
        if stage.command_type == "combined":
            if result.returncode != 0:
                self.risk_engine.add("CRITICAL", "SYSTEM", "Combined load returned non-zero", f"Return code: {result.returncode}", output[-3000:], stage.name, "Inspect raw combined load log and individual tool output.")
            for ctype in ("stress_ng", "gpu_burn", "fio"):
                self.risk_engine.evaluate_command_output(stage.name, ctype, output, result.returncode)
            return
        if stage.command_type == "fio" and result.timed_out:
            self.risk_engine.add(
                "HIGH",
                "DISK",
                "fio stage timed out",
                f"Stage {stage.name} exceeded timeout={result.timeout}s. This means the test did not finish in the configured window; inspect raw fio logs and kernel storage logs before treating it as a disk hardware failure.",
                output[-3000:],
                stage.name,
                "Check whether fio was creating/preconditioning files, whether fio.timeout_padding_sec is too small, and whether dmesg shows NVMe reset/I/O/media errors.",
            )
            return
        self.risk_engine.evaluate_command_output(stage.name, stage.command_type, output, result.returncode)

    def _check_fio_safety(self, stage_name: str) -> str:
        if not self.config.test_enabled("fio") and stage_name != "combined_load":
            return "fio disabled by config"
        if self.config.is_dangerous_fio_dir() and not self.force and not self.config.get("safety.allow_destructive_disk_test", False):
            self.risk_engine.add(
                "HIGH",
                "DISK",
                "Refusing fio on dangerous test directory",
                f"fio_test_dir={self.config.fio_test_dir} is considered unsafe for automated write tests.",
                evidence=str(self.config.fio_test_dir),
                stage=stage_name,
                suggested_action="Use a dedicated mounted test filesystem such as /mnt/test/fio_acceptance, or rerun with --force only after manual review.",
            )
            return "unsafe fio_test_dir"
        required = float(self.config.get("thresholds.disk_free_min_gb", 200))
        size_gb = parse_size_gb(str(self.config.get("fio.size", "100G")))
        required = max(required, size_gb)
        free = disk_free_gb(self.config.fio_test_dir)
        if free < required:
            self.risk_engine.add(
                "HIGH",
                "DISK",
                "Insufficient free space for fio",
                f"fio_test_dir={self.config.fio_test_dir} has {free:.1f} GB free; requires at least {required:.1f} GB.",
                evidence=str(self.config.fio_test_dir),
                stage=stage_name,
                suggested_action="Choose a larger dedicated test filesystem or reduce fio.size/runtime in acceptance.yaml.",
            )
            return "insufficient fio free space"
        if not self.dry_run:
            ensure_dir(self.config.fio_test_dir)
        return ""

    @staticmethod
    def _torch_ddp_available() -> bool:
        if shutil.which("torchrun"):
            return True
        code, _, _ = run_capture(["python3", "-c", "import torch; import torch.distributed.run"], timeout=15)
        return code == 0
