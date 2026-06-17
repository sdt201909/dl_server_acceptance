from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import AcceptanceConfig
from .parsers import parse_nvidia_smi_csv
from .risks import RiskEngine
from .utils import append_csv_row, disk_free_gb, ensure_dir, load_average, now_iso, safe_float


GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=timestamp,index,name,uuid,serial,pci.bus_id,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,utilization.gpu,utilization.memory,memory.total,memory.used,ecc.mode.current",
    "--format=csv,noheader,nounits",
]


SYSTEM_HEADER = [
    "timestamp",
    "stage",
    "cpu_percent",
    "load1",
    "load5",
    "load15",
    "mem_total_gb",
    "mem_used_gb",
    "mem_available_gb",
    "disk_path",
    "disk_free_gb",
    "net_bytes_sent",
    "net_bytes_recv",
]

GPU_HEADER = [
    "timestamp",
    "stage",
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "ecc.mode.current",
]


class MetricsSampler:
    def __init__(
        self,
        config: AcceptanceConfig,
        run_dir: Path,
        risk_engine: RiskEngine | None = None,
        suite: str = "",
        stage_getter: Callable[[], str] | None = None,
        active_command_getter: Callable[[], str | None] | None = None,
        interval: int | None = None,
        dashboard: bool = True,
    ):
        self.config = config
        self.run_dir = run_dir
        self.metrics_dir = ensure_dir(run_dir / "metrics")
        self.risk_engine = risk_engine
        self.suite = suite
        self.stage_getter = stage_getter or (lambda: "")
        self.active_command_getter = active_command_getter or (lambda: None)
        self.interval = interval or config.monitor_interval
        self.dashboard = dashboard
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] = {}
        self._rich_live = None
        self._rich_console = None
        self._nvml_available = False
        self._nvml = None
        self._nvml_handles = []
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._nvml = pynvml
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2, self.interval + 2))
        if self._rich_live:
            self._rich_live.stop()
        if self._nvml_available and self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def sample_once(self) -> dict[str, Any]:
        stage = self.stage_getter()
        system = self._sample_system(stage)
        gpus, gpu_error = self._sample_gpus()
        snapshot = {"timestamp": now_iso(), "stage": stage, "system": system, "gpus": gpus, "gpu_error": gpu_error}
        append_csv_row(self.metrics_dir / "system_metrics.csv", SYSTEM_HEADER, system)
        for gpu in gpus:
            row = {"timestamp": snapshot["timestamp"], "stage": stage}
            row.update(gpu)
            append_csv_row(self.metrics_dir / "gpu_metrics.csv", GPU_HEADER, row)
        if self.risk_engine:
            self.risk_engine.evaluate_metrics(snapshot, stage)
        self._latest = snapshot
        return snapshot

    def _loop(self) -> None:
        rich_ok = False
        if self.dashboard:
            try:
                from rich.console import Console  # type: ignore
                from rich.live import Live  # type: ignore

                self._rich_console = Console()
                self._rich_live = Live(self._render_rich(), console=self._rich_console, refresh_per_second=1)
                self._rich_live.start()
                rich_ok = True
            except Exception:
                rich_ok = False
        while not self._stop.is_set():
            snapshot = self.sample_once()
            if rich_ok and self._rich_live:
                self._rich_live.update(self._render_rich())
            else:
                self._print_plain(snapshot)
            self._stop.wait(self.interval)

    def _sample_system(self, stage: str) -> dict[str, Any]:
        load = load_average() or (None, None, None)
        disk_path = str(self.config.fio_test_dir)
        row: dict[str, Any] = {
            "timestamp": now_iso(),
            "stage": stage,
            "cpu_percent": None,
            "load1": load[0],
            "load5": load[1],
            "load15": load[2],
            "mem_total_gb": None,
            "mem_used_gb": None,
            "mem_available_gb": None,
            "disk_path": disk_path,
            "disk_free_gb": disk_free_gb(Path(disk_path)),
            "net_bytes_sent": None,
            "net_bytes_recv": None,
        }
        try:
            import psutil  # type: ignore

            mem = psutil.virtual_memory()
            net = psutil.net_io_counters()
            row.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "mem_total_gb": mem.total / (1024**3),
                    "mem_used_gb": mem.used / (1024**3),
                    "mem_available_gb": mem.available / (1024**3),
                    "net_bytes_sent": net.bytes_sent,
                    "net_bytes_recv": net.bytes_recv,
                }
            )
        except Exception:
            row.update(self._sample_proc_mem())
        return row

    @staticmethod
    def _sample_proc_mem() -> dict[str, Any]:
        out: dict[str, Any] = {}
        path = Path("/proc/meminfo")
        if not path.exists():
            return out
        values = {}
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = safe_float(parts[1])
        total = values.get("MemTotal")
        avail = values.get("MemAvailable")
        if total is not None:
            out["mem_total_gb"] = total / (1024**2)
        if avail is not None:
            out["mem_available_gb"] = avail / (1024**2)
        if total is not None and avail is not None:
            out["mem_used_gb"] = (total - avail) / (1024**2)
        return out

    def _sample_gpus(self) -> tuple[list[dict[str, Any]], str]:
        if self._nvml_available and self._nvml:
            try:
                rows = []
                for idx, handle in enumerate(self._nvml_handles):
                    name = self._nvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode(errors="replace")
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    power = None
                    power_limit = None
                    try:
                        power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        power_limit = self._nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                    except Exception:
                        pass
                    rows.append(
                        {
                            "index": idx,
                            "name": name,
                            "uuid": self._decode(self._nvml.nvmlDeviceGetUUID(handle)),
                            "pci.bus_id": self._decode(self._nvml.nvmlDeviceGetPciInfo(handle).busId),
                            "temperature.gpu": self._safe_nvml(lambda: self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU)),
                            "power.draw": power,
                            "power.limit": power_limit,
                            "clocks.sm": self._safe_nvml(lambda: self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_SM)),
                            "clocks.mem": self._safe_nvml(lambda: self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_MEM)),
                            "pstate": self._safe_nvml(lambda: f"P{self._nvml.nvmlDeviceGetPowerState(handle)}"),
                            "utilization.gpu": util.gpu,
                            "utilization.memory": util.memory,
                            "memory.total": mem.total / (1024**2),
                            "memory.used": mem.used / (1024**2),
                            "ecc.mode.current": None,
                        }
                    )
                return rows, ""
            except Exception as exc:
                return self._sample_gpus_nvidia_smi(f"NVML failed: {exc}")
        return self._sample_gpus_nvidia_smi("NVML unavailable")

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _safe_nvml(func) -> Any:
        try:
            return func()
        except Exception:
            return None

    def _sample_gpus_nvidia_smi(self, prefix_error: str = "") -> tuple[list[dict[str, Any]], str]:
        try:
            proc = subprocess.run(GPU_QUERY, text=True, capture_output=True, timeout=10)
            if proc.returncode != 0:
                return [], (prefix_error + "; " if prefix_error else "") + proc.stderr.strip()
            return parse_nvidia_smi_csv(proc.stdout), prefix_error
        except Exception as exc:
            return [], (prefix_error + "; " if prefix_error else "") + str(exc)

    def _render_rich(self):
        from rich import box  # type: ignore
        from rich.layout import Layout  # type: ignore
        from rich.panel import Panel  # type: ignore
        from rich.table import Table  # type: ignore

        snap = self._latest or {"system": {}, "gpus": []}
        system = snap.get("system", {})
        risks = self.risk_engine.risks[-10:] if self.risk_engine else []
        counts = self.risk_engine.counts() if self.risk_engine else {}
        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_row(f"Suite: {self.suite or '-'} | Stage: {snap.get('stage', '-')}", f"Active: {self.active_command_getter() or '-'}")
        top.add_row(
            f"CPU {system.get('cpu_percent', '?')}% | Load {system.get('load1', '?')}",
            f"Mem avail {self._fmt(system.get('mem_available_gb'))} GB | Disk free {self._fmt(system.get('disk_free_gb'))} GB",
        )
        gpu_table = Table(title="GPU", box=box.SIMPLE_HEAVY)
        for col in ["idx", "name", "bus", "temp", "power", "util", "mem", "pstate", "sm/mem clk"]:
            gpu_table.add_column(col)
        for gpu in snap.get("gpus", []):
            gpu_table.add_row(
                str(gpu.get("index", "")),
                str(gpu.get("name", ""))[:28],
                str(gpu.get("pci.bus_id", "")),
                self._fmt(gpu.get("temperature.gpu")),
                f"{self._fmt(gpu.get('power.draw'))}/{self._fmt(gpu.get('power.limit'))}",
                f"{self._fmt(gpu.get('utilization.gpu'))}%",
                f"{self._fmt(gpu.get('memory.used'))}/{self._fmt(gpu.get('memory.total'))} MiB",
                str(gpu.get("pstate", "")),
                f"{self._fmt(gpu.get('clocks.sm'))}/{self._fmt(gpu.get('clocks.mem'))}",
            )
        risk_table = Table(title=f"Recent risks {counts}", box=box.SIMPLE)
        for col in ["sev", "category", "stage", "title"]:
            risk_table.add_column(col)
        for risk in risks:
            risk_table.add_row(risk.severity, risk.category, risk.stage, risk.title[:70])
        layout = Layout()
        layout.split_column(Layout(Panel(top, title="dl_server_acceptance"), size=5), Layout(gpu_table), Layout(risk_table, size=14))
        return layout

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "?"
        try:
            return f"{float(value):.1f}"
        except Exception:
            return str(value)

    def _print_plain(self, snapshot: dict[str, Any]) -> None:
        system = snapshot.get("system", {})
        gpus = snapshot.get("gpus", [])
        gpu_bits = []
        for gpu in gpus:
            gpu_bits.append(
                f"GPU{gpu.get('index')} temp={self._fmt(gpu.get('temperature.gpu'))}C util={self._fmt(gpu.get('utilization.gpu'))}% power={self._fmt(gpu.get('power.draw'))}W"
            )
        print(
            f"[{snapshot.get('timestamp')}] stage={snapshot.get('stage')} cpu={system.get('cpu_percent')}% "
            f"load1={system.get('load1')} mem_avail={self._fmt(system.get('mem_available_gb'))}GB "
            f"disk_free={self._fmt(system.get('disk_free_gb'))}GB {' | '.join(gpu_bits)}",
            flush=True,
        )

