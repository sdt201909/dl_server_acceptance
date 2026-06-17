from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .utils import deep_merge, disk_free_gb


DEFAULT_CONFIG: dict[str, Any] = {
    "expected": {
        "gpu_count": 4,
        "gpu_name_regex": "(RTX PRO 6000|PRO 6000|NVIDIA.*6000)",
        "gpu_memory_gb_min": 90,
        "gpu_memory_gb_expected": 96,
    },
    "paths": {
        "work_dir": "./acceptance_runs",
        "fio_test_dir": "/mnt/test/fio_acceptance",
        "tools_dir": "./tools",
    },
    "monitor": {
        "interval_sec": 5,
        "capture_dmesg": True,
        "capture_journalctl": False,
        "dashboard": True,
    },
    "thresholds": {
        "gpu_temp_warn_c": 82,
        "gpu_temp_crit_c": 88,
        "gpu_peer_temp_delta_warn_c": 12,
        "gpu_util_under_load_min_pct": 90,
        "gpu_util_under_load_grace_sec": 120,
        "gpu_power_peer_ratio_warn": 0.70,
        "gpu_memory_error_crit": 1,
        "cpu_load_warn_ratio": 0.95,
        "mem_available_min_gb": 8,
        "disk_free_min_gb": 200,
        "nccl_bandwidth_outlier_ratio_warn": 0.65,
        "nvbandwidth_outlier_ratio_warn": 0.65,
    },
    "timeouts": {
        "command_graceful_shutdown_sec": 15,
        "monitor_short_sec": 60,
        "dcgm_r1_sec": 300,
        "dcgm_r2_sec": 900,
        "dcgm_r3_sec": 3600,
        "dcgm_r4_sec": 7200,
        "gpu_burn_sec_quick": 600,
        "gpu_burn_sec_standard": 7200,
        "combined_sec_full": 21600,
        "combined_sec_burnin": 86400,
    },
    "tests": {
        "enable_cpu_stress": True,
        "enable_memtester": True,
        "enable_fio": True,
        "enable_smart": True,
        "enable_dcgm": True,
        "enable_dcgm_r4": False,
        "enable_gpu_burn": True,
        "enable_cuda_memtest": True,
        "enable_nvbandwidth": True,
        "enable_nccl_tests": True,
        "enable_torch_ddp": True,
        "enable_combined": True,
        "enable_ipmi": True,
    },
    "safety": {
        "allow_destructive_disk_test": False,
        "allow_gpu_setting_changes": False,
        "stop_on_critical_risk": True,
        "require_confirm_for_full_or_burnin": True,
    },
    "fio": {
        "size": "100G",
        "runtime_sec": 1800,
        "direct": True,
        "numjobs": 4,
        "iodepth": 32,
        "bs": "1M",
    },
    "memtester": {
        "memory_fraction": 0.80,
        "passes": 2,
    },
    "stress_ng": {
        "cpu_method": "matrixprod",
        "cpu_timeout_sec": 7200,
        "vm_workers": 8,
        "vm_bytes_fraction": 0.60,
    },
    "nccl": {
        "min_bytes": "8",
        "max_bytes": "8G",
        "factor": 2,
        "gpus": 4,
    },
    "torch_ddp": {
        "nproc_per_node": 4,
        "matrix_size": 2048,
        "iterations": 20,
        "dtype": "fp16",
    },
    "gpu_burn": {
        "memory": "90%",
        "use_tensor_cores": True,
    },
    "commands": {},
}


DANGEROUS_FIO_DIRS = {
    Path("/"),
    Path("/home"),
    Path("/var"),
    Path("/usr"),
    Path("/etc"),
    Path("/boot"),
    Path("/root"),
    Path("/opt"),
    Path("/tmp"),
}


class AcceptanceConfig:
    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.path = path
        self.data = deep_merge(copy.deepcopy(DEFAULT_CONFIG), data or {})
        self.base_dir = path.parent.resolve() if path else Path.cwd()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AcceptanceConfig":
        cfg_path = Path(path).expanduser() if path else Path("acceptance.yaml")
        if not cfg_path.exists():
            return cls({}, None)
        try:
            import yaml  # type: ignore
        except Exception as exc:
            loaded = _load_simple_yaml(cfg_path)
            return cls(loaded, cfg_path.resolve())
        with cfg_path.open("r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a YAML mapping: {cfg_path}")
        return cls(loaded, cfg_path.resolve())

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self.data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path_value(self, dotted: str) -> Path:
        raw = str(self.get(dotted))
        p = Path(os.path.expandvars(raw)).expanduser()
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return p

    @property
    def work_dir(self) -> Path:
        return self.path_value("paths.work_dir")

    @property
    def fio_test_dir(self) -> Path:
        return self.path_value("paths.fio_test_dir")

    @property
    def tools_dir(self) -> Path:
        return self.path_value("paths.tools_dir")

    @property
    def monitor_interval(self) -> int:
        return int(self.get("monitor.interval_sec", 5))

    @property
    def graceful_shutdown_sec(self) -> int:
        return int(self.get("timeouts.command_graceful_shutdown_sec", 15))

    def command_override(self, stage: str) -> list[str] | str | None:
        commands = self.get("commands", {})
        if isinstance(commands, dict):
            return commands.get(stage)
        return None

    def test_enabled(self, name: str) -> bool:
        return bool(self.get(f"tests.enable_{name}", True))

    def is_dangerous_fio_dir(self) -> bool:
        path = self.fio_test_dir.resolve()
        if path in DANGEROUS_FIO_DIRS:
            return True
        for bad in DANGEROUS_FIO_DIRS:
            try:
                if path == bad or path.is_relative_to(bad) and bad in {Path("/home"), Path("/var"), Path("/tmp")}:
                    return True
            except AttributeError:  # pragma: no cover - Python 3.8 compatibility
                if str(path).startswith(str(bad) + os.sep) and bad in {Path("/home"), Path("/var"), Path("/tmp")}:
                    return True
        return False

    def fio_free_gb(self) -> float:
        return disk_free_gb(self.fio_test_dir)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Tiny fallback parser for the simple mapping style used by the example config.

    It intentionally supports only nested dictionaries and scalar values. Install
    PyYAML for full YAML support, especially command override lists.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, original in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_yaml_comment(original).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.startswith("- "):
            raise RuntimeError(
                f"PyYAML is not installed and fallback parser cannot parse YAML lists at {path}:{lineno}. "
                "Install with: pip install pyyaml"
            )
        if ":" not in stripped:
            raise RuntimeError(f"Cannot parse YAML line {path}:{lineno}: {original}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise RuntimeError(f"Invalid indentation at {path}:{lineno}: {original}")
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    out = []
    for ch in line:
        if ch == "\\" and in_double and not escaped:
            escaped = True
            out.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and not escaped:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
        escaped = False
    return "".join(out)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
