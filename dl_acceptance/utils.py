from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


NA_VALUES = {"", "N/A", "NA", "Not Supported", "[Not Supported]", "unknown", "Unknown", "None"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped in NA_VALUES else stripped
    return value


def safe_float(value: Any, default: float | None = None) -> float | None:
    value = normalize_value(value)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    number = safe_float(value, None)
    if number is None:
        return default
    return int(number)


def boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def json_dump_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False, sort_keys=False) + "\n")
    fp.flush()


class JsonlWriter:
    def __init__(self, path: Path):
        ensure_dir(path.parent)
        self.path = path
        self._fp = path.open("a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        json_dump_line(self._fp, obj)

    def close(self) -> None:
        self._fp.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"raw": line, "parse_error": True})
    return rows


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_text_limited(path: Path, max_bytes: int = 20 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fp:
        if size > max_bytes:
            fp.seek(max(0, size - max_bytes))
            prefix = f"\n[dl_server_acceptance: truncated first {size - max_bytes} bytes]\n"
        else:
            prefix = ""
        return prefix + fp.read().decode("utf-8", errors="replace")


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def command_path(command: str) -> str | None:
    return shutil.which(command)


def run_capture(cmd: list[str], timeout: int = 15) -> tuple[int | None, str, str]:
    if not cmd or shutil.which(cmd[0]) is None:
        return None, "", f"command not found: {cmd[0] if cmd else ''}"
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, _coerce_output(exc.stdout), _coerce_output(exc.stderr) + "\nTIMEOUT"
    except Exception as exc:  # pragma: no cover - defensive for odd platform errors
        return None, "", str(exc)


def command_version(command: str) -> str | None:
    path = shutil.which(command)
    if path is None:
        return None
    version_args = {
        "nvidia-smi": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        "dcgmi": ["dcgmi", "--version"],
        "stress-ng": ["stress-ng", "--version"],
        "memtester": ["memtester", "--version"],
        "fio": ["fio", "--version"],
        "smartctl": ["smartctl", "--version"],
        "nvme": ["nvme", "--version"],
        "ipmitool": ["ipmitool", "-V"],
        "torchrun": ["torchrun", "--help"],
        "python3": ["python3", "--version"],
        "bash": ["bash", "--version"],
        "dmesg": ["dmesg", "--version"],
        "journalctl": ["journalctl", "--version"],
        "lspci": ["lspci", "--version"],
        "lsblk": ["lsblk", "--version"],
    }
    cmd = version_args.get(command)
    if cmd is None:
        return None
    code, out, err = run_capture(cmd, timeout=8)
    text = (out or err).strip().splitlines()
    if code is None or not text:
        return None
    return text[0][:200]


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def shlex_join(cmd: list[str] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    try:
        import shlex

        return shlex.join([str(part) for part in cmd])
    except Exception:
        return " ".join(str(part) for part in cmd)


def disk_free_gb(path: Path) -> float:
    existing = path
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return usage.free / (1024**3)


def total_memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return None
        for line in meminfo.read_text(errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                kb = safe_float(line)
                return kb / (1024**2) if kb is not None else None
    return None


def available_memory_mb(fraction: float) -> int:
    try:
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available
    except Exception:
        avail = 0
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(errors="replace").splitlines():
                if line.startswith("MemAvailable:"):
                    kb = safe_float(line, 0) or 0
                    avail = int(kb * 1024)
                    break
    return max(128, int(avail * fraction / (1024**2)))


def cpu_model() -> str:
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def hostname() -> str:
    return os.uname().nodename if hasattr(os, "uname") else os.environ.get("HOSTNAME", "unknown")


def load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def append_csv_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in header})
        fp.flush()


def severity_rank(severity: str) -> int:
    return {"INFO": 0, "WARN": 1, "HIGH": 2, "CRITICAL": 3}.get(severity.upper(), 0)


def format_gb(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "unknown"
    return f"{number:.1f} GB"


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def grep_any(lines: Iterable[str], patterns: Iterable[str], flags: int = re.IGNORECASE) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    compiled = [(pat, re.compile(pat, flags)) for pat in patterns]
    for line in lines:
        for pat, rx in compiled:
            if rx.search(line):
                hits.append((pat, line.rstrip()))
    return hits
