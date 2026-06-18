from __future__ import annotations

import csv
import io
import json
import re
from statistics import median
from typing import Any

from .utils import normalize_value, safe_float, safe_int


NVIDIA_SMI_QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "serial",
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


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reader = csv.reader(io.StringIO(text.strip()))
    for raw in reader:
        if not raw or all(not item.strip() for item in raw):
            continue
        if raw[0].strip().lower() == "timestamp":
            continue
        values = [normalize_value(item) for item in raw]
        item: dict[str, Any] = {}
        for idx, field in enumerate(NVIDIA_SMI_QUERY_FIELDS):
            value = values[idx] if idx < len(values) else None
            item[field] = value
        for key in ["index"]:
            item[key] = safe_int(item.get(key))
        for key in [
            "temperature.gpu",
            "power.draw",
            "power.limit",
            "clocks.sm",
            "clocks.mem",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
        ]:
            item[key] = safe_float(item.get(key))
        rows.append(item)
    return rows


def parse_dcgm_output(text: str) -> dict[str, Any]:
    failures: list[str] = []
    skips: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if re.search(r"\b(fail|failed)\b", lower) and "no failures" not in lower:
            failures.append(line.strip())
        if re.search(r"\b(skip|skipped|not supported)\b", lower):
            skips.append(line.strip())
    return {
        "failed": bool(failures),
        "skipped": bool(skips),
        "failures": failures,
        "skips": skips,
    }


def parse_gpu_burn_output(text: str) -> dict[str, Any]:
    bad_lines = []
    for line in text.splitlines():
        if re.search(r"\b(error|fault|failed|compare|calculation error|hardware)\b", line, re.IGNORECASE):
            bad_lines.append(line.strip())
    return {"failed": bool(bad_lines), "bad_lines": bad_lines}


def parse_cuda_memtest_output(text: str) -> dict[str, Any]:
    bad_lines = []
    for line in text.splitlines():
        if re.search(r"\b(error|fail|failed|failure|mismatch)\b", line, re.IGNORECASE):
            bad_lines.append(line.strip())
    return {"failed": bool(bad_lines), "bad_lines": bad_lines}


def parse_nccl_tests_output(text: str) -> dict[str, Any]:
    bad_lines: list[str] = []
    rows: list[dict[str, Any]] = []
    header = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if stripped.startswith("#"):
            if "algbw" in lower and "busbw" in lower:
                header = stripped
            continue
        if re.search(
            r"\b(failed|timeout|unhandled system error|internal error|ncclsystemerror|ncclunhandledcudaerror)\b",
            lower,
        ):
            bad_lines.append(stripped)
        if re.search(r"\bwrong\b", lower):
            bad_lines.append(stripped)
        parts = stripped.split()
        if not parts or not re.match(r"^\d+$", parts[0]):
            continue
        numeric = [safe_float(part) for part in parts]
        wrong_counts = [safe_int(part) for part in parts if re.match(r"^\d+$", part)]
        if wrong_counts and wrong_counts[-1] and wrong_counts[-1] > 0:
            bad_lines.append(stripped)
        floats = [num for num in numeric if num is not None]
        algbw = None
        busbw = None
        if len(parts) >= 8:
            # nccl-tests keeps algbw/busbw near the end; exact offsets vary by collective.
            candidate = []
            for part in parts[2:]:
                val = safe_float(part)
                if val is not None:
                    candidate.append(val)
            if len(candidate) >= 3:
                algbw = candidate[-3] if len(candidate) >= 4 else candidate[-2]
                busbw = candidate[-2] if len(candidate) >= 4 else candidate[-1]
        rows.append({"raw": stripped, "algbw": algbw, "busbw": busbw, "header": header, "values": floats})
    return {"failed": bool(bad_lines), "bad_lines": bad_lines, "rows": rows}


def parse_nvbandwidth_output(text: str) -> dict[str, Any]:
    bad_lines: list[str] = []
    matrix_rows: list[list[float]] = []
    try:
        data = json.loads(text)
        return {"json": data, "failed": False, "bad_lines": [], "matrix": []}
    except Exception:
        pass
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"\b(error|failed|failure|timeout)\b", stripped, re.IGNORECASE):
            bad_lines.append(stripped)
        numbers = [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+\.\d+|(?<![A-Za-z])[-+]?\d+", stripped)]
        if len(numbers) >= 2:
            matrix_rows.append(numbers)
    return {"failed": bool(bad_lines), "bad_lines": bad_lines, "matrix": matrix_rows}


def parse_fio_json_or_text(text: str) -> dict[str, Any]:
    errors: list[str] = []
    data: Any = None
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            for job in data.get("jobs", []):
                err = job.get("error", 0)
                if err:
                    errors.append(f"job {job.get('jobname', 'unknown')} error={err}")
                for section in ("read", "write", "trim"):
                    if isinstance(job.get(section), dict) and job[section].get("io_bytes", 0) == 0:
                        # This is informational; not all jobs exercise all directions.
                        pass
        except json.JSONDecodeError as exc:
            errors.append(f"fio JSON parse error: {exc}")
    for match in re.finditer(r"\berr=\s*([0-9]+)", text):
        if int(match.group(1)) != 0:
            errors.append(match.group(0))
    if re.search(r"\b(error|failed|failure)\b", text, re.IGNORECASE) and not errors:
        errors.append("fio output contains error/failure")
    return {"failed": bool(errors), "errors": errors, "json": data}


def parse_smartctl_output(text: str) -> dict[str, Any]:
    critical: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if "critical warning" in lower and not re.search(r":\s*0x0+\b|:\s*0\b", lower):
            critical.append(line.strip())
        if re.search(r"media.*errors|media and data integrity errors", lower):
            number = safe_int(line)
            if number and number > 0:
                critical.append(line.strip())
        if re.search(r"\b(prefail|failed|failure|critical)\b", lower):
            if "passed" not in lower and "no critical warning" not in lower:
                critical.append(line.strip())
    return {"failed": bool(critical), "critical": critical}


CRITICAL_DMESG_PATTERNS = [
    r"NVRM:\s*Xid",
    r"GPU has fallen off the bus",
    r"fallen off the bus",
    r"GPU is lost",
    r"AER:\s*Uncorrected",
    r"\bMCE\b.*\b(error|fail|failed|failure|critical|uncorrected|exception|logged)\b",
    r"Machine Check Exception",
    r"Machine check events logged",
    r"hardware error",
    r"memory error",
]

HIGH_DMESG_PATTERNS = [
    r"AER:\s*Corrected",
    r"PCIe Bus Error.*corrected",
    r"\bEDAC\b.*\b(error|fail|failed|failure|warning|critical|corrected|uncorrected|CE|UE)\b",
    r"\b(error|fail|failed|failure|warning|critical|corrected|uncorrected|CE|UE)\b.*\bEDAC\b",
    r"thermal throttling",
]


def parse_dmesg_lines(text_or_lines: str | list[str]) -> list[dict[str, Any]]:
    lines = text_or_lines.splitlines() if isinstance(text_or_lines, str) else text_or_lines
    risks: list[dict[str, Any]] = []
    for line in lines:
        for pattern in CRITICAL_DMESG_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                risks.append(
                    {
                        "severity": "CRITICAL",
                        "category": "SYSTEM",
                        "title": "Kernel log contains critical hardware/GPU error",
                        "details": f"Matched pattern: {pattern}",
                        "evidence": line.strip(),
                    }
                )
        for pattern in HIGH_DMESG_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                risks.append(
                    {
                        "severity": "HIGH",
                        "category": "SYSTEM",
                        "title": "Kernel log contains corrected hardware/thermal error",
                        "details": f"Matched pattern: {pattern}",
                        "evidence": line.strip(),
                    }
                )
    return risks


def bandwidth_outliers(values: list[float], ratio: float) -> list[dict[str, Any]]:
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 3:
        return []
    med = median(clean)
    if med <= 0:
        return []
    return [{"value": value, "median": med, "ratio": value / med} for value in clean if value < med * ratio]
