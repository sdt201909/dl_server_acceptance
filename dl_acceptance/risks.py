from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from . import parsers
from .config import AcceptanceConfig
from .utils import JsonlWriter, ensure_dir, now_iso, safe_float, severity_rank


@dataclass
class Risk:
    timestamp: str
    severity: str
    category: str
    title: str
    details: str
    evidence: str
    stage: str
    suggested_action: str


class RiskEngine:
    LOAD_STAGES = {"gpu_burn", "combined", "torch_ddp", "nccl", "nccl_all_reduce", "nccl_all_gather", "nccl_reduce_scatter"}

    def __init__(self, config: AcceptanceConfig, run_dir: Path | None = None):
        self.config = config
        self.run_dir = run_dir
        self.risks: list[Risk] = []
        self._seen: set[str] = set()
        self._low_util_since: dict[tuple[str, int], float] = {}
        self._baseline_gpu_count: int | None = None
        self._writers: list[JsonlWriter] = []
        if run_dir:
            ensure_dir(run_dir)
            ensure_dir(run_dir / "metrics")
            self._writers = [JsonlWriter(run_dir / "risks.jsonl"), JsonlWriter(run_dir / "metrics" / "risks.jsonl")]

    def close(self) -> None:
        for writer in self._writers:
            writer.close()

    def add(
        self,
        severity: str,
        category: str,
        title: str,
        details: str,
        evidence: str = "",
        stage: str = "",
        suggested_action: str = "",
        dedupe_key: str | None = None,
    ) -> Risk:
        severity = severity.upper()
        key = dedupe_key or f"{stage}|{severity}|{category}|{title}|{evidence[:160]}"
        if key in self._seen:
            return self.risks[-1] if self.risks else Risk(now_iso(), severity, category, title, details, evidence, stage, suggested_action)
        self._seen.add(key)
        risk = Risk(
            timestamp=now_iso(),
            severity=severity,
            category=category,
            title=title,
            details=details,
            evidence=evidence,
            stage=stage,
            suggested_action=suggested_action,
        )
        self.risks.append(risk)
        row = asdict(risk)
        for writer in self._writers:
            writer.write(row)
        return risk

    def counts(self) -> dict[str, int]:
        counts = {"INFO": 0, "WARN": 0, "HIGH": 0, "CRITICAL": 0}
        for risk in self.risks:
            counts[risk.severity] = counts.get(risk.severity, 0) + 1
        return counts

    def highest(self) -> str:
        if not self.risks:
            return "INFO"
        return max((risk.severity for risk in self.risks), key=severity_rank)

    def has_high_or_critical(self) -> bool:
        return any(severity_rank(r.severity) >= severity_rank("HIGH") for r in self.risks)

    def has_critical(self) -> bool:
        return any(r.severity == "CRITICAL" for r in self.risks)

    def evaluate_inventory(self, inventory: dict[str, Any], stage: str = "inventory") -> None:
        expected_count = int(self.config.get("expected.gpu_count", 0))
        expected_name = str(self.config.get("expected.gpu_name_regex", ".*"))
        min_mem_gb = float(self.config.get("expected.gpu_memory_gb_min", 0))
        gpus = inventory.get("gpus", []) or []
        self._baseline_gpu_count = len(gpus)
        tools = inventory.get("tools", {})
        nvidia_smi = tools.get("nvidia-smi", {})
        if not nvidia_smi.get("found"):
            self.add(
                "CRITICAL",
                "SOFTWARE",
                "nvidia-smi is unavailable",
                "GPU inventory and health cannot be verified without nvidia-smi.",
                evidence=str(nvidia_smi),
                stage=stage,
                suggested_action="Install/repair NVIDIA driver and verify nvidia-smi before acceptance.",
            )
        if expected_count and len(gpus) != expected_count:
            self.add(
                "CRITICAL",
                "GPU",
                "GPU count does not match expected configuration",
                f"Expected {expected_count} GPU(s), detected {len(gpus)}.",
                evidence=", ".join(str(g.get("name", "unknown")) for g in gpus),
                stage=stage,
                suggested_action="Check physical installation, BIOS PCIe settings, driver, and vendor BOM.",
            )
        for gpu in gpus:
            name = str(gpu.get("name") or "")
            idx = gpu.get("index")
            if expected_name and not re.search(expected_name, name, re.IGNORECASE):
                self.add(
                    "HIGH",
                    "GPU",
                    "GPU model does not match expected regex",
                    f"GPU {idx} name '{name}' does not match {expected_name}.",
                    evidence=str(gpu),
                    stage=stage,
                    suggested_action="Verify delivered GPU SKU with supplier and update config only if intentionally different.",
                )
            mem_mib = safe_float(gpu.get("memory.total"))
            mem_gb = mem_mib / 1024 if mem_mib is not None else safe_float(gpu.get("memory_gb"))
            if mem_gb is not None and mem_gb < min_mem_gb:
                self.add(
                    "HIGH",
                    "GPU",
                    "GPU memory is lower than expected",
                    f"GPU {idx} reports {mem_gb:.1f} GB, below configured minimum {min_mem_gb:.1f} GB.",
                    evidence=str(gpu),
                    stage=stage,
                    suggested_action="Verify SKU, driver reporting, and whether MIG/vGPU/firmware settings are limiting memory.",
                )
        driver = inventory.get("driver", {})
        if driver.get("query_failed"):
            self.add(
                "HIGH",
                "SOFTWARE",
                "Driver/CUDA/NVML query failed",
                "Driver/CUDA information could not be fully queried.",
                evidence=str(driver),
                stage=stage,
                suggested_action="Check NVIDIA driver installation and CUDA runtime visibility.",
            )

    def evaluate_metrics(self, snapshot: dict[str, Any], stage: str) -> None:
        gpus = snapshot.get("gpus", []) or []
        if not gpus:
            if self._baseline_gpu_count:
                self.add(
                    "CRITICAL",
                    "GPU",
                    "GPU disappeared during monitoring",
                    f"Expected to keep seeing {self._baseline_gpu_count} GPU(s), but current sample has none.",
                    evidence=str(snapshot.get("gpu_error", "")),
                    stage=stage,
                    suggested_action="Stop stress tests; collect dmesg/NVIDIA logs and contact supplier.",
                    dedupe_key=f"{stage}|gpu_disappeared_all",
                )
            return
        expected_count = int(self.config.get("expected.gpu_count", 0))
        if expected_count and len(gpus) < expected_count:
            self.add(
                "CRITICAL",
                "GPU",
                "GPU count dropped during run",
                f"Expected {expected_count}, current sample has {len(gpus)}.",
                evidence=str([g.get("index") for g in gpus]),
                stage=stage,
                suggested_action="Stop high-load tests and inspect PCIe/power/cooling/NVIDIA Xid logs.",
                dedupe_key=f"{stage}|gpu_count_drop",
            )
        warn_temp = float(self.config.get("thresholds.gpu_temp_warn_c", 82))
        crit_temp = float(self.config.get("thresholds.gpu_temp_crit_c", 88))
        peer_delta = float(self.config.get("thresholds.gpu_peer_temp_delta_warn_c", 12))
        temps = [safe_float(g.get("temperature.gpu")) for g in gpus if safe_float(g.get("temperature.gpu")) is not None]
        temp_median = median(temps) if temps else None
        for gpu in gpus:
            idx = gpu.get("index")
            temp = safe_float(gpu.get("temperature.gpu"))
            if temp is not None and temp >= crit_temp:
                self.add(
                    "CRITICAL",
                    "THERMAL",
                    "GPU temperature exceeds critical threshold",
                    f"GPU {idx} temperature {temp:.1f} C >= {crit_temp:.1f} C.",
                    evidence=str(gpu),
                    stage=stage,
                    suggested_action="Stop high-load test, inspect airflow/fans/heatsink seating, and collect vendor logs.",
                    dedupe_key=f"{stage}|gpu{idx}|temp_crit",
                )
            elif temp is not None and temp >= warn_temp:
                self.add(
                    "WARN",
                    "THERMAL",
                    "GPU temperature exceeds warning threshold",
                    f"GPU {idx} temperature {temp:.1f} C >= {warn_temp:.1f} C.",
                    evidence=str(gpu),
                    stage=stage,
                    suggested_action="Check chassis airflow, room temperature, fan policy, and neighboring card temperatures.",
                    dedupe_key=f"{stage}|gpu{idx}|temp_warn",
                )
            if temp is not None and temp_median is not None and temp - temp_median >= peer_delta:
                sev = "HIGH" if temp - temp_median >= peer_delta * 1.5 else "WARN"
                self.add(
                    sev,
                    "THERMAL",
                    "GPU temperature is high relative to peer median",
                    f"GPU {idx} is {temp - temp_median:.1f} C above peer median {temp_median:.1f} C.",
                    evidence=str(gpu),
                    stage=stage,
                    suggested_action="Compare slot airflow and card seating; inspect whether one GPU is thermally disadvantaged.",
                    dedupe_key=f"{stage}|gpu{idx}|temp_peer_delta|{sev}",
                )
        load_stage = any(token in stage for token in self.LOAD_STAGES)
        if load_stage:
            min_util = float(self.config.get("thresholds.gpu_util_under_load_min_pct", 90))
            grace = float(self.config.get("thresholds.gpu_util_under_load_grace_sec", 120))
            now = time.time()
            powers = [safe_float(g.get("power.draw")) for g in gpus if safe_float(g.get("power.draw")) and safe_float(g.get("power.draw")) > 20]
            power_med = median(powers) if len(powers) >= 2 else None
            power_ratio = float(self.config.get("thresholds.gpu_power_peer_ratio_warn", 0.70))
            for gpu in gpus:
                idx = int(gpu.get("index") or 0)
                util = safe_float(gpu.get("utilization.gpu"))
                key = (stage, idx)
                if util is not None and util < min_util:
                    self._low_util_since.setdefault(key, now)
                    if now - self._low_util_since[key] >= grace:
                        self.add(
                            "HIGH",
                            "GPU",
                            "GPU utilization remains low during load stage",
                            f"GPU {idx} utilization {util:.1f}% stayed below {min_util:.1f}% for >= {grace:.0f}s.",
                            evidence=str(gpu),
                            stage=stage,
                            suggested_action="Verify process placement, NCCL visibility, PCIe link health, and whether the workload reached all GPUs.",
                            dedupe_key=f"{stage}|gpu{idx}|low_util",
                        )
                else:
                    self._low_util_since.pop(key, None)
                power = safe_float(gpu.get("power.draw"))
                if power_med and power and power < power_med * power_ratio:
                    self.add(
                        "HIGH",
                        "POWER",
                        "GPU power draw is low relative to peers during load",
                        f"GPU {idx} power {power:.1f} W < median {power_med:.1f} W * {power_ratio:.2f}.",
                        evidence=str(gpu),
                        stage=stage,
                        suggested_action="Check GPU clocks, power cabling, throttling, process placement, and PCIe/NCCL health.",
                        dedupe_key=f"{stage}|gpu{idx}|low_power",
                    )

    def evaluate_dmesg(self, text: str, stage: str = "dmesg") -> None:
        for item in parsers.parse_dmesg_lines(text):
            suggested = "Stop load tests and collect full dmesg/journal/NVIDIA bug report for supplier analysis."
            self.add(
                item["severity"],
                item["category"],
                item["title"],
                item["details"],
                item["evidence"],
                stage,
                suggested,
                dedupe_key=f"{stage}|dmesg|{item['severity']}|{item['evidence'][:160]}",
            )

    def evaluate_ecc_remap_text(self, text: str, stage: str = "ecc_remap") -> None:
        for line in text.splitlines():
            lower = line.lower()
            if not line.strip() or "gpu_uuid" in lower:
                continue
            if "uncorrectable" in lower or "pending" in lower or "failure" in lower:
                nums = [int(x) for x in re.findall(r"\b\d+\b", line)]
                if nums and any(n > 0 for n in nums):
                    self.add(
                        "CRITICAL",
                        "GPU",
                        "GPU ECC/remap critical counter is non-zero",
                        "Uncorrectable, pending, or remap failure counters should be zero on a new acceptance server.",
                        evidence=line.strip(),
                        stage=stage,
                        suggested_action="Do not sign off; ask supplier to diagnose/RMA the affected GPU.",
                    )
            if "retired" in lower and re.search(r"\b\d+\b", line):
                self.add(
                    "HIGH",
                    "GPU",
                    "GPU retired page/remap signal found",
                    "Retired pages or row remaps are suspicious during new-server acceptance.",
                    evidence=line.strip(),
                    stage=stage,
                    suggested_action="Ask supplier to explain the counter and consider GPU replacement.",
                )

    def evaluate_command_output(self, stage: str, command_type: str, output: str, returncode: int | None) -> None:
        if command_type == "dcgm":
            parsed = parsers.parse_dcgm_output(output)
            if returncode not in (0, None):
                self.add("CRITICAL", "DCGM", "DCGM diagnostic returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Review DCGM failure section and provide raw log to supplier.")
            if parsed["failed"]:
                self.add("CRITICAL", "DCGM", "DCGM diagnostic reports Fail/Failed", "DCGM output contains failed test lines.", "\n".join(parsed["failures"][:20]), stage, "Review DCGM result details and contact supplier if hardware tests failed.")
            if parsed["skipped"]:
                self.add("WARN", "DCGM", "DCGM diagnostic skipped unsupported tests", "Skipped/Not Supported is recorded but not treated as hardware failure.", "\n".join(parsed["skips"][:20]), stage, "Confirm whether skipped tests are expected for this driver/GPU/DCGM version.")
        elif command_type == "gpu_burn":
            parsed = parsers.parse_gpu_burn_output(output)
            if returncode not in (0, None):
                self.add("CRITICAL", "GPU", "gpu-burn returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Stop acceptance and provide raw gpu-burn log to supplier.")
            if parsed["failed"]:
                self.add("CRITICAL", "GPU", "gpu-burn output contains error/fault/failure", "gpu-burn reported suspicious calculation or hardware text.", "\n".join(parsed["bad_lines"][:20]), stage, "Treat as hardware instability until supplier proves otherwise.")
        elif command_type == "cuda_memtest":
            parsed = parsers.parse_cuda_memtest_output(output)
            if returncode not in (0, None):
                self.add("CRITICAL", "GPU", "cuda_memtest returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Inspect failing GPU and memory test log.")
            if parsed["failed"]:
                self.add("CRITICAL", "GPU", "cuda_memtest reported memory errors", "cuda_memtest output contains error/fail/mismatch.", "\n".join(parsed["bad_lines"][:20]), stage, "Ask supplier to diagnose/RMA affected GPU.")
        elif command_type == "nccl":
            parsed = parsers.parse_nccl_tests_output(output)
            if returncode not in (0, None):
                self.add("CRITICAL", "NCCL", "NCCL test returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Collect NCCL_DEBUG logs and verify driver/CUDA/NCCL fabric topology.")
            if parsed["failed"]:
                self.add("CRITICAL", "NCCL", "NCCL test output contains failure/error", "Output contains wrong/failed/timeout/internal error patterns.", "\n".join(parsed["bad_lines"][:20]), stage, "Check GPU visibility, PCIe/NVLink topology, NCCL version, and vendor cabling.")
            bus_values = [row.get("busbw") for row in parsed["rows"] if row.get("busbw")]
            outliers = parsers.bandwidth_outliers([float(v) for v in bus_values], float(self.config.get("thresholds.nccl_bandwidth_outlier_ratio_warn", 0.65)))
            if outliers:
                self.add("WARN", "NCCL", "NCCL bandwidth outlier detected", "A parsed bandwidth value is far below the median; topology-specific review is needed.", str(outliers[:10]), stage, "Do not fail solely on this; compare with expected topology and rerun with NCCL_DEBUG=INFO.")
        elif command_type == "nvbandwidth":
            parsed = parsers.parse_nvbandwidth_output(output)
            if returncode not in (0, None):
                self.add("HIGH", "NCCL", "nvbandwidth returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Verify nvbandwidth build, CUDA runtime, and GPU peer access.")
            if parsed.get("failed"):
                self.add("HIGH", "NCCL", "nvbandwidth output contains error/failure", "nvbandwidth reported a problem.", "\n".join(parsed.get("bad_lines", [])[:20]), stage, "Provide nvbandwidth log to supplier and compare topology.")
        elif command_type == "fio":
            parsed = parsers.parse_fio_json_or_text(output)
            if returncode not in (0, None):
                self.add("CRITICAL", "DISK", "fio returned non-zero", f"Return code: {returncode}", output[-2000:], stage, "Inspect fio job errors and storage health.")
            if parsed["failed"]:
                self.add("CRITICAL", "DISK", "fio output reports non-zero error", "fio JSON/text output contains job errors.", "\n".join(parsed["errors"][:20]), stage, "Check disk path, filesystem, kernel logs, and drive SMART/NVMe health.")
        elif command_type == "smart":
            parsed = parsers.parse_smartctl_output(output)
            if parsed["failed"]:
                self.add("CRITICAL", "DISK", "SMART/NVMe health output is critical", "SMART/NVMe output contains critical warnings or media/data integrity errors.", "\n".join(parsed["critical"][:30]), stage, "Ask supplier to replace or diagnose the affected drive.")
        elif command_type == "stress_ng":
            if returncode not in (0, None) or re.search(r"\b(fail|failed|verify failed)\b", output, re.IGNORECASE):
                self.add("CRITICAL", "CPU", "stress-ng reported failure", f"Return code: {returncode}", output[-2000:], stage, "Inspect CPU thermals, memory, BIOS, and kernel MCE logs.")
        elif command_type == "memtester":
            if returncode not in (0, None) or re.search(r"\b(failure|fail|failed)\b", output, re.IGNORECASE):
                self.add("CRITICAL", "MEMORY", "memtester reported failure", f"Return code: {returncode}", output[-2000:], stage, "Do not sign off; ask supplier to diagnose DIMMs/CPU memory controller.")
        elif command_type == "ipmi":
            if returncode not in (0, None):
                self.add("WARN", "BMC", "ipmitool could not query BMC", f"Return code: {returncode}", output[-1000:], stage, "If BMC is expected, check permissions/network/channel configuration.")
            if re.search(r"\b(critical|non-recoverable|fan.*fail|psu.*fail|power.*critical|thermal.*critical)\b", output, re.IGNORECASE):
                self.add("HIGH", "BMC", "BMC/IPMI reports critical power/thermal/fan event", "ipmitool output contains critical power/thermal/fan text.", output[-3000:], stage, "Review SEL/sensor details and contact supplier.")

