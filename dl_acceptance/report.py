from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .utils import ensure_dir, read_jsonl, safe_float, severity_rank, write_json


class ReportBuilder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def build(self) -> dict[str, Any]:
        inventory = self._read_json(self.run_dir / "inventory.json", {})
        stages = read_jsonl(self.run_dir / "stages.jsonl")
        commands = read_jsonl(self.run_dir / "commands.jsonl")
        risks = read_jsonl(self.run_dir / "risks.jsonl")
        events = read_jsonl(self.run_dir / "events.jsonl")
        gpu_summary = self._gpu_summary()
        conclusion = self._conclusion(stages, risks, events)
        summary = {
            "run_dir": str(self.run_dir),
            "conclusion": conclusion,
            "inventory": inventory,
            "stages": stages,
            "commands": commands,
            "risks": risks,
            "risk_counts": self._risk_counts(risks),
            "gpu_summary": gpu_summary,
            "paths": {
                "summary_md": str(self.run_dir / "summary.md"),
                "summary_json": str(self.run_dir / "summary.json"),
                "raw_logs": str(self.run_dir / "raw_logs"),
                "metrics": str(self.run_dir / "metrics"),
                "commands_jsonl": str(self.run_dir / "commands.jsonl"),
                "risks_jsonl": str(self.run_dir / "risks.jsonl"),
            },
        }
        write_json(self.run_dir / "summary.json", summary)
        (self.run_dir / "summary.md").write_text(self._markdown(summary), encoding="utf-8")
        return summary

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _risk_counts(risks: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"INFO": 0, "WARN": 0, "HIGH": 0, "CRITICAL": 0}
        for risk in risks:
            sev = str(risk.get("severity", "INFO")).upper()
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _conclusion(self, stages: list[dict[str, Any]], risks: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
        highest = "INFO"
        for risk in risks:
            sev = str(risk.get("severity", "INFO")).upper()
            if severity_rank(sev) > severity_rank(highest):
                highest = sev
        interrupted = any(event.get("event") == "user_interrupt" for event in events)
        required_skipped = [s for s in stages if s.get("required") and s.get("status") == "SKIPPED"]
        failed_required_unknown = [s for s in stages if s.get("required") and s.get("status") == "FAIL" and highest in {"INFO", "WARN"}]
        if interrupted:
            result = "INCOMPLETE"
        elif severity_rank(highest) >= severity_rank("HIGH"):
            result = "FAIL"
        elif required_skipped or failed_required_unknown:
            result = "INCOMPLETE"
        elif highest == "WARN":
            result = "PASS_WITH_WARNINGS"
        else:
            result = "PASS"
        return {
            "result": result,
            "highest_risk": highest,
            "recommend_signoff": result == "PASS",
            "recommend_supplier_or_rma": severity_rank(highest) >= severity_rank("HIGH"),
            "required_skipped": [s.get("stage") for s in required_skipped],
            "interrupted": interrupted,
        }

    def _gpu_summary(self) -> list[dict[str, Any]]:
        path = self.run_dir / "metrics" / "gpu_metrics.csv"
        if not path.exists():
            return []
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        names: dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                idx = str(row.get("index", "unknown"))
                names[idx] = row.get("name", "") or names.get(idx, "")
                for key in ["temperature.gpu", "power.draw", "memory.used", "utilization.gpu"]:
                    value = safe_float(row.get(key))
                    if value is not None:
                        buckets[idx][key].append(value)
        summary = []
        for idx, values in sorted(buckets.items(), key=lambda kv: kv[0]):
            summary.append(
                {
                    "index": idx,
                    "name": names.get(idx, ""),
                    "max_temp_c": max(values["temperature.gpu"]) if values["temperature.gpu"] else None,
                    "avg_temp_c": mean(values["temperature.gpu"]) if values["temperature.gpu"] else None,
                    "max_power_w": max(values["power.draw"]) if values["power.draw"] else None,
                    "avg_power_w": mean(values["power.draw"]) if values["power.draw"] else None,
                    "max_memory_used_mib": max(values["memory.used"]) if values["memory.used"] else None,
                    "avg_util_pct": mean(values["utilization.gpu"]) if values["utilization.gpu"] else None,
                }
            )
        return summary

    def _markdown(self, summary: dict[str, Any]) -> str:
        inv = summary.get("inventory", {})
        conclusion = summary.get("conclusion", {})
        risks = summary.get("risks", [])
        stages = summary.get("stages", [])
        commands = {row.get("stage"): row for row in summary.get("commands", [])}
        counts = summary.get("risk_counts", {})
        lines: list[str] = []
        lines.append("# 深度学习服务器验收报告")
        lines.append("")
        lines.append("## 1. 总体结论")
        lines.append(f"- 结论：**{conclusion.get('result')}**")
        lines.append(f"- 最高风险等级：**{conclusion.get('highest_risk')}**")
        lines.append(f"- 风险计数：INFO={counts.get('INFO', 0)} WARN={counts.get('WARN', 0)} HIGH={counts.get('HIGH', 0)} CRITICAL={counts.get('CRITICAL', 0)}")
        lines.append(f"- 是否建议签收：{'是' if conclusion.get('recommend_signoff') else '否'}")
        lines.append(f"- 是否建议联系供应商/RMA：{'是' if conclusion.get('recommend_supplier_or_rma') else '否'}")
        if conclusion.get("required_skipped"):
            lines.append(f"- 未完成必要测试：{', '.join(conclusion.get('required_skipped'))}")
        lines.append("")
        lines.append("## 2. 服务器配置")
        lines.append(f"- hostname：{inv.get('hostname', 'unknown')}")
        lines.append(f"- OS：{self._first_os_line(inv.get('os_release', ''))}")
        lines.append(f"- kernel：{inv.get('kernel', 'unknown')}")
        cpu = inv.get("cpu", {})
        lines.append(f"- CPU：{cpu.get('model', 'unknown')}，逻辑核心 {cpu.get('logical_cores', 'unknown')}")
        mem = inv.get("memory", {}).get("total_gb")
        lines.append(f"- memory：{mem:.1f} GB" if isinstance(mem, (int, float)) else "- memory：unknown")
        fio = inv.get("fio", {})
        lines.append(f"- disk/fio：{fio.get('test_dir')}，free={self._fmt(fio.get('free_gb'))} GB，dangerous={fio.get('dangerous_path')}")
        lines.append(f"- driver/CUDA/NVML：{inv.get('driver', {})}")
        lines.append("")
        lines.append("### GPU Inventory")
        lines.append("| index | name | memory | uuid | bus id |")
        lines.append("|---:|---|---:|---|---|")
        for gpu in inv.get("gpus", []):
            mem_gb = (gpu.get("memory.total") or 0) / 1024 if gpu.get("memory.total") else None
            lines.append(f"| {gpu.get('index')} | {gpu.get('name')} | {self._fmt(mem_gb)} GB | {gpu.get('uuid')} | {gpu.get('pci.bus_id')} |")
        if not inv.get("gpus"):
            lines.append("| - | No GPU detected | - | - | - |")
        topo = inv.get("topology", {}).get("nvidia_smi_topo_m", "")
        lines.append("")
        lines.append("### PCIe / NVIDIA Topology")
        lines.append("```text")
        lines.append(topo.rstrip() if topo else "nvidia-smi topo -m output not available")
        lines.append("```")
        lines.append("")
        lines.append("## 3. 测试套件与参数")
        lines.append("| stage | status | required | command | start | end | rc | reason |")
        lines.append("|---|---|---:|---|---|---|---:|---|")
        for stage in stages:
            cmd = stage.get("cmd") or commands.get(stage.get("stage"), {}).get("cmd", "")
            lines.append(
                f"| {stage.get('stage')} | {stage.get('status')} | {stage.get('required')} | `{self._escape_table(cmd)}` | "
                f"{stage.get('start')} | {stage.get('end')} | {stage.get('returncode')} | {self._escape_table(stage.get('reason', ''))} |"
            )
        lines.append("")
        lines.append("## 4. 风险项")
        for severity in ["CRITICAL", "HIGH", "WARN", "INFO"]:
            lines.append(f"### {severity}")
            subset = [risk for risk in risks if str(risk.get("severity", "")).upper() == severity]
            if not subset:
                lines.append("- 无")
            for risk in subset:
                lines.append(f"- **[{risk.get('category')}] {risk.get('title')}**（stage: {risk.get('stage')}）")
                lines.append(f"  - details: {risk.get('details')}")
                evidence = str(risk.get("evidence", "")).strip()
                if evidence:
                    lines.append(f"  - evidence: `{self._escape_inline(evidence[:500])}`")
                action = str(risk.get("suggested_action", "")).strip()
                if action:
                    lines.append(f"  - suggested_action: {action}")
        lines.append("")
        lines.append("## 5. GPU 压测摘要")
        lines.append("| GPU | name | max temp C | avg temp C | max power W | avg power W | max memory MiB | avg util % | Xid/ECC/掉卡 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        gpu_risk_text = "；".join(r.get("title", "") for r in risks if r.get("category") in {"GPU", "THERMAL", "POWER"} and severity_rank(r.get("severity", "INFO")) >= 2)
        for gpu in summary.get("gpu_summary", []):
            lines.append(
                f"| {gpu.get('index')} | {gpu.get('name')} | {self._fmt(gpu.get('max_temp_c'))} | {self._fmt(gpu.get('avg_temp_c'))} | "
                f"{self._fmt(gpu.get('max_power_w'))} | {self._fmt(gpu.get('avg_power_w'))} | {self._fmt(gpu.get('max_memory_used_mib'))} | "
                f"{self._fmt(gpu.get('avg_util_pct'))} | {'有风险' if gpu_risk_text else '未见记录'} |"
            )
        if not summary.get("gpu_summary"):
            lines.append("| - | metrics not available | - | - | - | - | - | - | - |")
        lines.append("")
        lines.append("## 6. NCCL/nvbandwidth 摘要")
        for stage in stages:
            if stage.get("command_type") in {"nccl", "nvbandwidth", "torch_ddp"}:
                lines.append(f"- {stage.get('stage')}：{stage.get('status')}，rc={stage.get('returncode')}，reason={stage.get('reason')}")
        lines.append("- 若报告中出现 bandwidth outlier，仅表示性能异常需要结合实际 PCIe/NVLink 拓扑人工核对，不单独作为硬件失败结论。")
        lines.append("")
        lines.append("## 7. CPU/内存/存储摘要")
        for stage in stages:
            if stage.get("command_type") in {"stress_ng", "memtester", "fio", "smart"}:
                lines.append(f"- {stage.get('stage')}：{stage.get('status')}，rc={stage.get('returncode')}，reason={stage.get('reason')}")
        lines.append("")
        lines.append("## 8. 附录")
        paths = summary.get("paths", {})
        lines.append(f"- raw_logs：`{paths.get('raw_logs')}`")
        lines.append(f"- metrics CSV：`{paths.get('metrics')}`")
        lines.append(f"- commands.jsonl：`{paths.get('commands_jsonl')}`")
        lines.append(f"- risks.jsonl：`{paths.get('risks_jsonl')}`")
        lines.append(f"- inventory.json：`{self.run_dir / 'inventory.json'}`")
        lines.append(f"- environment.txt：`{self.run_dir / 'environment.txt'}`")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _first_os_line(os_release: str) -> str:
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
        return os_release.splitlines()[0] if os_release.splitlines() else "unknown"

    @staticmethod
    def _fmt(value: Any) -> str:
        number = safe_float(value)
        return "?" if number is None else f"{number:.1f}"

    @staticmethod
    def _escape_table(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")[:300]

    @staticmethod
    def _escape_inline(value: str) -> str:
        return value.replace("`", "'").replace("\n", " ")

