from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import AcceptanceConfig
from .monitor import MetricsSampler
from .report import ReportBuilder
from .risks import RiskEngine
from .suites import SUITES, SuiteRunner, collect_inventory, print_preflight
from .utils import ensure_dir, now_iso, write_json


def load_config_or_exit(path: str | None) -> AcceptanceConfig:
    try:
        return AcceptanceConfig.load(path)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        raise SystemExit(3)


def default_run_dir(config: AcceptanceConfig, suite: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return config.work_dir / f"{suite}_{stamp}"


def result_exit_code(result: str) -> int:
    return {"PASS": 0, "PASS_WITH_WARNINGS": 1, "FAIL": 2, "INCOMPLETE": 3}.get(result, 3)


def cmd_preflight(args: argparse.Namespace) -> int:
    config = load_config_or_exit(args.config)
    inventory = collect_inventory(config)
    print_preflight(inventory)
    risk_engine = RiskEngine(config)
    risk_engine.evaluate_inventory(inventory)
    if risk_engine.risks:
        print("\nBaseline risks:")
        for risk in risk_engine.risks:
            print(f"  [{risk.severity}] {risk.category}: {risk.title} - {risk.details}")
    return 0 if not risk_engine.has_high_or_critical() else 2


def cmd_list_suites(args: argparse.Namespace) -> int:
    for name, desc in SUITES.items():
        print(f"{name:10} {desc}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    config = load_config_or_exit(args.config)
    interval = args.interval or config.monitor_interval
    run_dir = Path(args.output).expanduser().resolve() if args.output else default_run_dir(config, "monitor")
    ensure_dir(run_dir)
    risk_engine = RiskEngine(config, run_dir)
    stage = "monitor"
    sampler = MetricsSampler(config, run_dir, risk_engine, suite="monitor", stage_getter=lambda: stage, interval=interval, dashboard=not args.no_dashboard)
    print(f"Monitoring started. Writing metrics to {run_dir}")
    try:
        sampler.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 130
    finally:
        sampler.stop()
        risk_engine.close()


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = ReportBuilder(run_dir).build()
    result = summary["conclusion"]["result"]
    print(f"Generated: {run_dir / 'summary.md'}")
    print(f"Generated: {run_dir / 'summary.json'}")
    print(f"Conclusion: {result}")
    return result_exit_code(result)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config_or_exit(args.config)
    if args.suite not in SUITES:
        print(f"Unknown suite: {args.suite}", file=sys.stderr)
        return 3
    run_dir = Path(args.output).expanduser().resolve() if args.output else default_run_dir(config, args.suite).resolve()
    if args.dry_run:
        print_dry_run(config, args.suite, run_dir, args.force)
        return 0
    if args.suite in {"full", "burnin"} and config.get("safety.require_confirm_for_full_or_burnin", True) and not args.yes:
        expected = "YES"
        answer = input(f"{args.suite} will run long high-load tests. Type {expected} to continue: ").strip()
        if answer != expected:
            print("Aborted by user.")
            return 3
    ensure_dir(run_dir)
    write_json(
        run_dir / "run_meta.json",
        {
            "suite": args.suite,
            "start": now_iso(),
            "config_path": str(args.config) if args.config else None,
            "config": config.as_dict(),
        },
    )
    risk_engine = RiskEngine(config, run_dir)
    runner = SuiteRunner(
        config,
        args.suite,
        run_dir,
        risk_engine,
        dry_run=False,
        continue_on_error=args.continue_on_error,
        stop_on_critical_risk=args.stop_on_critical_risk,
        force=args.force,
    )
    sampler = MetricsSampler(
        config,
        run_dir,
        risk_engine,
        suite=args.suite,
        stage_getter=lambda: runner.current_stage,
        active_command_getter=lambda: runner.active_command,
        dashboard=not args.no_dashboard,
    )
    interrupted = False
    try:
        sampler.start()
        runner.run()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Generating partial report...")
    finally:
        sampler.stop()
        runner.close()
        risk_engine.close()
    summary = ReportBuilder(run_dir).build()
    result = "INCOMPLETE" if interrupted else summary["conclusion"]["result"]
    print(f"\nRun directory: {run_dir}")
    print(f"Report: {run_dir / 'summary.md'}")
    print(f"JSON: {run_dir / 'summary.json'}")
    print(f"Conclusion: {result}")
    if interrupted:
        return 130
    return result_exit_code(result)


def print_dry_run(config: AcceptanceConfig, suite: str, run_dir: Path, force: bool) -> None:
    risk_engine = RiskEngine(config)
    runner = SuiteRunner(config, suite, run_dir, risk_engine, dry_run=True, force=force)
    try:
        print(f"Dry run suite: {suite}")
        print(f"Expected write path: {run_dir}")
        print(f"Raw logs: {run_dir / 'raw_logs'}")
        print(f"Metrics: {run_dir / 'metrics'}")
        print("\nThresholds:")
        print(json.dumps(config.get("thresholds", {}), ensure_ascii=False, indent=2))
        print("\nSafety:")
        print(json.dumps(config.get("safety", {}), ensure_ascii=False, indent=2))
        print("\nPlanned stages:")
        for row in runner.plan_rows():
            print(
                f"- {row['stage']}: enabled={row['enabled']} required={row['required']} "
                f"tool={row['tool']} timeout={row['timeout']} cmd={row['cmd']}"
            )
    finally:
        runner.close()
        risk_engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acceptance.py", description="Deep-learning server acceptance automation tool")
    parser.add_argument("--version", action="version", version="dl_server_acceptance 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Run preflight inventory and dependency checks")
    preflight.add_argument("--config", default="acceptance.yaml")
    preflight.set_defaults(func=cmd_preflight)

    run = sub.add_parser("run", help="Run an acceptance suite")
    run.add_argument("--suite", choices=sorted(SUITES), default="quick")
    run.add_argument("--config", default="acceptance.yaml")
    run.add_argument("--output", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--yes", action="store_true", help="Skip full/burnin confirmation")
    run.add_argument("--force", action="store_true", help="Allow explicitly reviewed risky operations such as dangerous fio dir")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--no-dashboard", action="store_true", help="Disable rich dashboard/plain monitor output")
    stop_group = run.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-critical-risk", dest="stop_on_critical_risk", action="store_true", default=None)
    stop_group.add_argument("--no-stop-on-critical-risk", dest="stop_on_critical_risk", action="store_false")
    run.set_defaults(func=cmd_run)

    monitor = sub.add_parser("monitor", help="Only collect live metrics")
    monitor.add_argument("--config", default="acceptance.yaml")
    monitor.add_argument("--interval", type=int, default=None)
    monitor.add_argument("--output", default=None)
    monitor.add_argument("--no-dashboard", action="store_true")
    monitor.set_defaults(func=cmd_monitor)

    report = sub.add_parser("report", help="Build report from an existing run directory")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=cmd_report)

    list_suites = sub.add_parser("list-suites", help="List available suites")
    list_suites.set_defaults(func=cmd_list_suites)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

