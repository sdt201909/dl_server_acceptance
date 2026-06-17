from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .config import AcceptanceConfig
from .utils import JsonlWriter, ensure_dir, now_iso, shlex_join


@dataclass
class CommandResult:
    stage: str
    cmd: str
    start: str
    end: str
    returncode: int | None
    timeout: int | None
    timed_out: bool
    stdout_path: str
    stderr_path: str


class CommandRunner:
    def __init__(
        self,
        config: AcceptanceConfig,
        run_dir: Path,
        event_writer: JsonlWriter | None = None,
        active_command_callback: Callable[[str | None], None] | None = None,
    ):
        self.config = config
        self.run_dir = run_dir
        self.raw_logs_dir = ensure_dir(run_dir / "raw_logs")
        self.commands_writer = JsonlWriter(run_dir / "commands.jsonl")
        self.event_writer = event_writer
        self.active_command_callback = active_command_callback

    def close(self) -> None:
        self.commands_writer.close()

    def _emit_event(self, event: str, stage: str, **extra) -> None:
        if self.event_writer:
            row = {"timestamp": now_iso(), "event": event, "stage": stage}
            row.update(extra)
            self.event_writer.write(row)

    def run(
        self,
        cmd: list[str] | str,
        stage: str,
        timeout: int | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
        dry_run: bool = False,
        stream_callback: Callable[[str, str], None] | None = None,
    ) -> CommandResult:
        cmd_display = shlex_join(cmd)
        stdout_path = self.raw_logs_dir / f"{stage}.stdout.log"
        stderr_path = self.raw_logs_dir / f"{stage}.stderr.log"
        start = now_iso()
        self._emit_event("command_start", stage, cmd=cmd_display, timeout=timeout)
        if self.active_command_callback:
            self.active_command_callback(cmd_display)
        if dry_run:
            result = CommandResult(stage, cmd_display, start, now_iso(), 0, timeout, False, str(stdout_path), str(stderr_path))
            self.commands_writer.write(asdict(result))
            self._emit_event("command_dry_run", stage, cmd=cmd_display)
            if self.active_command_callback:
                self.active_command_callback(None)
            return result

        ensure_dir(stdout_path.parent)
        timed_out = False
        returncode: int | None = None
        proc: subprocess.Popen[str] | None = None
        stdout_fp = stdout_path.open("w", encoding="utf-8", errors="replace")
        stderr_fp = stderr_path.open("w", encoding="utf-8", errors="replace")

        def reader(pipe, fp, stream_name: str) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    fp.write(line)
                    fp.flush()
                    if stream_callback:
                        stream_callback(stream_name, line)
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        try:
            popen_cmd = cmd if shell or isinstance(cmd, str) else [str(part) for part in cmd]
            proc = subprocess.Popen(
                popen_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=str(cwd) if cwd else None,
                env=env,
                shell=shell,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            threads = [
                threading.Thread(target=reader, args=(proc.stdout, stdout_fp, "stdout"), daemon=True),
                threading.Thread(target=reader, args=(proc.stderr, stderr_fp, "stderr"), daemon=True),
            ]
            for thread in threads:
                thread.start()
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process_group(proc, signal.SIGTERM)
                try:
                    returncode = proc.wait(timeout=self.config.graceful_shutdown_sec)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(proc, signal.SIGKILL)
                    returncode = proc.wait()
            for thread in threads:
                thread.join(timeout=3)
        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                self._terminate_process_group(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=self.config.graceful_shutdown_sec)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(proc, signal.SIGKILL)
            raise
        finally:
            stdout_fp.close()
            stderr_fp.close()
            if self.active_command_callback:
                self.active_command_callback(None)

        result = CommandResult(stage, cmd_display, start, now_iso(), returncode, timeout, timed_out, str(stdout_path), str(stderr_path))
        self.commands_writer.write(asdict(result))
        self._emit_event("command_end", stage, cmd=cmd_display, returncode=returncode, timed_out=timed_out)
        return result

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), sig)
            else:  # pragma: no cover - non-POSIX fallback
                proc.send_signal(sig)
        except ProcessLookupError:
            return
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                pass

