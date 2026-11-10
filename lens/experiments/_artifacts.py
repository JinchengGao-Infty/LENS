"""Minimal formal-run artifact writer used by all v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


@dataclass
class ArtifactRun:
    root: Path
    config: dict[str, Any]
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def start(cls, root: str | Path, config: dict[str, Any]) -> "ArtifactRun":
        path = Path(root)
        path.mkdir(parents=True, exist_ok=False)
        run = cls(path, config)
        runtime_config = {
            **config,
            "command": [sys.executable, *sys.argv],
            "git_commit": _git_commit(),
            "started_at": run.started_at,
        }
        run._write_json("run_config.json", runtime_config)
        run._write_json(
            "manifest.json",
            {
                "schema_version": "lens-v2-artifact-1",
                "status": "RUNNING",
                "started_at": run.started_at,
                "required_files": [
                    "manifest.json",
                    "run_config.json",
                    "run_log.jsonl",
                    "candidate_rows.jsonl",
                    "summary_metrics.json",
                    "DONE",
                ],
            },
        )
        run.log("run_started")
        return run

    def _write_json(self, relative: str, payload: Any) -> None:
        destination = self.root / relative
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
        )
        temporary.replace(destination)

    def log(self, event: str, **payload: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with (self.root / "run_log.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")

    def write_candidate_rows(self, rows: list[dict[str, Any]]) -> None:
        destination = self.root / "candidate_rows.jsonl"
        temporary = self.root / ".candidate_rows.jsonl.tmp"
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        temporary.replace(destination)

    def finish(
        self,
        summary: dict[str, Any],
        *,
        extra_required_files: list[str] | tuple[str, ...] = (),
    ) -> None:
        extras = [str(name) for name in extra_required_files]
        if len(extras) != len(set(extras)):
            raise ValueError("extra_required_files contains duplicates")
        missing_extras = [name for name in extras if not (self.root / name).exists()]
        if missing_extras:
            raise ValueError(f"required artifact files do not exist: {missing_extras}")
        self._write_json("summary_metrics.json", summary)
        finished_at = datetime.now(timezone.utc).isoformat()
        self._write_json(
            "manifest.json",
            {
                "schema_version": "lens-v2-artifact-1",
                "status": "DONE",
                "started_at": self.started_at,
                "finished_at": finished_at,
                "required_files": [
                    "manifest.json",
                    "run_config.json",
                    "run_log.jsonl",
                    "candidate_rows.jsonl",
                    "summary_metrics.json",
                    *extras,
                    "DONE",
                ],
            },
        )
        (self.root / "DONE").touch()
        self.log("run_finished", finished_at=finished_at)

    def fail(self, error: BaseException) -> None:
        failed_at = datetime.now(timezone.utc).isoformat()
        self._write_json(
            "FAILED_RUN.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": failed_at,
            },
        )
        self._write_json(
            "manifest.json",
            {
                "schema_version": "lens-v2-artifact-1",
                "status": "FAILED",
                "started_at": self.started_at,
                "failed_at": failed_at,
                "failure_record": "FAILED_RUN.json",
            },
        )
        self.log("run_failed", error_type=type(error).__name__, error=str(error))
