"""Small JSON artifact cache for PEPS workflow runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from peps.core.io import execution_trace_from_dict, read_json, to_plain_data, write_json
from peps.core.trace import ExecutionTrace


@dataclass(slots=True)
class WorkflowCache:
    """Filesystem cache for traces and intermediate workflow artifacts."""

    root_dir: Path | str = Path("peps/data/traces")

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def key_for_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def artifact_path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        safe_namespace = self._safe(namespace)
        safe_key = self._safe(key)
        directory = self.root_dir / safe_namespace
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe_key}{suffix}"

    def write_artifact(self, namespace: str, key: str, data: Any) -> Path:
        path = self.artifact_path(namespace, key)
        write_json(path, to_plain_data(data))
        return path

    def read_artifact(self, namespace: str, key: str) -> Any | None:
        path = self.artifact_path(namespace, key)
        if not path.exists():
            return None
        return read_json(path)

    def save_trace(self, trace: ExecutionTrace, *, name: str | None = None) -> Path:
        key = name or trace.trace_id
        return self.write_artifact("traces", key, trace)

    def load_trace(self, name: str) -> ExecutionTrace | None:
        data = self.read_artifact("traces", name)
        if data is None:
            return None
        return execution_trace_from_dict(data)

    def _safe(self, value: str) -> str:
        allowed = []
        for char in value:
            if char.isalnum() or char in {"-", "_", "."}:
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed).strip("_") or "artifact"

