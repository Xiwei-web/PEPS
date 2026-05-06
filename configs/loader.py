"""Config loader for JSON-compatible YAML PEPS config files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from peps.core.exceptions import SerializationError


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, with optional PyYAML fallback."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SerializationError(f"Failed to read config: {source}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise SerializationError(
                f"{source} is not JSON-compatible YAML and PyYAML is not installed."
            ) from exc
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise SerializationError(f"Config must be a mapping: {source}")
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge config mappings, with later configs overriding earlier ones."""
    merged: dict[str, Any] = {}
    for config in configs:
        merged = _deep_merge(merged, config)
    return merged


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(slots=True)
class PEPSConfig:
    """Loaded PEPS configuration bundle."""

    root: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    workflow: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_default(cls, path: str | Path = "peps/configs/default.yaml") -> "PEPSConfig":
        return load_default_config(path)

    @property
    def paths(self) -> dict[str, Any]:
        return self.root.get("paths", {})

    def path(self, key: str, default: str | None = None) -> str | None:
        value = self.paths.get(key, default)
        return str(value) if value is not None else None

    def default_model(self) -> str | None:
        """Return the configured default model, if present."""
        value = self.models.get("defaults", {}).get("model")
        return str(value) if value is not None else None

    def agent_config(self, agent: str) -> dict[str, Any]:
        """Return merged default + per-agent model settings."""
        defaults = self.models.get("defaults", {})
        agent_settings = self.models.get("agents", {}).get(agent, {})
        if not isinstance(defaults, dict):
            defaults = {}
        if not isinstance(agent_settings, dict):
            agent_settings = {}
        merged = merge_configs(defaults, agent_settings)
        if merged.get("model") is None:
            merged["model"] = self.default_model()
        return merged

    def agent_model(self, agent: str) -> str | None:
        """Return the per-agent model override, falling back to defaults.model."""
        value = self.agent_config(agent).get("model")
        return str(value) if value is not None else None


def load_default_config(path: str | Path = "peps/configs/default.yaml") -> PEPSConfig:
    """Load default config plus referenced model/tool/workflow/schema configs."""
    root_path = Path(path)
    root = load_config(root_path)
    defaults = root.get("defaults", {})

    def load_ref(key: str) -> dict[str, Any]:
        ref = defaults.get(key)
        if not ref:
            return {}
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            candidates = [
                Path(ref),
                root_path.parent / ref_path,
                root_path.parent.parent.parent / ref_path,
            ]
            ref_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        return load_config(ref_path)

    return PEPSConfig(
        root=root,
        models=load_ref("models_config"),
        tools=load_ref("tools_config"),
        workflow=load_ref("workflow_config"),
        schema=load_ref("schema_config"),
        source=str(root_path),
    )
