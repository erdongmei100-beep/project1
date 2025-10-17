"""Configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def load_config(path: Path | str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    try:
        from ultralytics.utils import yaml_load  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to parse configuration files but is not installed."
        ) from exc
    return yaml_load(config_path)


def save_config(path: Path | str, data: Dict[str, Any]) -> None:
    """Persist configuration data back to disk."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to save configuration files.")
    config_path = Path(path)
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    config_path.write_text(text, encoding="utf-8")


def resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / value).resolve()
    return candidate

