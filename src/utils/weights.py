"""权重文件获取工具。"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable

from .paths import resolve_plate_weights

PLATE_MIN_BYTES = 1_000_000
PLATE_URLS: tuple[str, ...] = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "https://github.com/keremberke/yolov8-license-plate/releases/download/v0.0.0/yolov8n-license-plate.pt",
)


def _is_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= PLATE_MIN_BYTES


def _download(urls: Iterable[str], destination: Path) -> None:
    for url in urls:
        try:
            print(f"[weights] downloading plate weights from {url} ...")
            urllib.request.urlretrieve(url, destination)
        except Exception as exc:  # pragma: no cover - network failure
            print(f"[weights] download failed from {url}: {exc}")
            continue
        if _is_ready(destination):
            print(f"[weights] plate weights ready: {destination}")
            return
    raise FileNotFoundError(
        "Plate detector weights not found and download failed. "
        f"Expected at: {destination}"
    )


def ensure_plate_weights(
    cli_value: str | None = None,
    env_value: str | None = None,
    configured_value: str | None = None,
) -> Path:
    """确保车牌检测权重就绪，必要时自动下载到默认路径。"""

    if cli_value:
        candidate = resolve_plate_weights(cli_value)
        if _is_ready(candidate):
            return candidate
        message = (
            f"Plate detector weights not found at CLI override: {candidate}\n"
            "Please provide a valid path via --weights-plate."
        )
        raise FileNotFoundError(message)

    if env_value:
        candidate = resolve_plate_weights(env_value)
        if _is_ready(candidate):
            return candidate
        message = (
            f"Plate detector weights not found at PLATE_WEIGHTS path: {candidate}\n"
            "Update the environment variable or remove it to allow auto-download."
        )
        raise FileNotFoundError(message)

    default_path = resolve_plate_weights(None)
    if configured_value:
        candidate = resolve_plate_weights(configured_value)
        if _is_ready(candidate):
            return candidate
        if candidate.resolve() != default_path.resolve():
            message = (
                f"Configured plate weights not found: {candidate}\n"
                "Update configs/default.yaml or place the file manually."
            )
            raise FileNotFoundError(message)

    if _is_ready(default_path):
        return default_path

    default_path.parent.mkdir(parents=True, exist_ok=True)
    _download(PLATE_URLS, default_path)
    return default_path
