"""Utility helpers for selecting torch devices with graceful fallbacks."""

from __future__ import annotations

import importlib
import importlib.util


def select_device(
    requested: str | None = None,
    *,
    prefer_cuda: bool = True,
    fallback: str = "cpu",
    feature: str = "auto ROI",
) -> str:
    """Resolve the computation device used for inference-heavy components.

    Parameters
    ----------
    requested:
        Explicit device string provided by the caller (e.g. ``"cuda:0"`` or ``"cpu"``).
    prefer_cuda:
        When ``True`` the helper defaults to ``"cuda"`` if no explicit value is given.
    fallback:
        Device string returned when CUDA is not available or PyTorch is missing.
    feature:
        Human readable label included in warning messages.

    Returns
    -------
    str
        Device string safe to pass to downstream modules.
    """

    default = "cuda" if prefer_cuda else fallback
    candidate_raw = (requested or default).strip()
    candidate = candidate_raw.lower()

    spec = importlib.util.find_spec("torch")
    torch_module = importlib.import_module("torch") if spec is not None else None

    if candidate.startswith("cuda"):
        if torch_module is None:
            print(f"[WARN] PyTorch not available; falling back to {fallback} for {feature}.")
            return fallback
        if not torch_module.cuda.is_available():  # type: ignore[union-attr]
            print(f"[WARN] CUDA unavailable; falling back to {fallback} for {feature}.")
            return fallback
        return candidate_raw or "cuda"

    return candidate_raw or fallback


__all__ = ["select_device"]

