"""Emergency lane ROI utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .auto_cv import AutoCVParams, AutoCVResult, estimate_roi, save_roi_json
from .laneatt import LaneATTParams, estimate_roi_laneatt


def generate_emergency_lane_roi(
    video_path: Path,
    mode: str,
    laneatt_params: Optional[LaneATTParams] = None,
    auto_cv_params: Optional[AutoCVParams] = None,
    overlay: bool = False,
    overlay_dir: Optional[Path] = None,
) -> AutoCVResult:
    """Generate an emergency-lane ROI using the requested backend.

    Parameters
    ----------
    video_path:
        Path to the source video that should be analysed.
    mode:
        ``"laneatt"`` selects the LaneATT integration.  ``"auto_cv"`` and
        ``"auto"`` keep the legacy heuristic implementation.
    laneatt_params / auto_cv_params:
        Optional configuration objects.  When omitted sensible defaults are
        used.
    overlay:
        Whether to persist the visual debugging overlay.
    overlay_dir:
        Optional directory used to store overlay images.
    """

    mode = (mode or "laneatt").lower()
    auto_params = auto_cv_params or AutoCVParams()
    if mode in {"laneatt", "lane_att", "lane-att"}:
        params = laneatt_params or LaneATTParams()
        result = estimate_roi_laneatt(
            video_path,
            params,
            auto_cv_params=auto_params,
            overlay=overlay or params.save_debug,
            overlay_dir=overlay_dir,
        )
        if not result.success and params.allow_auto_cv_fallback:
            return estimate_roi(
                video_path,
                auto_params,
                overlay=overlay or auto_params.save_debug,
                overlay_dir=overlay_dir,
            )
        return result
    if mode in {"auto", "auto_cv"}:
        return estimate_roi(
            video_path,
            auto_params,
            overlay=overlay or auto_params.save_debug,
            overlay_dir=overlay_dir,
        )
    raise ValueError(f"Unsupported ROI mode: {mode}")


__all__ = [
    "AutoCVParams",
    "AutoCVResult",
    "LaneATTParams",
    "estimate_roi",
    "estimate_roi_laneatt",
    "save_roi_json",
    "generate_emergency_lane_roi",
]
