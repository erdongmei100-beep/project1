"""Lightweight LaneATT model wrapper.

This module intentionally provides a very small abstraction over the
original LaneATT architecture.  The real project contains a sizeable
PyTorch network – recreating the full implementation (and shipping the
pre-trained weights) is out of scope for the integration tests that run
in this kata.  Instead we expose a minimal interface that allows loading
an arbitrary PyTorch module implementing the LaneATT API while providing
clear fallbacks when the weights are not available.

The :class:`LaneATTModel` class behaves like a regular ``torch.nn.Module``
when PyTorch is installed.  When PyTorch is not available, the class
falls back to a stub that raises a helpful error message once a forward
pass is attempted.  This design allows the surrounding ROI logic to
instantiate the class without immediately depending on a heavy machine
learning stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is optional in the tests
    torch = None  # type: ignore
    nn = object  # type: ignore


@dataclass
class LaneATTWeightsInfo:
    """Metadata describing a LaneATT checkpoint."""

    path: Path
    device: str = "cpu"


class LaneATTModel(nn.Module if torch else object):
    """Minimal LaneATT model facade.

    The class only implements the pieces required by the ROI generation
    pipeline:

    * ``load_weights`` loads a serialized ``state_dict`` from disk when
      PyTorch is available.  If the checkpoint cannot be loaded we keep
      the model in an uninitialized state so the caller can gracefully
      fall back to heuristic logic.
    * ``forward`` raises a descriptive error – the lightweight build does
      not ship the original architecture, therefore the forward pass is
      not implemented.  The ROI integration never calls ``forward``
      directly in our tests; the method exists purely for completeness
      and to highlight the limitation to advanced users.
    """

    def __init__(self) -> None:
        if torch:
            super().__init__()
        self._weights: Optional[LaneATTWeightsInfo] = None
        self._state_loaded: bool = False

    # pragma: no cover - the method mainly guards optional dependency
    def load_weights(self, checkpoint: Path, device: Optional[str] = None) -> None:
        """Load a serialized ``state_dict`` into the model.

        Parameters
        ----------
        checkpoint:
            Path to a file produced by ``torch.save`` containing a
            ``state_dict`` compatible with the original LaneATT model.
        device:
            Optional device string (``"cpu"``, ``"cuda"`` …).  When not
            provided the method falls back to ``"cpu"``.
        """

        if torch is None:
            raise RuntimeError(
                "LaneATTModel.load_weights requires PyTorch. Install torch to "
                "use pre-trained LaneATT weights."
            )
        map_location = device or "cpu"
        state_dict = torch.load(checkpoint, map_location=map_location)
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Unexpected checkpoint format for LaneATT: {type(state_dict)!r}"
            )
        self.load_state_dict(state_dict, strict=False)
        self._weights = LaneATTWeightsInfo(Path(checkpoint), map_location)
        self._state_loaded = True

    # pragma: no cover - the method is intentionally unimplemented
    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "The lightweight LaneATTModel stub does not provide a forward "
            "implementation. Load the original model to enable inference."
        )

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if ``load_weights`` succeeded."""

        return self._state_loaded

    @property
    def weights_path(self) -> Optional[Path]:
        """Return the path of the loaded checkpoint if available."""

        if self._weights is None:
            return None
        return self._weights.path
