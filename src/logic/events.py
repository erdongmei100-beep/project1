"""Occupancy event segmentation logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class FrameOccupancy:
    frame_idx: int
    footpoint: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: Optional[float] = None


@dataclass
class OccupancyEvent:
    track_id: int
    start_frame: int
    end_frame: int
    frames: List[FrameOccupancy] = field(default_factory=list)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class _TrackState:
    inside_streak: int = 0
    outside_streak: int = 0
    active: bool = False
    start_frame: Optional[int] = None
    last_inside_frame: Optional[int] = None
    frames: List[FrameOccupancy] = field(default_factory=list)

    def reset(self) -> None:
        self.inside_streak = 0
        self.outside_streak = 0
        self.active = False
        self.start_frame = None
        self.last_inside_frame = None
        self.frames = []


class EventAccumulator:
    """Track ROI occupancy events based on consecutive frame thresholds."""

    def __init__(self, min_frames_in: int, min_frames_out: int):
        if min_frames_in < 1 or min_frames_out < 1:
            raise ValueError("min_frames_in and min_frames_out must be >= 1")
        self.min_frames_in = min_frames_in
        self.min_frames_out = min_frames_out
        self._states: Dict[int, _TrackState] = {}
        self.completed: List[OccupancyEvent] = []

    def update(
        self, frame_idx: int, track_records: Iterable[dict], *, roi_active: bool = True
    ) -> List[OccupancyEvent]:
        present_ids = set()
        new_events: List[OccupancyEvent] = []

        neutral_frame = not roi_active

        for record in track_records:
            track_id = int(record["track_id"])
            inside = bool(record["inside"])
            footpoint = tuple(record["footpoint"])  # type: ignore[arg-type]
            bbox = tuple(record["bbox"])  # type: ignore[arg-type]
            confidence = record.get("confidence")

            state = self._states.setdefault(track_id, _TrackState())
            present_ids.add(track_id)

            if neutral_frame:
                continue

            if inside:
                state.inside_streak += 1
                state.outside_streak = 0
                frame_event = FrameOccupancy(
                    frame_idx=frame_idx,
                    footpoint=(float(footpoint[0]), float(footpoint[1])),
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    confidence=float(confidence) if confidence is not None else None,
                )
                state.frames.append(frame_event)
                state.last_inside_frame = frame_idx

                if not state.active and state.inside_streak >= self.min_frames_in:
                    start_index = max(len(state.frames) - self.min_frames_in, 0)
                    state.frames = state.frames[start_index:]
                    state.active = True
                    state.start_frame = state.frames[0].frame_idx
            else:
                state.inside_streak = 0
                if state.active:
                    state.outside_streak += 1
                    if state.outside_streak >= self.min_frames_out:
                        event = self._close_event(track_id, state)
                        if event:
                            new_events.append(event)
                else:
                    state.frames.clear()
                    state.outside_streak = min(state.outside_streak + 1, self.min_frames_out)

        # Handle tracks that were not observed in this frame.
        if not neutral_frame:
            missing_ids = set(self._states.keys()) - present_ids
            for track_id in list(missing_ids):
                state = self._states[track_id]
                if state.active:
                    state.outside_streak += 1
                    if state.outside_streak >= self.min_frames_out:
                        event = self._close_event(track_id, state)
                        if event:
                            new_events.append(event)
                else:
                    del self._states[track_id]

        self.completed.extend(new_events)
        return new_events

    def flush(self) -> List[OccupancyEvent]:
        """Force-close all active events (e.g., on stream end)."""
        remaining: List[OccupancyEvent] = []
        for track_id, state in list(self._states.items()):
            if state.active and state.frames:
                event = self._close_event(track_id, state, force=True)
                if event:
                    remaining.append(event)
            else:
                del self._states[track_id]
        self.completed.extend(remaining)
        return remaining

    def _close_event(
        self, track_id: int, state: _TrackState, force: bool = False
    ) -> Optional[OccupancyEvent]:
        if not state.frames:
            state.reset()
            self._states[track_id] = _TrackState()
            return None

        start_frame = state.start_frame if state.start_frame is not None else state.frames[0].frame_idx
        end_frame = (
            state.last_inside_frame
            if state.last_inside_frame is not None
            else state.frames[-1].frame_idx
        )
        event = OccupancyEvent(
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            frames=list(state.frames),
        )
        self._states[track_id] = _TrackState()
        return event

