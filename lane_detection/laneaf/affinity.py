"""Affinity field decoding utilities copied from the LaneAF project."""
from __future__ import annotations

import numpy as np


def decode_afs(
    mask: np.ndarray,
    vaf: np.ndarray,
    haf: np.ndarray,
    fg_thresh: float = 128.0,
    err_thresh: float = 5.0,
) -> np.ndarray:
    """Decode instance ids from segmentation logits.

    Parameters
    ----------
    mask: np.ndarray
        Foreground probability map scaled to 0-255.
    vaf: np.ndarray
        Vertical affinity field (H, W, 2).
    haf: np.ndarray
        Horizontal affinity field (H, W, 1).
    fg_thresh: float
        Threshold applied to ``mask`` to determine foreground points.
    err_thresh: float
        Maximum error when matching clusters to existing lane instances.
    """

    output = np.zeros_like(mask, dtype=np.uint8)
    lane_end_pts: list[np.ndarray] = []
    next_lane_id = 1

    height, width = mask.shape

    for row in range(height - 1, -1, -1):
        cols = np.where(mask[row, :] > fg_thresh)[0]
        clusters: list[list[int]] = [[]]
        if cols.size > 0:
            prev_col = int(cols[0])
        else:
            prev_col = 0
        for col in cols:
            if col - prev_col > err_thresh:
                clusters.append([])
                clusters[-1].append(int(col))
                prev_col = int(col)
                continue
            if haf[row, prev_col] >= 0 and haf[row, col] >= 0:
                clusters[-1].append(int(col))
                prev_col = int(col)
                continue
            if haf[row, prev_col] >= 0 and haf[row, col] < 0:
                clusters[-1].append(int(col))
                prev_col = int(col)
            elif haf[row, prev_col] < 0 and haf[row, col] >= 0:
                clusters.append([])
                clusters[-1].append(int(col))
                prev_col = int(col)
                continue
            elif haf[row, prev_col] < 0 and haf[row, col] < 0:
                clusters[-1].append(int(col))
                prev_col = int(col)
                continue

        assigned = [False for _ in clusters]
        if lane_end_pts:
            cost = np.full((len(lane_end_pts), len(clusters)), np.inf, dtype=np.float64)
        else:
            cost = np.zeros((0, len(clusters)), dtype=np.float64)
        for r, pts in enumerate(lane_end_pts):
            for c, cluster in enumerate(clusters):
                if not cluster:
                    continue
                cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                vafs = np.array([
                    vaf[int(round(x[1])), int(round(x[0])), :]
                    for x in pts
                ], dtype=np.float32)
                norms = np.linalg.norm(vafs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vafs = vafs / norms
                pred_points = pts + vafs * np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                cost[r, c] = error
        if cost.size:
            order = np.argsort(cost, axis=None)
            row_ind, col_ind = np.unravel_index(order, cost.shape)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= err_thresh:
                    break
                if assigned[c]:
                    continue
                assigned[c] = True
                output[row, clusters[c]] = r + 1
                lane_end_pts[r] = np.stack(
                    (
                        np.array(clusters[c], dtype=np.float32),
                        row * np.ones_like(clusters[c], dtype=np.float32),
                    ),
                    axis=1,
                )
        for c, cluster in enumerate(clusters):
            if not cluster:
                continue
            if not assigned[c]:
                output[row, cluster] = next_lane_id
                lane_end_pts.append(
                    np.stack(
                        (
                            np.array(cluster, dtype=np.float32),
                            row * np.ones_like(cluster, dtype=np.float32),
                        ),
                        axis=1,
                    )
                )
                next_lane_id += 1

    return output


__all__ = ["decode_afs"]
