import unittest
from pathlib import Path

import numpy as np

from src.lane_detection.laneatt import LaneATTConfig, LaneATTDetector
from src.lane_detection.laneatt import detector as laneatt_detector_module
from src.roi.laneatt import LaneATTParams, estimate_roi_laneatt


class LaneATTLaneDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        if laneatt_detector_module.cv2 is None:
            self.skipTest("OpenCV is required for LaneATT detector tests")
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = self.frame.shape[:2]
        xs = np.linspace(320, 520, num=240).astype(int)
        ys = np.linspace(470, 120, num=240).astype(int)
        for x, y in zip(xs, ys):
            x0 = max(0, x - 2)
            x1 = min(w, x + 3)
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            self.frame[y0:y1, x0:x1] = 255

    def test_detector_returns_lane(self) -> None:
        detector = LaneATTDetector(LaneATTConfig())
        lanes = detector.detect_lanes(self.frame)
        self.assertTrue(lanes, "Expected at least one lane detection")
        for lane in lanes:
            self.assertEqual(lane.shape[1], 2)
            self.assertTrue((lane[:, 0] >= 0).all())
            self.assertTrue((lane[:, 1] >= 0).all())
            self.assertLessEqual(lane[:, 0].max(), self.frame.shape[1] - 1)
            self.assertLessEqual(lane[:, 1].max(), self.frame.shape[0] - 1)

    def test_estimate_roi_laneatt(self) -> None:
        params = LaneATTParams(
            sample_frames=1,
            min_lane_frames=1,
            save_debug=False,
            bottom_ratio=0.4,
        )
        result = estimate_roi_laneatt(Path("synthetic.mp4"), params, frames=[self.frame])
        self.assertTrue(result.success)
        self.assertIsNotNone(result.polygon)
        polygon = result.polygon
        assert polygon is not None
        self.assertGreaterEqual(len(polygon), 4)
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        self.assertTrue(all(0 <= x < self.frame.shape[1] for x in xs))
        self.assertTrue(all(0 <= y <= self.frame.shape[0] for y in ys))
        self.assertEqual(result.metrics.get("engine"), "laneatt")
        self.assertIsNotNone(result.overlay)


if __name__ == "__main__":
    unittest.main()
