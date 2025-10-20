import json
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.roi.manager import ROIManager


class ROIManagerTests(unittest.TestCase):
    def test_scaling(self):
        roi_data = {
            "base_size": [100, 100],
            "polygon": [[10, 10], [90, 10], [90, 90], [10, 90]],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(roi_data, tmp)
            tmp_path = Path(tmp.name)
        try:
            manager = ROIManager(tmp_path)
            manager.ensure_ready((200, 200))
            scaled = manager.polygon
            self.assertAlmostEqual(scaled[0][0], 20.0)
            self.assertTrue(manager.point_in_roi((100, 100)))
            self.assertFalse(manager.point_in_roi((10, 10)))
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
