import unittest

from src.logic.events import EventAccumulator


class EventLogicTests(unittest.TestCase):
    def test_event_segmentation(self):
        acc = EventAccumulator(min_frames_in=2, min_frames_out=1)
        track = {"track_id": 1, "bbox": [0, 0, 2, 2], "footpoint": [1, 2], "inside": True}
        acc.update(0, [track])
        acc.update(1, [track])
        leaving = {"track_id": 1, "bbox": [0, 0, 2, 2], "footpoint": [1, 2], "inside": False}
        events = acc.update(2, [leaving])
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.track_id, 1)
        self.assertEqual(event.start_frame, 0)
        self.assertEqual(event.end_frame, 1)
        self.assertEqual(event.duration_frames, 2)


if __name__ == "__main__":
    unittest.main()
