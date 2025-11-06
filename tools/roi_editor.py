"""Simple polygon ROI editor using OpenCV."""

import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np

pts: List[Tuple[int, int]] = []
done = False


def mouse(event, x, y, flags, param):
    del flags, param
    global pts, done
    if event == cv2.EVENT_LBUTTONDOWN and not done:
        pts.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(pts) >= 3:
        done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--save-dir", default="data/rois")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("无法读取视频首帧")

    win = "ROI Editor (左键添加点，右键结束，c撤销，s保存)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    global pts, done
    pts = []
    done = False

    while True:
        canvas = frame.copy()
        if pts:
            pts_np = np.array(pts, np.int32)
            cv2.polylines(canvas, [pts_np], False, (0, 255, 255), 2)
            for p in pts:
                cv2.circle(canvas, p, 3, (0, 255, 0), -1)
        help_lines = [
            "左键：添加点",
            "右键：完成多边形",
            "C：撤销最近的点",
            "S：保存并退出",
            "Esc：放弃退出",
        ]
        for idx, text in enumerate(help_lines):
            cv2.putText(
                canvas,
                text,
                (10, 24 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            break
        if key == ord("c") and pts:
            pts.pop()
        if key == ord("s") or done:
            if len(pts) >= 3:
                stem = os.path.splitext(os.path.basename(args.source))[0]
                path = os.path.join(args.save_dir, f"{stem}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"polygon": pts}, f, ensure_ascii=False, indent=2)
                print("ROI saved to", path)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
