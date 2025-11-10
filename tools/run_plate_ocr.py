# tools/run_plate_ocr.py
import argparse
from modules.lp_pipeline import process_vehicle_folder

def main():
    ap = argparse.ArgumentParser("YOLOv5 plate detect + HyperLPR recognize")
    ap.add_argument("--vehicle_dir", required=True, help="车辆裁剪图目录（现有流程的输出）")
    ap.add_argument("--out_dir", default="runs/plates", help="输出目录")
    ap.add_argument("--yolo_weights", default="weights/plate_best.pt", help="车牌检测权重")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--conf_thres", type=float, default=0.25)
    ap.add_argument("--iou_thres", type=float, default=0.45)
    ap.add_argument("--expand_ratio", type=float, default=0.10)
    ap.add_argument("--save_candidates", type=int, default=1)
    ap.add_argument("--use_hub", type=int, default=1, help="1=PyTorch Hub, 0=本地yolov5")
    ap.add_argument(
        "--download_url",
        default="",
        help="当使用本地YOLOv5且权重缺失时的自动下载链接（可留空使用默认）",
    )
    args = ap.parse_args()

    csv_path = process_vehicle_folder(
        vehicle_dir=args.vehicle_dir,
        out_dir=args.out_dir,
        yolo_weights=args.yolo_weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        expand_ratio=args.expand_ratio,
        save_candidates=bool(args.save_candidates),
        use_hub=bool(args.use_hub),
        download_url=args.download_url or None,
    )
    print(f"[OK] results.csv => {csv_path}")

if __name__ == "__main__":
    main()
