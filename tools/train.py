import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


# Windows requires strict __main__ guards to avoid multiprocessing issues
# when using libraries like PyTorch. Keeping the training logic inside this
# block ensures stable execution, especially with small datasets.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 instance segmentation model")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", type=str, default="0", help="Device ID (e.g., '0' for GPU)")
    parser.add_argument("--name", type=str, default="emergency_lane_seg", help="Training run name")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0 for Windows stability)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load the YOLO model for instance segmentation
    model = YOLO(args.model)

    # Print a concise configuration summary for clarity
    print(
        "Starting training with configuration:\n"
        f"  Model: {args.model}\n"
        f"  Data: {args.data}\n"
        f"  Epochs: {args.epochs}\n"
        f"  Batch size: {args.batch}\n"
        f"  Image size: {args.imgsz}\n"
        f"  Device: {args.device}\n"
        f"  Run name: {args.name}\n"
        f"  Workers: {args.workers}\n"
    )

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
            workers=args.workers,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print("Training failed. Please verify your data.yaml path and environment configuration.")
        print(f"Error details: {exc}")
        sys.exit(1)

    # Determine the path to the best checkpoint for quick access
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    print(f"Training completed successfully. Best model saved at: {best_model.resolve()}")


if __name__ == "__main__":
    main()
