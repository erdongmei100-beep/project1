import cv2
import argparse
import random
import sys
from pathlib import Path


def extract_random_frames(video_path, output_dir, num_frames=15):
    """从视频中随机抽取指定数量的帧并保存。"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    # 打开视频读取元数据
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("错误：无法打开视频。")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("错误：无法获取视频帧数。")
        return

    print(f"视频总帧数: {total_frames}")

    # 确保不会抽取超过视频总帧数的图片
    num_to_extract = min(num_frames, total_frames)

    # 随机生成不重复的帧索引（避开前50帧和后50帧，通常开头结尾不稳定）
    safe_start = min(50, total_frames // 10)
    safe_end = max(total_frames - 50, total_frames - (total_frames // 10))

    if safe_end <= safe_start:
        # 视频太短，直接全范围随机
        indices = sorted(random.sample(range(total_frames), num_to_extract))
    else:
        indices = sorted(random.sample(range(safe_start, safe_end), num_to_extract))

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"准备抽取 {len(indices)} 张图片到: {output_dir}")

    saved_count = 0
    for idx in indices:
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            # 生成文件名：视频名_帧号.jpg
            save_name = f"{video_path.stem}_frame_{idx:06d}.jpg"
            save_path = output_dir / save_name

            cv2.imwrite(str(save_path), frame)
            print(f"[{saved_count + 1}/{num_to_extract}] 已保存: {save_name}")
            saved_count += 1
        else:
            print(f"警告：无法读取帧 {idx}")

    cap.release()
    print("\n完成！现在你可以去标注这些图片了。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机从视频中抽取帧用于制作数据集")
    parser.add_argument("--source", type=str, required=True, help="视频文件路径")
    parser.add_argument("--out", type=str, default="data/capture", help="图片保存目录 (默认: data/capture)")
    parser.add_argument("--count", type=int, default=15, help="抽取图片数量 (默认: 15)")

    args = parser.parse_args()

    extract_random_frames(args.source, args.out, args.count)