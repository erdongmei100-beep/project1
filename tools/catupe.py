import os
import cv2
import random
import argparse


def extract_random_frames(video_path: str,
                          output_dir: str,
                          num_frames: int = 15,
                          prefix: str = "frame"):
    """
    从视频中随机抽取指定数量的帧并保存为图片。

    :param video_path: 输入视频路径
    :param output_dir: 输出图片文件夹
    :param num_frames: 需要随机截取的帧数
    :param prefix: 输出文件名前缀
    """
    # 检查视频文件
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    # 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"无法获取视频总帧数，或视频为空: {video_path}")

    # 实际要取的帧数不能超过总帧数
    num_frames = min(num_frames, total_frames)

    # 随机选择不重复的帧下标（0 ~ total_frames-1）
    selected_indices = random.sample(range(total_frames), num_frames)
    selected_indices.sort()  # 排序一下，方便顺序读取

    print(f"视频总帧数: {total_frames}")
    print(f"随机抽取 {num_frames} 帧：{selected_indices}")

    # 遍历视频，保存选中的帧
    current_index = 0
    selected_ptr = 0

    while current_index < total_frames and selected_ptr < len(selected_indices):
        ret, frame = cap.read()
        if not ret:
            print(f"在第 {current_index} 帧读取失败，提前结束。")
            break

        # 如果当前帧是我们选中的帧，就保存
        if current_index == selected_indices[selected_ptr]:
            save_name = f"{prefix}_{current_index:06d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, frame)
            print(f"保存帧 {current_index} -> {save_path}")
            selected_ptr += 1

        current_index += 1

    cap.release()
    print("完成。")


def main():
    parser = argparse.ArgumentParser(
        description="从视频中随机截取指定数量的帧并保存为图片"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频路径，例如: /path/to/video.mp4"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出帧图片保存的文件夹路径"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=15,
        help="随机截取的帧数量，默认 15"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="输出图片文件名前缀，默认 'frame'"
    )

    args = parser.parse_args()

    extract_random_frames(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
