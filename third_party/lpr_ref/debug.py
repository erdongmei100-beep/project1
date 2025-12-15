from pathlib import Path
import cv2
import os

# 复用您现有 utils 中的函数，确保逻辑一致
from utils import run_hyperlpr

# 设定放置测试图片的文件夹
TEST_DIR = Path("debug_images")

def run_debug():
    # 1. 检查文件夹
    if not TEST_DIR.exists():
        print(f"❌ 文件夹不存在！请在项目根目录新建一个名为 '{TEST_DIR.name}' 的文件夹。")
        return

    image_files = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.bmp'}]
    
    if not image_files:
        print(f"⚠️ '{TEST_DIR.name}' 文件夹是空的！请放进去几张手动裁剪的车牌大头照。")
        return

    print(f"🔍 开始测试 {len(image_files)} 张图片...\n")

    # 2. 循环识别
    for img_file in image_files:
        # 读取
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"❌ 无法读取: {img_file.name}")
            continue

        # 识别
        text, conf, bbox = run_hyperlpr(img)

        # 打印结果
        if text:
            print(f"✅ {img_file.name}")
            print(f"   └── 结果: [{text}]  置信度: {conf:.4f}")
        else:
            print(f"❌ {img_file.name}")
            print(f"   └── 未识别到车牌")
            
    print("\n测试结束。")

if __name__ == "__main__":
    run_debug()