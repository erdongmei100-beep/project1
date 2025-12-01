from ultralytics import YOLO
# 加载您的模型
model = YOLO("weights/lane_seg.pt")  # 确保路径对

print("\n" + "="*30)
print("🔍 模型内部标签表 (model.names):")
print(model.names)
print("="*30 + "\n")