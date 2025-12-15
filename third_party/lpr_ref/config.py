"""配置模块：集中管理可调参数。"""

from pathlib import Path


FONT_PATH: Path = Path("simhei.ttf") 


CONF_THRESHOLD: float = 0.7

# 默认参数定义
DEFAULT_IMAGE_COLUMN = "image_name"
DEFAULT_PLATE_COLUMN = "plate_text"
DEFAULT_PLATE_IMAGE_COLUMN = "plate_image_path"
DEFAULT_DEVICE = "auto"
DEFAULT_PROGRESS_INTERVAL = 1
DEFAULT_OUTPUT_DIR = Path("outputs")