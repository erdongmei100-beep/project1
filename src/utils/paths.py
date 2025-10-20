from pathlib import Path
import os

# 项目根（仓库根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/utils/ -> src -> ROOT

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS_DIR = DATA_DIR / "outputs"


def project_path(*parts: str | Path) -> Path:
    """以仓库根为基准拼接路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(value: str | Path | None, default: Path) -> Path:
    """解析 CLI/配置中的路径，相对路径会转换为仓库根的绝对路径。"""
    if isinstance(value, (str, Path)) and str(value).strip():
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return default


# 车牌权重（支持三种方式：参数 > 环境变量 > 默认相对路径）
def resolve_plate_weights(cli_value: str | None = None) -> Path:
    if cli_value:
        p = Path(cli_value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    env = os.getenv("PLATE_WEIGHTS")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return WEIGHTS_DIR / "plate" / "yolov8n-plate.pt"
