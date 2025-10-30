from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = DATA_DIR / "outputs"

__all__ = ["ROOT", "DATA_DIR", "WEIGHTS_DIR", "CONFIGS_DIR", "OUTPUTS_DIR"]
