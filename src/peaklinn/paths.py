from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / 'data'
CONFIGS_DIR = PROJECT_DIR / 'src' / 'peaklinn' / 'configs'
WEIGHTS_DIR = DATA_DIR / 'model_weights'