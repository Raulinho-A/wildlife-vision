from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJ_ROOT / "data"

ANNOTATIONS = DATA_DIR / "annotations"
DATALOADER = DATA_DIR / "dataloader"
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"

# Models
MODELS_DIR = PROJ_ROOT / "models"



