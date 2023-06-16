from pathlib import Path

ROOT_PATH = Path().resolve()

FEATURES_PATH = ROOT_PATH / "features"
LOGGING_PATH = ROOT_PATH / "logs"
GRIDS_PATH = ROOT_PATH / "grids"
RF_GRID_PATH = GRIDS_PATH / "rf.json"

FACENET_MODEL_PATH = ROOT_PATH / "models" / "inception-resnet-v1-facenet.h5"

ROOT_DATASET_FOLDER = Path("/home/yves/Área de Trabalho")