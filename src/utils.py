import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATASET_PATH = os.path.join(DATA_DIR, "asteroids.csv")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "hazard_model.joblib")
SCALER_PATH = os.path.join(OUTPUTS_DIR, "feature_scaler.joblib")


def ensure_dirs():
  
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def section(title: str):
   
    bar = "=" * 60
    print(f"\n{bar}\n {title}\n{bar}")
