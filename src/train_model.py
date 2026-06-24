
# train_model.py


import joblib                                    
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import DATASET_PATH, MODEL_PATH, SCALER_PATH, ensure_dirs, section



FEATURE_COLUMNS = [
    "est_diameter_km",
    "relative_velocity_kms",
    "miss_distance_km",
    "absolute_magnitude",
    "orbit_uncertainty",
    "inclination_deg",
]
LABEL_COLUMN = "is_hazardous"


def load_dataset() -> pd.DataFrame:
    
    return pd.read_csv(DATASET_PATH)


def train():
   
    ensure_dirs()
    section("STEP 1: Loading dataset")
    df = load_dataset()
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    section("STEP 2: Splitting into train/test sets")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Hazardous in training set: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"Hazardous in test set:     {y_test.sum()} ({y_test.mean():.1%})")

    section("STEP 3: Scaling features")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    section("STEP 4: Training the RandomForestClassifier")
    model = RandomForestClassifier(
        n_estimators=200,       
        max_depth=8,              
        class_weight="balanced",  
        random_state=42,         
        n_jobs=-1,                 
    )
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    section("STEP 5: Saving model + scaler + test set")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

  
    import numpy as np
    np.save(MODEL_PATH.replace("hazard_model.joblib", "X_test.npy"), X_test_scaled)
    np.save(MODEL_PATH.replace("hazard_model.joblib", "y_test.npy"), y_test.values)

    print(f"Saved model to:  {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")


if __name__ == "__main__":
    train()
