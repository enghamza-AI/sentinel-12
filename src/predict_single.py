

import argparse      # stdlib: parse command-line arguments cleanly
import sys
import joblib
import numpy as np
import pandas as pd

from utils import MODEL_PATH, SCALER_PATH, OUTPUTS_DIR, section

# Feature column names in the SAME order as training. Order matters: if
# the scaler and model were trained with features [A, B, C], you must
# predict with features in the same sequence. Mixing up column order is
# a silent bug -- the model will silently give wrong answers.
FEATURE_COLUMNS = [
    "est_diameter_km",
    "relative_velocity_kms",
    "miss_distance_km",
    "absolute_magnitude",
    "orbit_uncertainty",
    "inclination_deg",
]

# The probability threshold chosen during evaluate_model.py's tuning
# (targeting 90% recall). We hardcode it here so this script is consistent
# with the evaluated system -- in a real pipeline you'd load this from a
# config file or model metadata.
TUNED_THRESHOLD = 0.35


def load_model_and_scaler():
    """Load the trained model and fitted scaler saved by train_model.py."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(asteroid_features: dict, threshold: float = TUNED_THRESHOLD) -> dict:
    """
    Run the full prediction pipeline for one asteroid.

    Parameters
    ----------
    asteroid_features : dict
        Keys must match FEATURE_COLUMNS exactly. Values are numeric.
    threshold : float
        Probability above which we call the asteroid "hazardous".
        Defaults to the value tuned in evaluate_model.py (0.35).

    Returns
    -------
    dict with keys:
        - hazard_probability : float, the raw model confidence (0.0 to 1.0)
        - prediction         : int, 0 = safe, 1 = hazardous
        - label              : str, human-readable "SAFE" or "HAZARDOUS"
        - threshold_used     : float
    """
    model, scaler = load_model_and_scaler()

    # Build a single-row DataFrame in the exact column order the model
    # expects. Using a DataFrame (not a bare numpy array) ensures columns
    # don't get silently mis-ordered.
    row = pd.DataFrame([asteroid_features], columns=FEATURE_COLUMNS)

    # Scale the features using the SAME scaler fitted on training data.
    # NEVER call .fit() here -- we're in "production" now.
    row_scaled = scaler.transform(row)

    # Get the probability of being hazardous (column index 1).
    # predict_proba returns [[P(safe), P(hazardous)]] for a single row.
    hazard_prob = model.predict_proba(row_scaled)[0, 1]

    prediction = int(hazard_prob >= threshold)
    label = "⚠  HAZARDOUS" if prediction == 1 else "SAFE"

    return {
        "hazard_probability": hazard_prob,
        "prediction": prediction,
        "label": label,
        "threshold_used": threshold,
    }


def print_result(features: dict, result: dict):
    """Print a clear, readable summary of the prediction."""
    section("SENTINEL-12 SINGLE OBJECT ASSESSMENT")

    print("Input features:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:<25} {features[col]}")

    print(f"\nHazard probability:  {result['hazard_probability']:.1%}")
    print(f"Decision threshold:  {result['threshold_used']:.0%}")
    print(f"Prediction:          {result['label']}")

    # Plain-English explanation of what this probability means
    p = result["hazard_probability"]
    if p >= 0.80:
        context = "Very high confidence hazard — would trigger immediate follow-up tracking."
    elif p >= 0.50:
        context = "Elevated hazard signal — above the default 50% threshold."
    elif p >= TUNED_THRESHOLD:
        context = (
            f"Below 50% but above our tuned {TUNED_THRESHOLD:.0%} recall-priority threshold. "
            "Flagged because we'd rather have a false alarm than miss a real threat."
        )
    else:
        context = "Probability too low to flag even at our sensitive threshold. Likely safe."
    print(f"\nContext:  {context}")


def parse_args():
    """
    Parse command-line arguments so users can run custom scenarios without
    editing the script. All arguments are optional; defaults represent a
    moderately interesting but non-threatening space rock.
    """
    parser = argparse.ArgumentParser(
        description="Predict whether a single asteroid is hazardous using the trained Sentinel-12 model."
    )
    parser.add_argument("--diameter",    type=float, default=0.3,
                        help="Estimated diameter in km (default: 0.3)")
    parser.add_argument("--velocity",    type=float, default=17.0,
                        help="Relative velocity in km/s (default: 17.0)")
    parser.add_argument("--distance",    type=float, default=2_500_000.0,
                        help="Miss distance in km (default: 2,500,000)")
    parser.add_argument("--magnitude",   type=float, default=22.0,
                        help="Absolute magnitude (default: 22.0)")
    parser.add_argument("--uncertainty", type=int,   default=3,
                        help="Orbit uncertainty 0-9 (default: 3)")
    parser.add_argument("--inclination", type=float, default=10.0,
                        help="Orbital inclination in degrees (default: 10.0)")
    parser.add_argument("--threshold",   type=float, default=TUNED_THRESHOLD,
                        help=f"Decision threshold (default: {TUNED_THRESHOLD})")
    return parser.parse_args()


# --------------------------------------------------------------------------
# HARDCODED EXAMPLE SCENARIOS
# These give you interesting outputs to compare without needing to type
# anything on the command line. Just run `python predict_single.py`.
# --------------------------------------------------------------------------

EXAMPLES = {
    "Likely safe (small, distant)": {
        "est_diameter_km": 0.05,
        "relative_velocity_kms": 12.0,
        "miss_distance_km": 7_500_000,
        "absolute_magnitude": 26.0,
        "orbit_uncertainty": 1,
        "inclination_deg": 8.0,
    },
    "Borderline (medium, moderate pass)": {
        "est_diameter_km": 0.45,
        "relative_velocity_kms": 20.0,
        "miss_distance_km": 800_000,
        "absolute_magnitude": 21.5,
        "orbit_uncertainty": 4,
        "inclination_deg": 15.0,
    },
    "Definitely scary (large, fast, close)": {
        "est_diameter_km": 2.1,
        "relative_velocity_kms": 32.0,
        "miss_distance_km": 120_000,
        "absolute_magnitude": 16.0,
        "orbit_uncertainty": 8,
        "inclination_deg": 25.0,
    },
}


if __name__ == "__main__":
    # If extra flags are passed (e.g. --diameter 1.5), use them.
    # Otherwise, run all three example scenarios for easy comparison.
    if len(sys.argv) > 1:
        args = parse_args()
        features = {
            "est_diameter_km": args.diameter,
            "relative_velocity_kms": args.velocity,
            "miss_distance_km": args.distance,
            "absolute_magnitude": args.magnitude,
            "orbit_uncertainty": args.uncertainty,
            "inclination_deg": args.inclination,
        }
        result = predict(features, threshold=args.threshold)
        print_result(features, result)
    else:
        # Run all three built-in scenarios so you immediately see a range of outputs.
        for scenario_name, features in EXAMPLES.items():
            section(f"SCENARIO: {scenario_name}")
            result = predict(features)
            print_result(features, result)
