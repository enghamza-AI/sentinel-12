"""
generate_data.py
=================

Real asteroid tracking data (like NASA's Near-Earth Object database) is messy,
requires an internet connection, and changes over time -- not great for a
learning project where you want the same result every time you run it.

So instead, this file BUILDS A FAKE BUT REALISTIC dataset of near-Earth
objects (NEOs) from scratch

THE FEATURES (COLUMNS) WE SIMULATE
------------------------------------
- est_diameter_km      : estimated diameter of the object, in kilometers.
                          Bigger objects do more damage if they hit.
- relative_velocity_kms: how fast the object is moving relative to Earth,
                          in km/s. Faster impacts release more energy.
- miss_distance_km      : how far the object will pass from Earth at its
                          closest approach, in kilometers. Smaller = scarier.
- absolute_magnitude    : a brightness measure astronomers use. CONFUSINGLY,
                          a LOWER number means a BIGGER/brighter object.
- orbit_uncertainty     : a 0-9 score for how well we know the object's
                          orbit. Higher = we're less sure where it'll go.
- inclination_deg       : tilt of the object's orbit relative to Earth's
                          orbital plane, in degrees. Just adds realistic
                          noise -- on its own it doesn't drive hazard risk.

"""

import numpy as np      
import pandas as pd     
import os                


def generate_asteroid_dataset(n_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:

    
    rng = np.random.default_rng(random_seed)

    est_diameter_km = rng.lognormal(mean=-2.0, sigma=1.0, size=n_samples)
    est_diameter_km = np.clip(est_diameter_km, 0.001, 15.0)  
    relative_velocity_kms = rng.normal(loc=17.0, scale=7.0, size=n_samples)
    relative_velocity_kms = np.clip(relative_velocity_kms, 1.0, 45.0)
    miss_distance_km = rng.lognormal(mean=16.5, sigma=1.3, size=n_samples)
    miss_distance_km = np.clip(miss_distance_km, 50_000, 9_000_000)

    absolute_magnitude = 23 - 5 * np.log10(est_diameter_km + 0.01) + rng.normal(0, 0.5, n_samples)

    orbit_uncertainty = rng.integers(0, 10, size=n_samples)

    inclination_deg = rng.uniform(0, 35, size=n_samples)


    def normalize(x):
        
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    size_score = normalize(est_diameter_km)
    speed_score = normalize(relative_velocity_kms)

    closeness_score = 1 - normalize(miss_distance_km)
    uncertainty_score = normalize(orbit_uncertainty)

    risk_score = (
        0.40 * size_score +
        0.20 * speed_score +
        0.30 * closeness_score +
        0.10 * uncertainty_score
    )

    
    
    risk_score_noisy = risk_score + rng.normal(0, 0.07, n_samples)

    cutoff = np.percentile(risk_score_noisy, 94)
    is_hazardous = (risk_score_noisy >= cutoff).astype(int)

    df = pd.DataFrame({
        "est_diameter_km": est_diameter_km,
        "relative_velocity_kms": relative_velocity_kms,
        "miss_distance_km": miss_distance_km,
        "absolute_magnitude": absolute_magnitude,
        "orbit_uncertainty": orbit_uncertainty,
        "inclination_deg": inclination_deg,
        "is_hazardous": is_hazardous,
    })

    return df


if __name__ == "__main__":
    

    dataset = generate_asteroid_dataset()

    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "asteroids.csv")
    dataset.to_csv(output_path, index=False)

    hazardous_count = dataset["is_hazardous"].sum()
    total_count = len(dataset)
    print(f"Saved {total_count} simulated asteroids to {output_path}")
    print(f"Hazardous: {hazardous_count} ({hazardous_count / total_count:.1%})")
    print(f"Safe:      {total_count - hazardous_count} ({1 - hazardous_count / total_count:.1%})")
