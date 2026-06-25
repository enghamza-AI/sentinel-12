import subprocess  # for running each script as a separate process
import sys         # for reading the Python interpreter path and exiting
import os


def run_step(script_name: str, description: str):
    
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  PIPELINE STEP: {description}")
    print(f"  Script: {script_name}")
    print(bar)

    # Build the absolute path to the script so this works regardless of
    # which directory you call run_all.py from.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(src_dir, script_name)

    
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            # cwd=src_dir ensures relative imports (like `from utils import ...`)
            # work correctly inside each sub-script.
            cwd=src_dir,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[INFO] PIPELINE FAILED at step: {description}")
        print(f"   Script: {script_name}")
        print(f"   Exit code: {e.returncode}")
        print("   Fix the error above and re-run.")
        sys.exit(1)  


if __name__ == "__main__":
    print("\n☄  SENTINEL-12  --  Full Pipeline Runner")
    print("This will generate data, train the model, evaluate it, and run example predictions.\n")

    run_step("generate_data.py",   "1/4  Generate synthetic asteroid dataset")
    run_step("train_model.py",     "2/4  Train the hazard classifier")
    run_step("evaluate_model.py",  "3/4  Evaluate: confusion matrix, metrics, plots")
    run_step("predict_single.py",  "4/4  Live prediction: three example asteroids")

    print("\n" + "=" * 60)
    print("ALL PIPELINE STEPS COMPLETE")
    print("=" * 60)
    print("\nCheck the outputs/ folder for:")
    print("  confusion_matrix.png        -- the 2x2 grid of TP/TN/FP/FN")
    print("  precision_recall_curve.png  -- precision vs recall trade-off")
    print("  roc_curve.png               -- ROC curve + AUC score")
    print("\nRead ABOUT_THE_PROJECT.md for a full explanation of every concept.\n")
