
import os

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from utils import MODEL_PATH, SCALER_PATH, OUTPUTS_DIR, section


def load_test_artifacts():
    """Load the trained model and the held-out test set saved by train_model.py."""
    model = joblib.load(MODEL_PATH)
    X_test = np.load(MODEL_PATH.replace("hazard_model.joblib", "X_test.npy"))
    y_test = np.load(MODEL_PATH.replace("hazard_model.joblib", "y_test.npy"))
    return model, X_test, y_test


def print_confusion_matrix(y_true, y_pred, threshold_label="default (0.5)"):

    # confusion_matrix returns a 2x2 array. With labels=[0, 1], scikit-learn
    # guarantees the layout is:
    #   [[TN, FP],
    #    [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion matrix at threshold = {threshold_label}")
    print("                    Predicted SAFE   Predicted HAZARDOUS")
    print(f"  Actually SAFE         TN={tn:<5}         FP={fp:<5}   <- false alarms")
    print(f"  Actually HAZARDOUS    FN={fn:<5}         TP={tp:<5}   <- FN = MISSED THREATS (the dangerous mistake)")

    return tn, fp, fn, tp


def print_metrics(y_true, y_pred):
    """Calculate and print accuracy, precision, recall, specificity, F1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:    {accuracy:.3f}  (fraction of ALL predictions correct -- can be misleading, see below)")
    print(f"  Precision:   {precision:.3f}  (of what we flagged hazardous, fraction that really was)")
    print(f"  Recall:      {recall:.3f}  (of all REAL hazards, fraction we caught)  <-- the priority metric")
    print(f"  Specificity: {specificity:.3f}  (of all safe objects, fraction correctly left alone)")
    print(f"  F1 score:    {f1:.3f}  (balanced precision/recall summary)")

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
    }


def demonstrate_accuracy_paradox(y_true):
    """
    Show, very concretely, why accuracy alone is a trap on imbalanced data.
    We simulate the laziest possible model: one that predicts "safe" (0)
    for literally every single object, no matter what.
    """
    section("THE ACCURACY PARADOX (a 'do-nothing' model)")
    lazy_predictions = np.zeros_like(y_true)  # predicts 0 (safe) every time
    lazy_accuracy = accuracy_score(y_true, lazy_predictions)
    lazy_recall = recall_score(y_true, lazy_predictions, zero_division=0)

    print("A model that predicts 'SAFE' for every single asteroid, with zero intelligence:")
    print(f"  Accuracy: {lazy_accuracy:.1%}   <- looks great on paper!")
    print(f"  Recall:   {lazy_recall:.1%}   <- but it catches ZERO real hazards. Useless in practice.")
    print("This is why we look at the confusion matrix and recall, not just accuracy.")


def tune_threshold_for_recall(model, X_test, y_test, target_recall=0.90):
    """
    Sweep the decision threshold from high to low, recompute the confusion
    matrix at each step, and find the highest threshold that still reaches
    at least `target_recall`. Higher threshold = fewer false alarms, so we
    want the LEAST aggressive threshold that still hits our recall target.
    """
    section(f"THRESHOLD TUNING (target recall >= {target_recall:.0%})")

    
    hazard_probabilities = model.predict_proba(X_test)[:, 1]

    # Try every threshold from 0.95 down to 0.05 in steps of 0.05.
    thresholds_to_try = np.arange(0.95, 0.0, -0.05)

    print(f"{'Threshold':>10} | {'Recall':>7} | {'Precision':>9} | {'FN (missed)':>11} | {'FP (false alarms)':>18}")
    print("-" * 70)

    chosen_threshold = None  # will hold the first (highest) threshold that meets the target
    for t in thresholds_to_try:
        preds_at_t = (hazard_probabilities >= t).astype(int)
        r = recall_score(y_test, preds_at_t, zero_division=0)
        p = precision_score(y_test, preds_at_t, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_at_t, labels=[0, 1]).ravel()
        print(f"{t:>10.2f} | {r:>7.2f} | {p:>9.2f} | {fn:>11} | {fp:>18}")

        # We scan thresholds from HIGH to LOW. Recall only ever increases
        # (or stays the same) as the threshold drops, so the FIRST time we
        # cross the target is the HIGHEST threshold that achieves it --
        # i.e. the fewest false alarms for the recall we need. We lock it
        # in with `chosen_threshold is None` and deliberately do NOT keep
        # overwriting it on later (lower, noisier) thresholds.
        if chosen_threshold is None and r >= target_recall:
            chosen_threshold = t

    if chosen_threshold is None:
        # Target recall was never reached even at the lowest threshold tried.
        # Fall back to the most aggressive threshold we tried.
        chosen_threshold = thresholds_to_try[-1]

    print(f"\nChosen threshold: {chosen_threshold:.2f} "
          f"(the highest threshold that still achieves >= {target_recall:.0%} recall)")
    print("Notice the trade-off: lowering the threshold catches more real hazards (fewer FN),")
    print("but also raises false alarms (more FP). That trade is the whole point of this exercise.")

    return chosen_threshold, hazard_probabilities


def plot_confusion_matrix_heatmap(y_true, y_pred, save_path):
    """Save a simple, clearly-labeled heatmap image of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Safe", "Predicted Hazardous"])
    ax.set_yticklabels(["Actually Safe", "Actually Hazardous"])
    ax.set_title("Sentinel-12 Confusion Matrix (tuned threshold)")

    # Write the actual count inside each of the 4 cells.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix heatmap to {save_path}")


def plot_precision_recall_curve(y_true, probabilities, save_path):
    """
    Save a precision-recall curve: as the decision threshold changes, how
    do precision and recall move against each other? This is the visual
    version of the table printed in tune_threshold_for_recall().
    """
    precisions, recalls, _ = precision_recall_curve(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recalls, precisions, color="darkorange")
    ax.set_xlabel("Recall (fraction of real hazards caught)")
    ax.set_ylabel("Precision (fraction of alerts that are real)")
    ax.set_title("Precision vs Recall Trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved precision-recall curve to {save_path}")


def plot_roc_curve(y_true, probabilities, save_path):
    """
    Save an ROC curve: true positive rate (recall) vs false positive rate,
    across all thresholds. The AUC (area under curve) score summarizes
    overall ranking quality in one number, independent of any single
    threshold choice.
    """
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved ROC curve to {save_path} (AUC = {auc:.3f})")


def evaluate():
    model, X_test, y_test = load_test_artifacts()

    section("STEP 1: Default-threshold predictions (0.5 cutoff)")
    default_predictions = model.predict(X_test)
    print_confusion_matrix(y_test, default_predictions, threshold_label="default (0.5)")
    print_metrics(y_test, default_predictions)

    demonstrate_accuracy_paradox(y_test)

    chosen_threshold, hazard_probabilities = tune_threshold_for_recall(
        model, X_test, y_test, target_recall=0.90
    )

    section(f"STEP 2: Predictions at TUNED threshold ({chosen_threshold:.2f})")
    tuned_predictions = (hazard_probabilities >= chosen_threshold).astype(int)
    print_confusion_matrix(y_test, tuned_predictions, threshold_label=f"{chosen_threshold:.2f}")
    print_metrics(y_test, tuned_predictions)

    section("STEP 3: Saving plots to outputs/")
    plot_confusion_matrix_heatmap(
        y_test, tuned_predictions, os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    )
    plot_precision_recall_curve(
        y_test, hazard_probabilities, os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
    )
    plot_roc_curve(
        y_test, hazard_probabilities, os.path.join(OUTPUTS_DIR, "roc_curve.png")
    )

    section("DONE")
    print("Open the outputs/ folder to see the saved plots.")
    print("Compare the 'default (0.5)' confusion matrix above to the 'tuned' one --")
    print("the tuned version should show fewer FN (missed threats) at the cost of more FP (false alarms).")


if __name__ == "__main__":
    evaluate()
