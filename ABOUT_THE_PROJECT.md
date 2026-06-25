# 🪐 ABOUT THE PROJECT — Sentinel-12

> **This file is a study companion.**
> Read it alongside the code. Every ML concept used in this project is
> explained here from scratch, with the asteroid context woven in so you
> always know *why* it matters, not just *what* it is.

---

## Table of Contents

1. [What is this project really teaching?](#1-what-is-this-project-really-teaching)
2. [The Problem Space: Why asteroid detection?](#2-the-problem-space-why-asteroid-detection)
3. [The Dataset: What are we working with?](#3-the-dataset-what-are-we-working-with)
4. [Core ML Concept: Binary Classification](#4-core-ml-concept-binary-classification)
5. [Core ML Concept: Class Imbalance](#5-core-ml-concept-class-imbalance)
6. [Core ML Concept: The Confusion Matrix (THE BIG ONE)](#6-core-ml-concept-the-confusion-matrix-the-big-one)
7. [Core ML Concept: Metrics Derived from the Confusion Matrix](#7-core-ml-concept-metrics-derived-from-the-confusion-matrix)
8. [Core ML Concept: The Accuracy Paradox](#8-core-ml-concept-the-accuracy-paradox)
9. [Core ML Concept: Decision Threshold Tuning](#9-core-ml-concept-decision-threshold-tuning)
10. [Core ML Concept: The Precision-Recall Trade-off](#10-core-ml-concept-the-precision-recall-trade-off)
11. [Core ML Concept: ROC Curve and AUC](#11-core-ml-concept-roc-curve-and-auc)
12. [Core ML Concept: Feature Engineering](#12-core-ml-concept-feature-engineering)
13. [Core ML Concept: Train / Test Split and Data Leakage](#13-core-ml-concept-train--test-split-and-data-leakage)
14. [Core ML Concept: Feature Scaling](#14-core-ml-concept-feature-scaling)
15. [Core ML Concept: The Random Forest Model](#15-core-ml-concept-the-random-forest-model)
16. [Core ML Concept: class_weight and Handling Imbalance in Training](#16-core-ml-concept-class_weight-and-handling-imbalance-in-training)
17. [Core ML Concept: Probability Output vs Hard Prediction](#17-core-ml-concept-probability-output-vs-hard-prediction)
18. [The Full Learning Journey: Concept Map](#18-the-full-learning-journey-concept-map)
19. [What to Experiment With Next](#19-what-to-experiment-with-next)
20. [Glossary: Every Term in One Place](#20-glossary-every-term-in-one-place)

---

## 1. What is this project really teaching?

Most beginner ML tutorials end the lesson the moment a model finishes training and prints one number: **accuracy**. They show you "93% accurate!" and move on. This project argues that is **the wrong place to stop** — especially for any real-world problem where mistakes have unequal costs.

Sentinel-12 deliberately creates a scenario where 93% accuracy is **actively misleading**. The goal is to make you feel that sting concretely, then give you the full set of tools — the confusion matrix and the metrics that come from it — to see through it.

**Everything in this project is in service of one core skill: learning to evaluate a model properly.**

---

## 2. The Problem Space: Why asteroid detection?

We chose a planetary defense setting because it has a very clean, very teachable **asymmetry of errors**:

| Error type | What happened | Real-world cost |
|---|---|---|
| **False Positive** | We flagged a safe asteroid as dangerous | Extra tracking effort, a few phone calls. Annoying. |
| **False Negative** | We missed a real hazard | Potentially catastrophic. Unacceptable. |

This asymmetry is what drives the whole project. A system optimized to minimize *total errors* equally would behave very differently from one that says "I would rather have 10 false alarms than miss one real threat." The second framing is correct here. The confusion matrix is the tool that lets you **see which type of error your model is actually making**.

---

## 3. The Dataset: What are we working with?

We generate a **synthetic** dataset modeled on NASA's Near-Earth Object (NEO) classification system. The features are realistic (they match the kinds of measurements real surveys like Pan-STARRS or Catalina Sky Survey collect), but the data is created by a script so the project runs with no internet connection and gives identical results every time.

### Features (the inputs to the model)

| Feature | Unit | What it represents |
|---|---|---|
| `est_diameter_km` | km | How big the object is. Bigger = more damage if it hits. |
| `relative_velocity_kms` | km/s | Speed relative to Earth. Faster = more impact energy (∝ v²). |
| `miss_distance_km` | km | Closest approach distance. Smaller = scarier. |
| `absolute_magnitude` | dimensionless | Brightness. Lower number = bigger/brighter object (inverted scale). |
| `orbit_uncertainty` | 0–9 integer | How well-known the orbit is. 9 = very uncertain = riskier. |
| `inclination_deg` | degrees | Tilt of the orbit. Mostly noise in this dataset — a good lesson. |

### Label (the output we're predicting)

- `is_hazardous`: **1** = Potentially Hazardous Asteroid (PHA), **0** = Safe

### Why is the dataset imbalanced?

Only about **6% of the 6,000 rows** are labeled hazardous. This is intentional, and it mirrors reality: most space rocks that cross Earth's path are not threatening. This rarity is the entire source of the "accuracy paradox" you'll meet in Section 8.

---

## 4. Core ML Concept: Binary Classification

**Classification** means teaching a model to assign a category (class) to each input.

**Binary** means there are exactly two possible classes. In our case:
- **Class 0**: Safe (not hazardous)
- **Class 1**: Hazardous

The model learns a **decision boundary** — a rule, drawn somewhere in the space of all possible feature combinations, that separates the two classes. Everything on one side of the boundary gets called "safe"; everything on the other side gets called "hazardous."

Real-world binary classification problems you might recognize:
- Email spam detection (spam / not spam)
- Medical diagnosis (disease / no disease)
- Fraud detection (fraud / legitimate)
- Credit scoring (default / repay)

These all share the same structure as our asteroid problem — and critically, they all have the same issue: mistakes in one direction are usually much worse than mistakes in the other.

---

## 5. Core ML Concept: Class Imbalance

**Class imbalance** means the two categories are not represented equally in the data.

In our dataset: 94% safe, 6% hazardous. That 6% is the **minority class** — the one we care most about detecting.

### Why is this a problem?

A model that learns from imbalanced data is implicitly rewarded for mostly ignoring the minority class. If you get a "point" for every correct prediction, you can earn a very high score by just always saying "safe", because 94% of the time you'd be right. The model never learns to recognize the thing you actually care about.

### How we address it in this project

Two ways, both explained in detail later:
1. **`class_weight="balanced"`** in the model — makes the training algorithm treat mistakes on the rare class as more costly.
2. **Threshold tuning** — adjusting the probability cutoff so the model's output favors catching hazards over being conservative.

---

## 6. Core ML Concept: The Confusion Matrix (THE BIG ONE)

This is the heart of the project. Read this section carefully.

A confusion matrix is a **2×2 grid** that counts every possible outcome of a binary classifier on a test set. The name "confusion" literally refers to cases where the model confuses one class for the other.

```
                        PREDICTED
                    Safe        Hazardous
              ┌───────────┬───────────────┐
ACTUAL  Safe  │     TN    │      FP       │
              ├───────────┼───────────────┤
      Hazard  │     FN    │      TP       │
              └───────────┴───────────────┘
```

### The four cells

**True Negative (TN)**
- Model predicted: Safe
- Reality: Safe
- Meaning: Correctly identified a safe object. No action needed, no mistake made. ✅

**False Positive (FP)**
- Model predicted: Hazardous
- Reality: Safe
- Meaning: False alarm. We'll waste some time checking it out, but no harm done. ⚠️
- Also called: Type I Error

**False Negative (FN)**
- Model predicted: Safe
- Reality: Hazardous
- Meaning: **Missed threat.** This is the dangerous cell. A real hazard slipped through. 🚨
- Also called: Type II Error, Miss

**True Positive (TP)**
- Model predicted: Hazardous
- Reality: Hazardous
- Meaning: Real threat, correctly caught. The system did its job. ✅

### Reading our model's confusion matrix (default threshold)

```
                    Predicted SAFE   Predicted HAZARDOUS
  Actually SAFE         TN=1326         FP=84
  Actually HAZARDOUS    FN=18           TP=72
```

- **1326** safe objects correctly left alone ✅
- **84** false alarms — safe objects we unnecessarily flagged ⚠️
- **18** missed threats — real hazards we called safe 🚨 (this is the number we want to drive down)
- **72** real hazards correctly caught ✅

### After threshold tuning (threshold = 0.35)

```
                    Predicted SAFE   Predicted HAZARDOUS
  Actually SAFE         TN=1282         FP=128
  Actually HAZARDOUS    FN=9            TP=81
```

- FN dropped from **18 → 9** (caught more real threats) ✅
- FP rose from **84 → 128** (more false alarms, acceptable trade-off) ⚠️

That single comparison — two confusion matrices side by side — tells you more about the model's real-world behavior than any single-number metric could.

---

## 7. Core ML Concept: Metrics Derived from the Confusion Matrix

Every common classification metric is just a ratio you can calculate from TN, FP, FN, TP. Knowing this means you never have to memorize formulas — you can always re-derive them by asking "what fraction do I want to express?"

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
"What fraction of ALL predictions were correct?"
- **When it's useful**: balanced datasets where both error types cost the same.
- **When it lies**: imbalanced datasets (see Section 8).

### Precision
```
Precision = TP / (TP + FP)
```
"Of everything we flagged as hazardous, how much was actually hazardous?"
- High precision = low false alarm rate.
- Optimize precision when **false alarms are expensive** (e.g. arresting innocent people, ordering unnecessary surgery).

### Recall (Sensitivity / True Positive Rate)
```
Recall = TP / (TP + FN)
```
"Of all the things that were actually hazardous, how much did we find?"
- High recall = low miss rate.
- **Optimize recall when missing a real positive is catastrophic** — medical screening, fraud detection, asteroid detection.
- This is the PRIMARY metric for Sentinel-12.

### Specificity (True Negative Rate)
```
Specificity = TN / (TN + FP)
```
"Of all safe objects, how many did we correctly leave alone?"
- The "recall for the negative class."
- Useful when you also care about not over-flagging safe items.

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- The **harmonic mean** of precision and recall.
- A single number that balances both — useful for comparing models quickly.
- Treats precision and recall as equally important, which isn't always true.
- For this project we report it but don't optimize for it (we care more about recall).

### Quick cheat sheet
| Metric | Numerator | Denominator | High value means |
|---|---|---|---|
| Accuracy | TP + TN | All predictions | Mostly right overall |
| Precision | TP | All predicted positive | Few false alarms |
| Recall | TP | All actually positive | Few missed threats |
| Specificity | TN | All actually negative | Few false alarms (negative class) |
| F1 | — | — | Balanced precision & recall |

---

## 8. Core ML Concept: The Accuracy Paradox

This is the single most important "aha" moment of the project.

**Scenario**: you take a trained model and throw it away. Replace it with a rule that says:

```python
def stupid_model(asteroid):
    return "SAFE"  # always, for every single input
```

Run this on our test set (1500 asteroids, 90 are hazardous, 1410 are safe):
- It gets 1410 right (all the safe ones).
- It misses all 90 hazardous ones.
- **Accuracy = 1410/1500 = 94%.**

94% accurate! Higher than most "real" models! And completely, dangerously useless.

**Why this happens**: Accuracy's formula rewards you for the big, easy TN block. When TN is large (lots of truly safe objects), accuracy gets pulled toward a high value even if TP is zero.

**The fix**: look at the confusion matrix, and focus on **recall** for the class you care about. A model with 94% accuracy and 0% recall on hazards is worse than useless — it gives false confidence.

This pattern shows up in many real domains:
- Disease screening: most people don't have the rare disease. "Predict healthy for everyone" → high accuracy, zero medical value.
- Fraud detection: most transactions are legitimate. "Predict legit for everything" → high accuracy, fraud runs unchecked.
- Quality control: most products pass. "Predict pass for everything" → 98% accurate, defects ship to customers.

---

## 9. Core ML Concept: Decision Threshold Tuning

A classifier doesn't think in "yes/no" — it thinks in **probabilities**.

When our Random Forest classifies an asteroid, what it really computes is something like: *"I'm 73% confident this is hazardous."* The `.predict()` method then applies a **default threshold of 0.5**: if confidence ≥ 50%, predict hazardous; otherwise predict safe.

But 0.5 is just a default. **You can change it.**

### What lowering the threshold does

Imagine setting the threshold from 0.5 to 0.35:
- Now anything the model is ≥35% confident about gets flagged as hazardous.
- More objects get flagged → more true hazards caught (**recall goes up**).
- But also more safe objects wrongly flagged (**precision goes down, more FP**).

| Threshold | Recall | Precision | FN (missed) | FP (false alarms) |
|-----------|--------|-----------|-------------|-------------------|
| 0.50 | 0.80 | 0.46 | 18 | 84 |
| 0.35 | 0.90 | 0.39 | 9 | 128 |
| 0.20 | 0.94 | 0.33 | 5 | 172 |

Each row is a different version of the same model. You're not retraining — you're just changing what probability cutoff triggers an alert.

**The key insight**: threshold selection is a *product decision*, not a math problem. It encodes your answer to the question: "How many false alarms am I willing to accept in exchange for catching one more real threat?"

---

## 10. Core ML Concept: The Precision-Recall Trade-off

Precision and recall are **inversely related** when you tune the threshold. This is not a flaw — it is the fundamental trade-off of any detection system.

```
        High threshold → conservative → few flags → high precision, low recall
        Low threshold  → aggressive  → many flags → low precision, high recall
```

The precision-recall curve (`outputs/precision_recall_curve.png`) plots this trade-off across all possible thresholds at once. Each point on the curve is one possible threshold. The curve shows you the best achievable recall *for any given precision target*, and vice versa.

A model that hugs the **top-right corner** of the curve (high precision AND high recall) is ideal. A model that collapses to the bottom-left is useless.

**In Sentinel-12**, we deliberately move toward the bottom-right of the curve: lower precision (more false alarms) in exchange for higher recall (fewer missed threats). That trade is worth it given what a miss costs.

---

## 11. Core ML Concept: ROC Curve and AUC

**ROC** stands for Receiver Operating Characteristic — a name from WWII radar signal theory, which is fitting given this project's theme.

The ROC curve plots two things as you sweep the threshold from 1.0 down to 0.0:
- **X-axis**: False Positive Rate = FP / (FP + TN) = 1 − Specificity
- **Y-axis**: True Positive Rate = Recall = TP / (TP + FN)

As the threshold drops, you catch more real hazards (Y goes up) but also flag more safe objects (X goes up). A perfect model would shoot straight up the Y-axis and across the top — it would reach 100% recall before generating any false positives.

**AUC (Area Under the Curve)** is the area under the ROC curve, between 0 and 1:
- **AUC = 1.0**: perfect model
- **AUC = 0.5**: random guessing (diagonal line)
- **AUC = 0.0**: perfectly wrong

Our model achieves **AUC ≈ 0.952**, which means it's doing very well at *ranking* hazardous objects above safe ones — the model is genuinely learning real signal from the data.

### ROC vs Precision-Recall Curve: when to use which

| Scenario | Prefer |
|---|---|
| Balanced classes | ROC / AUC |
| Severe class imbalance (like ours) | Precision-Recall Curve |
| You care about the minority class | Precision-Recall Curve |

With imbalanced data, ROC can look optimistically good even when the model fails on the rare class. The precision-recall curve is more honest in that situation.

---

## 12. Core ML Concept: Feature Engineering

The features you give the model matter more than which model you choose. This is one of the most widely held beliefs among experienced ML practitioners.

In `generate_data.py`, we made deliberate choices about features:

**Informative features** (`est_diameter_km`, `miss_distance_km`, `relative_velocity_kms`, `orbit_uncertainty`): all encoded in the hidden "risk score" formula. The model can learn from these.

**Redundant feature** (`absolute_magnitude`): computed from `est_diameter_km` with added noise. It's correlated with diameter, so it adds some redundant information — just like real-world data often contains.

**Noise feature** (`inclination_deg`): not related to hazard classification at all. A good model should learn to downweight this feature. This teaches you that not all columns in a dataset are useful, and adding irrelevant features can hurt some models.

You can explore the **feature importance** output from the Random Forest to see which features the model actually relied on most.

---

## 13. Core ML Concept: Train / Test Split and Data Leakage

### The split

We divide our 6,000 rows into:
- **Training set (75%, 4,500 rows)**: the model sees and learns from this data.
- **Test set (25%, 1,500 rows)**: held back completely; the model NEVER sees it during training.

Evaluating on the test set is the only honest way to know how the model will perform on **new, unseen data** — which is the only thing that matters in practice.

### Stratified splitting

We use `stratify=y` when calling `train_test_split`. Without this, a random split might accidentally put very few hazardous examples in the test set. Stratification guarantees both sets keep the original 6% / 94% class ratio.

### Data Leakage (a very common mistake)

**Data leakage** means information from the test set accidentally influences the training process. The result is a model that looks great on your test metrics but fails in production.

The most common form: fitting preprocessing (like scaling) on the full dataset including the test set.

We avoid this by:
1. Fitting the `StandardScaler` **only** on `X_train`.
2. Applying the fitted scaler to `X_test` using `.transform()` only (not `.fit_transform()`).

The scaler has "seen" only training data statistics. The test set is treated as if it came from the future, which is exactly what production data is.

---

## 14. Core ML Concept: Feature Scaling

Our features live on wildly different numeric scales:
- `est_diameter_km`: 0.001 – 15
- `miss_distance_km`: 50,000 – 9,000,000

Some algorithms (distance-based ones like KNN, SVMs, regularized regression) treat large-valued features as inherently more important just because their numbers are bigger, even if the actual signal is equivalent. **Scaling removes this bias.**

`StandardScaler` transforms each feature column to have:
- **Mean = 0**
- **Standard deviation = 1**

This is called **standardization** (or Z-score normalization). Formula for each value:

```
z = (x - mean) / std
```

After scaling, a value of 1.0 in any feature means "one standard deviation above average for this feature," regardless of the original units.

Random forests don't strictly require scaling (they're tree-based and compare relative order, not absolute magnitude). We include it here as best practice, because it's essential when you switch to other algorithms, and it's a habit worth building from the start.

---

## 15. Core ML Concept: The Random Forest Model

A **Decision Tree** is a flowchart of yes/no questions about the features, ending in a prediction. Example: "Is `miss_distance_km` < 500,000? → Yes → Is `est_diameter_km` > 0.5? → Yes → Predict HAZARDOUS."

Decision trees are easy to understand but brittle: small changes to the training data can produce wildly different trees (high variance).

A **Random Forest** fixes this by:
1. Building many decision trees (200, in our case) on random subsets of the training data.
2. At each split in each tree, considering only a random subset of features.
3. For classification, having all trees vote and taking the majority.

The randomness + averaging eliminates the brittleness of single trees. This is called **ensemble learning** — combining many weak-but-diverse learners into a stronger, more stable one.

**Why we chose Random Forest for this project**: it gives us `.predict_proba()` (probabilities, not just yes/no), it handles the non-linear relationships between features in our synthetic data, and it doesn't require careful hyperparameter tuning to work reasonably well — so it doesn't distract from the actual lesson (evaluation metrics).

---

## 16. Core ML Concept: class_weight and Handling Imbalance in Training

When we set `class_weight="balanced"`, scikit-learn automatically computes a weight for each class:

```
weight for class c = total_samples / (num_classes × count_of_class_c)
```

For us:
- Safe (class 0): 4500 total / (2 × 4230 safe) ≈ 0.53 — lower weight
- Hazardous (class 1): 4500 total / (2 × 270 hazardous) ≈ 8.33 — higher weight

This means: every time the model makes a mistake on a hazardous object during training, that mistake is penalized ~8.3× harder than a mistake on a safe object. The model is therefore **forced to pay attention to the rare class** rather than lazily predicting "safe" most of the time.

This is one of three common strategies for imbalanced data:
1. **class_weight** (what we use): adjust the loss function.
2. **Oversampling** (e.g. SMOTE): create synthetic minority-class examples to balance the dataset.
3. **Undersampling**: remove majority-class examples. Throws away real data, generally less preferred.

---

## 17. Core ML Concept: Probability Output vs Hard Prediction

`model.predict(X)` → returns `[0, 1, 0, 0, 1, ...]` — a hard yes/no for each row.

`model.predict_proba(X)` → returns `[[0.82, 0.18], [0.27, 0.73], ...]` — for each row, a probability for each class. Column 0 is P(safe), column 1 is P(hazardous).

The hard prediction is just the probability output with a threshold of 0.5 applied. By using `predict_proba`, you gain the ability to apply **any** threshold you want — which is exactly what `evaluate_model.py` does in the threshold tuning section.

The practical workflow:
```python
probs = model.predict_proba(X_test)[:, 1]   # hazard probability for each object
preds = (probs >= 0.35).astype(int)          # flag if probability ≥ 35%
```

This two-step approach gives you full control over the precision-recall trade-off at deployment time, without any retraining.

---

## 18. The Full Learning Journey: Concept Map

```
Data Generation
│
├── Synthetic data design ──────────────── Feature Engineering (§12)
├── Class imbalance (6% hazardous) ─────── Class Imbalance (§5)
└── Reproducible random seed ───────────── Good Practice

          │
          ▼

Model Training
│
├── Train/test split + stratify ─────────── Train/Test Split & Leakage (§13)
├── StandardScaler (fit on train only) ──── Feature Scaling (§14)
├── RandomForestClassifier ──────────────── Random Forest (§15)
└── class_weight="balanced" ────────────── Handling Imbalance (§16)

          │
          ▼

Evaluation
│
├── predict_proba vs predict ────────────── Probability vs Hard Pred (§17)
├── Confusion Matrix ────────────────────── THE BIG ONE (§6)
├── TP / TN / FP / FN labeled ──────────── Confusion Matrix Cells (§6)
├── Accuracy, Precision, Recall, F1 ─────── Derived Metrics (§7)
├── The "do-nothing" model demo ─────────── Accuracy Paradox (§8)
├── Threshold sweep table ───────────────── Threshold Tuning (§9)
├── Precision-Recall curve ──────────────── PR Trade-off (§10)
└── ROC curve + AUC ─────────────────────── ROC & AUC (§11)
```

---

## 19. What to Experiment With Next

Once you've run the full pipeline and understand the outputs, here are concrete ways to deepen your understanding by **changing things and observing what breaks or improves**:

### Easy experiments
- Change `target_recall` in `evaluate_model.py` from 0.90 to 0.80 or 0.95. How does the threshold change? The confusion matrix?
- Change the class imbalance: go to `generate_data.py` and change the 94th percentile cutoff to, say, 80th percentile, making 20% of objects hazardous. Re-run everything. What happens to the accuracy paradox? Is it still as dramatic?
- Remove `class_weight="balanced"` from the model in `train_model.py`. Re-run. Compare the confusion matrices. Does recall drop?

### Medium experiments
- Add a feature to `generate_data.py` that is pure random noise (unrelated to hazard). Does the model's AUC drop?
- Switch from `RandomForestClassifier` to `LogisticRegression`. Does the recall-vs-precision trade-off look different?
- Try SMOTE (from the `imbalanced-learn` library) as an alternative to `class_weight`. Does it give better results?

### Harder experiments
- Plot the **feature importances** from the trained Random Forest (`model.feature_importances_`). Does it correctly rank size and distance as most important and inclination as least?
- Compute the **calibration curve** to check whether the model's probability scores are trustworthy (does "60% probability hazardous" actually correspond to 60% of the time being hazardous?).
- Try cross-validation instead of a single train/test split. Are the metrics stable across folds?

---

## 20. Glossary: Every Term in One Place

| Term | One-line definition |
|---|---|
| **AUC** | Area under the ROC curve; summarizes ranking quality from 0 to 1. |
| **Accuracy** | Fraction of all predictions that were correct. Misleading on imbalanced data. |
| **Binary classification** | Predicting one of exactly two classes. |
| **Class imbalance** | When one class is much rarer than the other in the dataset. |
| **Class weight** | A penalty multiplier applied during training to focus the model on rare classes. |
| **Confusion matrix** | 2×2 grid counting TP, TN, FP, FN for all test predictions. |
| **Data leakage** | When test-set information accidentally influences training. Inflates metrics. |
| **Decision boundary** | The rule that separates predicted positive from predicted negative regions. |
| **Decision threshold** | Probability cutoff for converting a probability into a hard yes/no prediction. |
| **Ensemble learning** | Combining many models (e.g. decision trees) to get a more stable prediction. |
| **F1 score** | Harmonic mean of precision and recall; a balanced single metric. |
| **False Negative (FN)** | Model predicted negative (safe), reality was positive (hazardous). Missed threat. |
| **False Positive (FP)** | Model predicted positive (hazardous), reality was negative (safe). False alarm. |
| **Feature** | A measured input variable used to make a prediction. |
| **Feature engineering** | Designing or transforming features to make them more useful to the model. |
| **Feature importance** | A Random Forest measure of how much each feature contributed to predictions. |
| **Feature scaling** | Transforming features to a common numeric range (e.g. mean 0, std 1). |
| **Label** | The output variable we're trying to predict. |
| **Minority class** | The less frequent class in an imbalanced dataset. The one we usually care most about. |
| **Precision** | TP / (TP + FP). Fraction of positive predictions that were actually positive. |
| **Precision-Recall curve** | Plot of precision vs recall as threshold varies. Better than ROC for imbalanced data. |
| **Random Forest** | Ensemble of many decision trees with randomized training. Robust and versatile. |
| **Recall** | TP / (TP + FN). Fraction of actual positives that were correctly detected. |
| **ROC curve** | Plot of True Positive Rate vs False Positive Rate across all thresholds. |
| **Specificity** | TN / (TN + FP). Recall for the negative class. |
| **StandardScaler** | Scikit-learn tool that subtracts mean and divides by std dev for each feature. |
| **Stratified split** | Train/test split that preserves the class ratio in both subsets. |
| **Synthetic data** | Data created by code rather than measured from the real world. |
| **Threshold tuning** | Choosing a non-default decision threshold to shift the precision-recall balance. |
| **Train/test split** | Dividing data into a training portion and a held-back evaluation portion. |
| **True Negative (TN)** | Model predicted negative, reality was negative. Correctly left alone. |
| **True Positive (TP)** | Model predicted positive, reality was positive. Correct detection. |
| **Type I Error** | False Positive. Rejected the true null hypothesis. |
| **Type II Error** | False Negative. Failed to reject a false null hypothesis. |
