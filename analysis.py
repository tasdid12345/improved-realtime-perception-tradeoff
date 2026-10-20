"""Baseline MLP vs class-balanced, CV-tuned LightGBM on scikit-learn's bundled digits set.
Writes results.json and figures/comparison.png. Seed 42 everywhere; repeats use 42, 43, 44."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI runners
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]
NOISE_SIGMA = 2.0  # Gaussian pixel noise on the raw 0-16 pixel scale
N_LATENCY = 200    # single-sample predictions timed per model


def log(msg):
    print(f"[analysis] {msg}", flush=True)


# 1. Data: bundled digits (no download); one stratified train/test split
data = load_digits()
X, y = data.data.astype(np.float64), data.target
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
scaler = StandardScaler().fit(X_tr_raw)  # training statistics only
X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
# Noisy test set: one fixed noise draw, shared by both models
rng = np.random.default_rng(SEED)
X_noisy = scaler.transform(np.clip(X_te_raw + rng.normal(0, NOISE_SIGMA, X_te_raw.shape), 0, 16))
log(f"rows={len(X)} train={len(X_tr)} test={len(X_te)}")


def make_baseline(seed):
    # Standard MLP stand-in for a deep perception model: one hidden layer of 128 units
    return MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)


def make_improved(seed):
    # Class-balanced LightGBM; trees and learning rate picked by 3-fold CV on train only
    return GridSearchCV(
        LGBMClassifier(class_weight="balanced", n_jobs=1, random_state=seed, verbose=-1),
        param_grid={"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="f1_macro", n_jobs=1, refit=True)


def evaluate(model):
    # Same protocol for both models: macro-F1, accuracy, noisy accuracy, single-sample latency
    pred = model.predict(X_te)
    res = {"f1": f1_score(y_te, pred, average="macro"),
           "acc": accuracy_score(y_te, pred),
           "noisy_acc": accuracy_score(y_te, model.predict(X_noisy))}
    x1 = X_te[:N_LATENCY]
    for x in x1[:10]:  # warm-up
        model.predict(x.reshape(1, -1))
    t0 = time.perf_counter()
    for x in x1:
        model.predict(x.reshape(1, -1))
    res["latency_ms"] = (time.perf_counter() - t0) / N_LATENCY * 1000
    return res


# 2. Train and evaluate both models for each seed (test set used only in evaluate)
runs = {"baseline": [], "improved": []}
chosen = []
for s in SEEDS:
    log(f"seed {s}: fitting baseline MLP")
    runs["baseline"].append(evaluate(make_baseline(s).fit(X_tr, y_tr)))
    log(f"seed {s}: CV-tuning and fitting class-balanced LightGBM")
    gs = make_improved(s).fit(X_tr, y_tr)  # CV uses training data only
    chosen.append(gs.best_params_)
    runs["improved"].append(evaluate(gs.best_estimator_))
    log(f"seed {s}: best CV params {gs.best_params_}")


def agg(name, key):
    vals = [r[key] for r in runs[name]]
    return float(np.mean(vals)), float(np.std(vals))


b = {k: agg("baseline", k) for k in ["f1", "acc", "noisy_acc", "latency_ms"]}
i = {k: agg("improved", k) for k in ["f1", "acc", "noisy_acc", "latency_ms"]}
metrics = {
    "baseline_score": b["f1"][0], "baseline_score_std": b["f1"][1],
    "improved_score": i["f1"][0], "improved_score_std": i["f1"][1],
    "improvement": i["f1"][0] - b["f1"][0],  # higher macro-F1 is better
    "baseline_accuracy": b["acc"][0], "improved_accuracy": i["acc"][0],
    "baseline_noisy_accuracy": b["noisy_acc"][0], "improved_noisy_accuracy": i["noisy_acc"][0],
    "baseline_latency_ms": b["latency_ms"][0], "improved_latency_ms": i["latency_ms"][0],
}
if not all(np.isfinite(v) for v in metrics.values()) or metrics["baseline_score"] <= 0:
    log("metrics are not meaningful; exiting")
    sys.exit(1)

# 3. Chart: macro-F1 and noisy accuracy with std error bars
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, key, title in [(axes[0], "f1", "Test macro-F1"), (axes[1], "noisy_acc", "Accuracy, noisy test")]:
    ax.bar(["baseline MLP", "improved LightGBM"], [b[key][0], i[key][0]],
           yerr=[b[key][1], i[key][1]], capsize=4, color=["gray", "tab:blue"])
    ax.set_title(title)
    ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/comparison.png", dpi=120)
log("saved figures/comparison.png")

# 4. Write results.json
results = {
    "question": "Can a class-balanced, CV-tuned LightGBM model match or beat a standard MLP perception "
                "stand-in on macro-F1 while giving lower per-sample inference latency and better "
                "accuracy under input noise?",
    "method": "Stratified 75/25 split of scikit-learn digits; features standardised on training data. "
              "Baseline MLPClassifier (128 hidden units) vs class-balanced LightGBM with n_estimators and "
              "learning_rate chosen by 3-fold CV on training data. Test metrics, latency and noisy "
              "accuracy computed for both; repeated for seeds 42, 43, 44.",
    "dataset": {
        "name": "scikit-learn digits (load_digits, bundled)",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "n_rows": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "macro-F1 on held-out test split",
    "higher_is_better": True,
    "baseline": "scikit-learn MLPClassifier, one hidden layer of 128 units, trained on standardised pixels",
    "improved": "Class-balanced LightGBM classifier; n_estimators and learning_rate selected by 3-fold "
                "stratified CV on the training split only",
    "metrics": metrics,
    "notes": (f"Digits is a small stand-in for camera perception, not a driving dataset, so results do not "
              f"show real-world performance. Noise is Gaussian sigma={NOISE_SIGMA} on the raw pixel scale, "
              f"one fixed draw shared by both models. Latency is mean wall-clock time per single-sample "
              f"predict() over {N_LATENCY} test rows on this runner (LightGBM single-threaded); it is "
              f"hardware-dependent. Std values are across seeds {SEEDS}; the test split is identical for all. "
              f"Chosen LightGBM params per seed: {chosen}."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log("wrote results.json")
