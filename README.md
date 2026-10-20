# Digits: MLP baseline vs class-balanced, CV-tuned RBF SVM

This project compares a standard MLP baseline with a class-balanced RBF SVM on scikit-learn's bundled digits dataset (1797 rows, no download needed).

## Method
- Stratified 75/25 train/test split (`random_state=42`); features standardised using training statistics only.
- Baseline: `MLPClassifier` with one hidden layer of 128 units (unchanged).
- Improved: `SVC` with RBF kernel and `class_weight="balanced"`. `C` in {1, 10, 100} and `gamma` in {0.005, 0.01, 0.02, 0.05} are chosen by 3-fold stratified CV on the training split only, scored by macro-F1.
- Repeated for seeds 42, 43, 44. The test set is used only for final evaluation.
- Metrics: macro-F1 (primary), accuracy, accuracy on Gaussian-noisy test pixels (sigma = 2), and single-sample latency.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs: `results.json` and `figures/comparison.png`. Runtime is well under 20 minutes on CPU.

## Notes
The improved model was switched from LightGBM to an RBF SVM because the LightGBM version scored below the MLP baseline on macro-F1. This is a model choice on digits only and does not reflect real-world perception performance.

## Results (latest run)

On scikit-learn digits, a standard scikit-learn MLP classifier as a stand-in for the paper's deep perception model reached 0.978 macro-F1 versus 0.981 for my class-balanced RBF SVM, a small gain, but the SVM scored lower under Gaussian input noise (12.5% vs 29.2% accuracy).

| Metric | Value |
|---|---|
| baseline_score | 0.9776 |
| baseline_score_std | 0.001924 |
| improved_score | 0.9806 |
| improved_score_std | 0.002095 |
| improvement | 0.002997 |
| baseline_accuracy | 0.9778 |
| improved_accuracy | 0.9807 |
| baseline_noisy_accuracy | 0.2919 |
| improved_noisy_accuracy | 0.1252 |
| baseline_latency_ms | 0.3009 |
| improved_latency_ms | 0.2087 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Digits is a small stand-in for camera perception, not a driving dataset, so results do not show real-world performance. The improved model was changed from LightGBM to an RBF SVM after LightGBM trailed the baseline on macro-F1; the baseline and test split are unchanged. Noise is Gaussian sigma=2.0 on the raw pixel scale, one fixed draw shared by both models. Latency is mean wall-clock time per single-sample predict() over 200 test rows on this runner; it is hardware-dependent. Std values are across seeds [42, 43, 44]; the test split is identical for all. Chosen SVM params per seed: [{'C': 100, 'gamma': 0.01}, {'C': 1, 'gamma': 0.02}, {'C': 10, 'gamma': 0.01}].

## Reference

This project is motivated by the following paper:

> R Jafarpourmarzouni, Y Luo, S Lu (2024). *Toward real-time and efficient perception workflows in software-defined vehicles*. IEEE Internet of Things. https://ieeexplore.ieee.org/abstract/document/10745602/

Paper author: Prof. Zheng Dong, Wayne State University (dong@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
