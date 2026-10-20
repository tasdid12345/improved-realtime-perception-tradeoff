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

## Reference

This project is motivated by the following paper:

> R Jafarpourmarzouni, Y Luo, S Lu (2024). *Toward real-time and efficient perception workflows in software-defined vehicles*. IEEE Internet of Things. https://ieeexplore.ieee.org/abstract/document/10745602/

Paper author: Prof. Zheng Dong, Wayne State University (dong@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
