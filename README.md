# Class-balanced LightGBM vs MLP for lightweight perception

Software-defined vehicles need perception workflows that are both accurate and fast enough for real-time use, as discussed in "Toward real-time and efficient perception workflows in software-defined vehicles". Deep neural networks are the usual choice for perception, but their inference cost and sensitivity to sensor noise matter in production. This experiment asks whether a tree-based model can be a competitive, cheaper alternative on a small image classification task.

**Question:** Can a class-balanced, CV-tuned LightGBM model match or beat a standard MLP perception stand-in on macro-F1 while giving lower per-sample inference latency and better accuracy under input noise?

**Dataset:** scikit-learn's bundled handwritten-digit image set, loaded locally with no download. It is a small stand-in for camera-based perception and is not the paper's own dataset.

**Baseline:** a standard scikit-learn MLPClassifier with one hidden layer of 128 units, trained on standardised pixels.

**Improved model:** a class-balanced LightGBM classifier whose number of trees and learning rate are chosen by 3-fold stratified cross-validation on the training split only.

Both models use the same stratified train/test split, the same preprocessing, the same evaluation function and the same timing protocol. Results are repeated over several random seeds.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and figures/comparison.png).

## Reference

This project is motivated by the following paper:

> R Jafarpourmarzouni, Y Luo, S Lu (2024). *Toward real-time and efficient perception workflows in software-defined vehicles*. IEEE Internet of Things. https://ieeexplore.ieee.org/abstract/document/10745602/

Paper author: Prof. Zheng Dong, Wayne State University (dong@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
