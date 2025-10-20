# imbalanced-DL: Deep Imbalanced Learning in Python

## Overview
`imbalanced-DL` (imported as `imbalanceddl`) is a Python package designed to provide an efficient and research-ready framework for deep imbalanced learning.  
Building on our recent work *"Lightweight Data Augmentation Techniques to Handle Imbalanced Learning"* (Dinh et al., 2025), this release integrates a new lightweight method called **Imbalanced Lightweight Hybrid (ILWH)** — a practical and computationally efficient approach for real-world imbalance scenarios.

Imbalanced learning remains a core challenge in machine learning where certain classes have far fewer samples than others, causing biased predictions and degraded generalization. Traditional solutions such as GAN-based oversampling or heavy ensemble architectures often demand large computational resources. In contrast, **ILWH** introduces a hybrid yet lightweight framework that balances performance and efficiency through **data-level augmentation** and **algorithm-level adaptive re-weighting**.

ILWH unifies **Mixup** (data interpolation) and **Deferred Re-Weighting (DRW)** (cost-sensitive adaptation) into a single streamlined pipeline. It achieves competitive or superior accuracy on benchmark datasets (SVHN-10, CIFAR-10, CINIC-10, CIFAR-100, and Tiny-ImageNet-200) with less than **3% runtime overhead** compared to standard ERM training.  
By combining simple augmentations (Cutout, AutoAugment, RandAugment) with adaptive class-weighting and sampling (WRBS/WFBS), ILWH enhances both **minority-class recall** and **overall stability** under extreme imbalance (imbalance ratio 0.01).

Key highlights:
* Lightweight integration requiring **no additional networks or complex losses**
* Achieves up to **+10% Top-1 Accuracy gains** on heavily imbalanced datasets
* Offers standardized implementations for **ERM**, **DRW**, **Mixup**, **M2m**, **DeepSMOTE**, and **ILWH (ours)** for fair comparison and benchmarking.

This package thus provides a **unified framework for deep imbalanced learning research**, emphasizing scalability, reproducibility, and efficiency for practitioners and researchers alike.

---

## Strategy
We provide some baseline strategies as well as some state-of-the-art strategies in this package as the following:
* Empirical Risk Minimization (baseline strategy)
* [Reweighting with Class Balance (CB) Loss](https://arxiv.org/pdf/1901.05555.pdf)
* [Deferred Re-Weighting (DRW)](https://arxiv.org/pdf/1906.07413.pdf)
* [M2m: Major-to-minor translation](https://arxiv.org/pdf/2004.00431.pdf)
* [Label Distribution Aware Margin (LDAM) Loss with DRW](https://arxiv.org/pdf/1906.07413.pdf)
* [DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data](https://arxiv.org/pdf/2105.02340.pdf)
* [Mixup with DRW](https://arxiv.org/pdf/1710.09412.pdf)
* [Remix with DRW](https://arxiv.org/pdf/2007.03943.pdf)
* [MAMix with DRW (Link Coming Soon)]()
* **[ILWH (Ours): Imbalanced Lightweight Hybrid (Mixup + DRW)](https://github.com/duydinhthai27/lightweight-ILWH)**

---

## Environments
* This package is tested on Linux OS.
* You are suggested to use a different virtual environment so as to avoid package dependency issues.
* For Pyenv & Virtualenv users, follow the below steps to create a new virtual environment:
```bash
pyenv virtualenv 3.8.10 ilwh-env
pyenv local ilwh-env
```
* Then, install dependencies and build locally with:
```bash
python -m pip install -r requirements.txt
python setup.py install
```

---

## Usage
We highlight three key features of `imbalanced-DL`:
1. **Dataset Construction:** Supports 5 benchmark datasets (CIFAR-10, CIFAR-100, CINIC-10, SVHN-10, Tiny-ImageNet-200) and imbalanced sampling creation (`imb_type`, `imb_ratio`).
2. **Strategy Trainer:** Build a `Trainer` object and specify strategy via `config.strategy = ILWH` or others.
3. **Benchmark Environment:** Predefined configs for reproducing results in `example/config/`.

Example:
```python
from imbalanceddl.strategy.build_trainer import build_trainer

trainer = build_trainer(config, imbalance_dataset, model=model,strategy =config.strategy)
trainer.do_train_val()
trainer.eval_best_model()
```

---
* Or you can also just select the specific strategy you would like to use as:


```python
from imbalanceddl.strategy import LDAMDRWTrainer

# pick the trainer
trainer = LDAMDRWTrainer(config,
                         imbalance_dataset,
                         model=model,
                         strategy=config.strategy)

# train from scratch
trainer.do_train_val()

# Evaluate with best model
trainer.eval_best_model()

```
* To construct your own strategy trainer, you need to inherit from [`Trainer`](https://github.com/ntucllab/imbalanced-DL/blob/e63acaab958bf206edad9418e9c30352e9566356/imbalanceddl/strategy/trainer.py#L11) class, where in your own strategy you will have to implement `get_criterion()` and `train_one_epoch()` method. After this you can choose whether to add your strategy to `build_trainer()` function or you can just use it as the above demonstration.


(2) Benchmark research environment:
* To conduct deep imbalanced learning research, we provide example codes for training with different strategies, and provide benchmark results on five image datasets. To quickly start training CIFAR-10 with ERM strategy, you can do:

```
cd example
python main.py --gpu 0 --seed 1126 --c config/config_cifar10.yaml --strategy ERM

```
* Following the example code, you can not only get results from baseline training as well as state-of-the-art performance such as [LDAM](https://arxiv.org/pdf/1906.07413.pdf) or [Remix](https://arxiv.org/pdf/2007.03943.pdf), but also use this environment to develop your own algorithm / strategy. Feel free to add your own strategy into this package.
* For more information about example and usage, please see the [Example README](https://github.com/ntucllab/imbalanced-DL/tree/main/example)

## Benchmark Results
Results across multiple datasets with imbalance ratio 0.01:

| Dataset | ERM | DRW | Mixup | ILWH (Ours) |
|:--------:|:---:|:---:|:-----:|:------------:|
| **CIFAR-10** | 72.74 | 76.03 | 73.37 | **82.19** |
| **SVHN-10** | 81.60 | 80.57 | 80.73 | **85.52** |
| **CINIC-10** | 60.65 | 64.29 | 63.75 | **71.21** |
| **CIFAR-100** | 38.87 | 39.46 | 41.57 | **47.07** |
| **Tiny-ImageNet-200** | 34.17 | 33.90 | 33.69 | **38.69** |

**Efficiency comparison (CIFAR-10):**
| Method | Avg FLOPs/Epoch | Relative Cost | Runtime Overhead |
|:-------|:----------------:|:--------------:|:----------------:|
| M2m | 6,589.37G | 1.00× | baseline |
| DeepSMOTE | ≈9,884–13,179G | 1.5–2.0× | heavy |
| **ILWH (Ours)** | 6,589.37G | **0.80×** | **<3% overhead** |

---

## Contact
If you have any question, please don't hesitate to email `wccheng3011@gmail.com` or `maitanhaksdtvt6@gmail.com`. Thanks !


---
