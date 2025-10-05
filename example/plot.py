# tsne_visualization_from_trainer.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# ======== Import Trainer and builder ==========
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
import easydict

# ======== Define Configuration (cfg) ==========
cfg = easydict.EasyDict({
    "dataset": "cifar10",
    "imb_type": "exp",
    "imb_factor": 0.01,
    "rand_number": 0,
    "workers": 4,
    "backbone": "resnet32",
    "classifier": "dot_product_classifier",
    "gpu": 0,
    "start_epoch": 0,
    "epochs": 200,
    "batch_size": 128,
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 2e-4,
    "k_majority": 3,
    "tau": 0.5,
    "mamix_ratio": -0.25,
    "print_freq": 10,
    "root_log": "./log_cifar10",
    "root_model": "./checkpoint_cifar10",
    "store_name": "cifar10_exp_0.01_ERM_200_1126",  # very important for logging
    "warm": 160,
    "beta": 0.9999,
    "num_classes": 10,
    "attack_iter": 10,
    "step_size": 0.1,
    "lam": 0.5,
    "gamma": 0.9,
    "ratio": 100,
    "eff_beta": 1,
    "over": True,
    "imb_start": 5,
    "cifar_root": "./data",
    "sampling": "Random",
    "n_batches": 97,
    "alpha": 0.7,
    "kind": "fixed",
    "strategy": "ERM",  # important
    "cls_num_list": None,  # will be filled later
    "best_model": None  # we manually load
})

# ======== Main Logic ==========
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. Load Dataset
    train_set = IMBALANCECIFAR10(root=cfg.cifar_root,
                                 imb_type=cfg.imb_type,
                                 imb_factor=cfg.imb_factor,
                                 rand_number=cfg.rand_number,
                                 train=True,
                                 download=True)
    
    cfg.cls_num_list = train_set.get_cls_num_list()

    # 2. Build Trainer
    trainer = build_trainer(cfg, train_set, model=None, strategy=cfg.strategy)

    # 3. Load Checkpoint
    ckpt_path = '/home/hamt/light_weight/imbalanced-DL-sampling/example/checkpoint_cifar10/cifar10_exp_0.01_ERM_200_1126/ckpt.best.pth.tar'
    checkpoint = torch.load(ckpt_path, map_location=device)
    trainer.model.load_state_dict(checkpoint['state_dict'])
    trainer.model.eval()

    print("Checkpoint loaded successfully!")

    # 4. Load CIFAR10 test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_set = CIFAR10(root=cfg.cifar_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    # 5. Extract Features
    features, labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs, feature = trainer.model(inputs)
            features.append(feature.cpu())
            labels.append(targets)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    print("Extracted feature shape:", features.shape)

    # 6. Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.numpy())

    # 7. Plot
    plt.figure(figsize=(10, 8))
    for i in range(10):
        idx = labels == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'Class {i}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of CIFAR-10 Features (ERM, ResNet32)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
