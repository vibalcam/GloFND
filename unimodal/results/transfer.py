import os
import csv
import random
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
import re
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Set random seed for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Argument parser
parser = argparse.ArgumentParser(description="Image classification using pre-trained models")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=models.__dict__.keys(),
                    help='Model architecture from torchvision (default: resnet50)')
parser.add_argument('--pretrained', type=str, default="fnd",
                    help='Path to pretrained model checkpoint')
parser.add_argument('--name', type=str, default="default", help='Name of the experiment')
args = parser.parse_args()

# Define datasets and evaluation protocols
DATASETS = {
    "Caltech101": {"loader": datasets.Caltech101, "metric": "mean-per-class", "split": True},
    "StanfordCars": {"loader": datasets.StanfordCars, "metric": "top-1", "split": True},
    "CIFAR10": {"loader": datasets.CIFAR10, "metric": "top-1", "split": False},
    "CIFAR100": {"loader": datasets.CIFAR100, "metric": "top-1", "split": False},
    "Food101": {"loader": datasets.Food101, "metric": "top-1", "split": True},
    "DTD": {"loader": datasets.DTD, "metric": "top-1", "split": True},
    "OxfordPets": {"loader": datasets.OxfordIIITPet, "metric": "mean-per-class", "split": True, "split_key": "trainval"},
    "OxfordFlowers": {"loader": datasets.Flowers102, "metric": "mean-per-class", "split": True},
}

# Define preprocessing transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=mean, std=std),
])

# Feature extraction function
@torch.inference_mode()
def extract_features(loader, model, device='cuda'):
    features, labels = [], []
    model = model.to(device)
    for images, targets in tqdm(loader, desc="Extracting Features", leave=False):
        images = images.to(device)
        outputs = model(images)
        outputs = F.normalize(outputs, p=2, dim=1)
        features.append(outputs.cpu().numpy())
        labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

# Evaluate and save predictions
def evaluate_with_gridsearch(X_train, y_train, X_test, y_test, metric="top-1"):
    n_trials = 10
    param_grid = {'C': np.logspace(-6, 5, n_trials)}
    
    if metric == "top-1":
        scoring = make_scorer(accuracy_score)
    elif metric == "mean-per-class":
        def mean_per_class_score(y_true, y_pred):
            return np.mean([
                accuracy_score(y_true[y_true == c], y_pred[y_true == c]) 
                for c in np.unique(y_true)
            ])
        scoring = make_scorer(mean_per_class_score)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cv = PredefinedSplit(test_fold=np.hstack([-np.ones(len(X_train)), np.zeros(len(X_test))]))
    clf = GridSearchCV(
        LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500, n_jobs=3, verbose=0, random_state=42),
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_trials,
        verbose=3,
        refit=False,
    )
    clf.fit(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
    print(clf.cv_results_)
    test_score = round(clf.best_score_ * 100, 2)
    print(f"Best {metric} Score = {test_score:.2f}")

    return test_score


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    names = []

    with torch.inference_mode():
        # Iterate over provided folders
        print(f"Processing folder: {args.pretrained}...")
        
        # Get all checkpoint files (for 3 runs per folder)
        checkpoint_paths = [os.path.join(args.pretrained, f"{i}/checkpoint_0199.pth.tar") for i in range(1, 4)]
        
        # Initialize list to store scores for each dataset (per run)
        folder_scores = {}

        # Evaluate each run
        for checkpoint_path in checkpoint_paths:
            # Load model
            print(f"=> Creating model '{args.arch}'")
            model = models.__dict__[args.arch](pretrained=False)

            # Remove the final fully connected layer and add an identity layer
            if hasattr(model, 'fc'):
                linear_keyword = 'fc'
                hidden_dim = model.fc.weight.shape[1]
                model.fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                linear_keyword = 'classifier'
                hidden_dim = model.classifier.weight.shape[1]
                model.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported model architecture: {args.arch}")
        
            # Load checkpoint
            if os.path.isfile(checkpoint_path):
                print(f"=> Loading checkpoint '{checkpoint_path}'")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                state_dict = checkpoint.get('state_dict', checkpoint)
                # get base_encoder state_dict
                state_dict = {k.replace('module.base_encoder.', ''): v for k, v in state_dict.items() if 'module.base_encoder.' in k}
                msg = model.load_state_dict(state_dict, strict=False)
                print(f"=> Loaded checkpoint with missing keys: {msg.missing_keys}")
                assert all([linear_keyword in k for k in msg.missing_keys]), "Missing keys should be linear layers"
                model.eval()
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found!")
                raise FileNotFoundError

            # Extract features for each dataset
            for dataset_name, config in DATASETS.items():
                print(f"Processing {dataset_name}...")
                loader_cls = config["loader"]
                metric = config["metric"]
                use_split = config["split"]
                split_key = config.get("split_key", "train")

                if dataset_name == "Caltech101":
                    dataset = loader_cls(root=f'./data/{dataset_name.lower()}', download=True, transform=transform)
                    labels = dataset.y
                    train_indices, val_indices = train_test_split(
                        range(len(dataset)),
                        test_size=0.2,
                        stratify=labels,
                        random_state=42
                    )
                    train_dataset = Subset(dataset, train_indices)
                    test_dataset = Subset(dataset, val_indices)
                else:
                    if use_split:
                        train_dataset = loader_cls(root=f'./data/{dataset_name.lower()}', split=split_key, download=False, transform=transform)
                        test_dataset = loader_cls(root=f'./data/{dataset_name.lower()}', split='test', download=False, transform=transform)
                    else:
                        train_dataset = loader_cls(root=f'./data/{dataset_name.lower()}', train=True, download=False, transform=transform)
                        test_dataset = loader_cls(root=f'./data/{dataset_name.lower()}', train=False, download=False, transform=transform)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=False, num_workers=6)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=6)

                X_train, y_train = extract_features(train_loader, model, device)
                X_test, y_test = extract_features(test_loader, model, device)

                best_score = evaluate_with_gridsearch(X_train, y_train, X_test, y_test, metric)
                # Store the best score for this dataset
                if dataset_name not in folder_scores:
                    folder_scores[dataset_name] = []
                folder_scores[dataset_name].append(best_score)
            del model

        folder_scores["name"] = args.pretrained

    torch.save(folder_scores, f"results/transfer/{args.name}.pt")
