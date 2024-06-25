# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from image_transforms import get_transforms
from imbalanced_sampling import BalancedDataset
import yaml

def get_data_loaders(train_dir, val_dir, test_dir, config):
    print("Preparing data transforms...")
    train_transform, test_transform = get_transforms(config)
    
    print(f"Loading training data from {train_dir}...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    print(f"Loading validation data from {val_dir}...")
    val_dataset = ImageFolder(root=val_dir, transform=test_transform)
    print(f"Loading test data from {test_dir}...")
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    
    print("Applying BalancedDataset to training data...")
    balanced_train_dataset = BalancedDataset(train_dataset, config)
    
    print("Creating data loaders...")
    train_loader = DataLoader(dataset=balanced_train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    print("Data loading completed.")
    print(f"Original train samples: {len(train_dataset)}, Balanced train samples: {len(balanced_train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader