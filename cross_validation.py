# cross_validation.py
from sklearn.model_selection import KFold
import numpy as np
import torch

def k_fold_cross_validation(dataset, config):
    kf = KFold(n_splits=config['cross_validation']['k_folds'], shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'], 
            sampler=train_subsampler
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            sampler=val_subsampler
        )
        
        yield fold, train_loader, val_loader