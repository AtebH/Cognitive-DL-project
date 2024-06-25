# imbalanced_sampling.py
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset
import numpy as np

def balance_dataset_indices(y, config):
    indices = np.arange(len(y))
    
    if config['sampling']['oversampling']:
        oversampler = RandomOverSampler(sampling_strategy=config['sampling']['oversampling_ratio'])
        indices, _ = oversampler.fit_resample(indices.reshape(-1, 1), y)
    
    if config['sampling']['undersampling']:
        undersampler = RandomUnderSampler(sampling_strategy=config['sampling']['undersampling_ratio'])
        indices, _ = undersampler.fit_resample(indices.reshape(-1, 1), y)
    
    return indices.squeeze()

class BalancedDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        y = [item[1] for item in dataset]
        self.indices = balance_dataset_indices(y, config)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]