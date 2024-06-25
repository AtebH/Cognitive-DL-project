import torch
from data_loader import get_data_loaders
from train import train_model
from test import test_model
import yaml

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_loader, val_loader, test_loader = get_data_loaders(
        config['data']['train_dir'],
        config['data']['val_dir'],
        config['data']['test_dir'],
        config
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = train_model(train_loader, val_loader, config, device)
    test_model(models, test_loader, config, device)

if __name__ == '__main__':
    main()