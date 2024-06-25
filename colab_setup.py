import os
import sys
import yaml
import subprocess
import torch

def setup_environment():
    # Clone the repository
    repo_url = "https://github.com/pylNeuralNet/2024-ml-project-resnet-transfer-learning"
    subprocess.run(["git", "clone", repo_url])
    
    # Change to the repository directory
    os.chdir("2024-ml-project-resnet-transfer-learning")
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "docs/requirements.txt"])
    
    # Download the dataset
    dataset_url = "https://drive.google.com/drive/folders/1N9D68Uj6Y3R8_iYAE_dnP9J5BXUiDXRy?u"
    subprocess.run(["wget", dataset_url])
    subprocess.run(["unzip", "dataset.zip", "-d", "data"])
    
    # Update config.yaml with the correct paths
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    config['data']['train_dir'] = '/content/2024-ml-project-resnet-transfer-learning/data/train'
    config['data']['val_dir'] = '/content/2024-ml-project-resnet-transfer-learning/data/val'
    config['data']['test_dir'] = '/content/2024-ml-project-resnet-transfer-learning/data/test'
    config['training']['checkpoint_dir'] = '/content/2024-ml-project-resnet-transfer-learning/checkpoints'
    config['logging']['log_dir'] = '/content/2024-ml-project-resnet-transfer-learning/logs'
    
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)

def run_training():
    subprocess.run([sys.executable, "run.py"])

if __name__ == "__main__":
    setup_environment()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available. Using GPU for training.")
    else:
        print("GPU is not available. Using CPU for training.")
    
    run_training()