import torch
from torch import nn, optim
import yaml
import os
from metrics import calculate_metrics, log_metrics, plot_confusion_matrix
from cross_validation import k_fold_cross_validation
from hyperparameter_tuning import hyperparameter_tuning
from model import get_model
from ensemble import train_ensemble, validate_ensemble
import time

def save_checkpoints(models, epoch, fold, config, performance):
    checkpoint_dir = config['training']['checkpoint_dir']
    n_bootstrap_models = len(models)
    learning_rate = config['training']['learning_rate']
    batch_size = config['data']['batch_size']
    
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'config': config,
        'performance': performance,
        'n_bootstrap_models': n_bootstrap_models
    }
    
    for i, model in enumerate(models):
        checkpoint[f'model_{i}_state_dict'] = model.state_dict()
    
    checkpoint_name = f'checkpoint_n{n_bootstrap_models}_lr{learning_rate}_bs{batch_size}_fold{fold}_epoch{epoch + 1}_perf{performance:.4f}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_name}')
    return checkpoint_path

def load_checkpoints(checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    n_bootstrap_models = checkpoint['n_bootstrap_models']
    models = [get_model(config) for _ in range(n_bootstrap_models)]
    
    for i, model in enumerate(models):
        model.load_state_dict(checkpoint[f'model_{i}_state_dict'])
        model.eval()
    
    return models, checkpoint['epoch'], checkpoint['fold'], checkpoint['performance']

def perform_hyperparameter_tuning(train_loader, val_loader, config):
    print("Starting hyperparameter tuning...")
    start_time = time.time()
    
    best_params, best_model = hyperparameter_tuning(get_model, config['hyperparameter_tuning']['param_grid'], 
                                        train_loader.dataset, val_loader.dataset, config)
    
    # Update config with best hyperparameters
    config['training'].update(best_params)
    config['model'].update(best_params)
    
    tuning_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best hyperparameters: {best_params}")
    return config, best_model

def initialize_models_and_optimizers(config, device):
    n_bootstrap_models = config['training']['n_bootstrap_models']
    models = [get_model(config).to(device) for _ in range(n_bootstrap_models)]
    criterion = nn.BCELoss()
    optimizers = [optim.Adam(model.parameters(), lr=config['training']['learning_rate']) for model in models]
    return models, criterion, optimizers

def select_best_model(config):
    checkpoint_dir = config['training']['checkpoint_dir']
    best_performance = float('-inf')
    best_checkpoint_path = None

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            performance = float(filename.split('_perf')[-1].split('.pth')[0])
            if performance > best_performance:
                best_performance = performance
                best_checkpoint_path = os.path.join(checkpoint_dir, filename)

    if best_checkpoint_path is None:
        raise ValueError("No checkpoints found")

    return load_checkpoints(best_checkpoint_path, config)

def train_and_validate_fold(models, criterion, optimizers, fold_train_loader, fold_val_loader, fold, config, device):
    for epoch in range(config['training']['epochs']):
        train_ensemble(models, fold_train_loader, criterion, optimizers, device)
        all_labels, all_preds, all_scores = validate_ensemble(models, fold_val_loader, device)
        
        metrics = log_epoch_results(epoch, all_labels, all_preds, all_scores, config)
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoints(models, epoch, fold, config, metrics[config['training']['optimization_metric']])

    return metrics

def log_epoch_results(epoch, all_labels, all_preds, all_scores, config):
    accuracy, precision, recall, f1, auc_roc = calculate_metrics(all_labels, all_preds, all_scores)
    print(f'Epoch [{epoch + 1}/{config["training"]["epochs"]}], '
          f'Accuracy: {accuracy*100:.2f}%, Precision: {precision:.2f}, '
          f'Recall: {recall:.2f}, F1-Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}')
    
    log_metrics(config['model']['name'], config, accuracy, precision, recall, f1, auc_roc, all_labels, all_preds, epoch=epoch + 1)
    
    # TODO: see if below is obsolete and can be deleted
    # plot_confusion_matrix(all_labels, all_preds, config['data']['class_names'], 
    #                       save_path=f"{config['logging']['log_dir']}/confusion_matrix_epoch_{epoch + 1}.png")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }  # TODO: decide which metric to optimize the performance with

def train_model(train_loader, val_loader, config, device):
    print("Starting model training...")

    config, _ = perform_hyperparameter_tuning(train_loader, val_loader, config)
    models, criterion, optimizers = initialize_models_and_optimizers(config, device)

    for fold, fold_train_loader, fold_val_loader in k_fold_cross_validation(train_loader.dataset, config):
        print(f"Training fold {fold + 1}")
        train_and_validate_fold(models, criterion, optimizers, fold_train_loader, fold_val_loader, fold, config, device)

    print("Model training completed.")
    return select_best_model(config) # TODO: consider: return models vs return select_best_model(config)
# TODO: when it is done with lr and bs given in param grid, it still proceeds to default values from config, the log says Starting model fitting with learning_rate=None, batch_size=None