# metrics.py
import torch
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

# TODO: consider doing a common helper file with ensure_tensor and ensure_numpy
def ensure_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def calculate_metrics(y_true, y_pred, y_scores):
    y_true_np = ensure_numpy(y_true)
    y_pred_np = ensure_numpy(y_pred)
    y_scores_np = ensure_numpy(y_scores)
    
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np)
    recall = recall_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    auc_roc = roc_auc_score(y_true_np, y_scores_np)
    
    return accuracy, precision, recall, f1, auc_roc

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
    plt.close()  # Close the figure to free up memory

def log_metrics(model_name, config, accuracy, precision, recall, f1, auc_roc, y_true, y_pred, epoch=None):
    logging.basicConfig(filename=config['logging']['log_file'], level=logging.INFO)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Config: {config}")
    logging.info(f"Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, AUC-ROC: {auc_roc}")
    
    # Create the log directory if it doesn't exist
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Plot and save confusion matrix
    class_names = config['data']['class_names']
    if epoch is not None:
        filename = f"confusion_matrix_epoch_{epoch}_{timestamp}.png"
    else:
        filename = f"confusion_matrix_{timestamp}.png"
    save_path = os.path.join(config['logging']['log_dir'], filename)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path)