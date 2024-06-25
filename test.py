# test.py
import torch
from metrics import calculate_metrics, log_metrics, plot_confusion_matrix
from model_analysis import save_misclassified_images, plot_roc_curve, plot_precision_recall_curve

def get_predictions(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.unsqueeze(1).float()
            outputs = model(images)
            predicted = outputs.round()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
    return all_labels, all_preds, all_scores

def evaluate_model(all_labels, all_preds, all_scores, config):
    accuracy, precision, recall, f1, auc_roc = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds), torch.tensor(all_scores))
    print(f'Test Accuracy: {accuracy*100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}')
    
    log_metrics(config['model']['name'], config, accuracy, precision, recall, f1, auc_roc, all_labels, all_preds)

    # Plot confusion matrix
    class_names = config['data']['class_names']
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=f"{config['logging']['log_dir']}/test_confusion_matrix.png")
    
    # Additional analysis
    # plot_roc_curve(all_labels, all_scores, save_path=f"{config['logging']['log_dir']}/test_roc_curve.png")
    # plot_precision_recall_curve(all_labels, all_scores, save_path=f"{config['logging']['log_dir']}/test_pr_curve.png")

def test_model(model, test_loader, config):
    all_labels, all_preds, all_scores = get_predictions(model, test_loader)
    evaluate_model(all_labels, all_preds, all_scores, config)
    
    # Save misclassified images
    # save_misclassified_images(model, test_loader, f"{config['logging']['log_dir']}/misclassified_images/")