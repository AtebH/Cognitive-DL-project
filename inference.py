import yaml
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os
from image_transforms import get_transforms
from metrics import calculate_metrics, plot_confusion_matrix, log_metrics

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def preprocess_image(image, config):
    _, test_transform = get_transforms(config)
    image = test_transform(image)
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def load_models(config, checkpoint_path):
    n_bootstrap_models = config['training']['n_bootstrap_models']
    models = [get_model(config) for _ in range(n_bootstrap_models)]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for i, model in enumerate(models):
        model.load_state_dict(checkpoint[f'model_{i}_state_dict'])
        model.eval()
    return models

def get_prediction(models, image):
    with torch.no_grad():
        outputs = torch.mean(torch.stack([model(image) for model in models]), dim=0)
        probability = torch.sigmoid(outputs).item()
        predicted_class = 1 if probability > 0.5 else 0
    return predicted_class, probability

def infer(image_path, config, checkpoint_path):
    image = load_image(image_path)
    image = preprocess_image(image, config)
    models = load_models(config, checkpoint_path)
    return get_prediction(models, image)

def batch_inference(image_dir, config, checkpoint_path):
    models = load_models(config, checkpoint_path)
    all_true_labels, all_predicted_labels, all_probabilities = [], [], []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        true_label = 1 if 'positive' in image_name.lower() else 0
        
        image = preprocess_image(load_image(image_path), config)
        predicted_class, probability = get_prediction(models, image)
        
        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_class)
        all_probabilities.append(probability)

    return all_true_labels, all_predicted_labels, all_probabilities

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    checkpoint_path = config['inference']['checkpoint_path']
    
    # Single image inference
    image_path = config['inference']['image_path']
    predicted_class, probability = infer(image_path, config, checkpoint_path)
    print(f'Single Image - Predicted class: {config["data"]["class_names"][predicted_class]} with probability: {probability:.2f}')

    # Batch inference
    image_dir = config['inference']['image_dir']
    y_true, y_pred, y_scores = batch_inference(image_dir, config, checkpoint_path)
    
    accuracy, precision, recall, f1, auc_roc = calculate_metrics(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(y_scores))
    log_metrics('Ensemble Model', config, accuracy, precision, recall, f1, auc_roc, y_true, y_pred)
    
    plot_confusion_matrix(y_true, y_pred, config['data']['class_names'], 
                          save_path=f"{config['logging']['log_dir']}/confusion_matrix_inference.png")

    print('Batch Inference Metrics:')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')