# ensemble.py
import torch

def train_ensemble(models, train_loader, criterion, optimizers, device):
    for model, optimizer in zip(models, optimizers):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def validate_ensemble(models, val_loader, device):
    for model in models:
        model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = torch.mean(torch.stack([model(images) for model in models]), dim=0)
            predicted = outputs.round()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
    return all_labels, all_preds, all_scores
