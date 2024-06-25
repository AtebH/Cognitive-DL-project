import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
import time
import numpy as np
from sklearn.metrics import accuracy_score

# TODO: consider creating another file for common helper functions
def to_tensor(X, device=None):
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(X, np.ndarray):
        tensor = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        tensor = X.float()
    else:
        raise TypeError(f"Unsupported data type: {type(X)}")
    
    if device is not None:
        tensor = tensor.to(device)
    return tensor

class ModelWrapper(BaseEstimator):
    def __init__(self, get_model_func, config, device, learning_rate=None, batch_size=None):
        self.get_model_func = get_model_func
        self.config = config
        self.learning_rate = learning_rate or config['training']['learning_rate']
        self.batch_size = batch_size or config['data']['batch_size']
        self.model = None
        self.device = device

    def set_params(self, **params):
        for param, value in params.items():
            if param == 'learning_rate':
                self.learning_rate = value or self.config['training']['learning_rate']
            elif param == 'batch_size':
                self.batch_size = value or self.config['data']['batch_size']
            else:
                setattr(self, param, value)
        return self

    def fit(self, X, y):
        print(f"Starting model fitting with learning_rate={self.learning_rate}, batch_size={self.batch_size}")
        
        X = to_tensor(X)
        y = to_tensor(y)

        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size or self.config['data']['batch_size'], 
            shuffle=True
        )

        self.model = self.get_model_func(self.config).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate or self.config['training']['learning_rate']
        )

        self.model.train()
        total_batches = len(train_loader)
        start_time = time.time()
        for _ in range(self.config['hyperparameter_tuning']['eval_epochs']):
            epoch_loss = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f"Batch {batch_idx + 1}/{total_batches} completed. Current loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / total_batches
            print(f"Epoch completed. Average loss: {avg_loss:.4f}") # TODO: thats some statistic about the run but im not sure if it is meaningful. Consider deletion
        
        training_time = time.time() - start_time
        print(f"Model fitting completed in {training_time:.2f} seconds")
        return self

    def predict(self, X):
        print("Starting prediction...")
        self.model.eval()
        with torch.no_grad():
            X = to_tensor(X)
            outputs = self.model(torch.Tensor(X))
            predictions = (outputs > 0.5).float().squeeze()
        print("Prediction completed")
        return predictions.cpu().numpy()
    
    def score(self, X, y):
        print("Calculating score...")
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Score calculation completed. Accuracy: {accuracy:.4f}")
        return accuracy # TODO: we have a config field for metric which is f1

def hyperparameter_tuning(get_model_func, param_grid, train_dataset, val_dataset, config):
    print("Creating X_train, y_Train, X_val, y_val from datasets...")
    X_train = [item[0].numpy() for item in train_dataset]
    y_train = [item[1] for item in train_dataset]
    X_val = [item[0].numpy() for item in val_dataset]
    y_val = [item[1] for item in val_dataset]

    print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper = ModelWrapper(get_model_func, config, device)

    search = RandomizedSearchCV(
        model_wrapper,
        param_distributions=param_grid,
        n_iter=min(config['hyperparameter_tuning']['n_iter'], len(param_grid['learning_rate']) * len(param_grid['batch_size'])),
        cv=config['hyperparameter_tuning']['cv'],
        random_state=42,
        scoring='accuracy',
        verbose=2
    )

    print("Starting RandomizedSearchCV...")
    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    print(f"RandomizedSearchCV completed in {tuning_time:.2f} seconds")
    
    best_model = search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return search.best_params_, best_model

# TODO: create a new config field "best_params" and save hyperparameter results there instead of training.learning_rate and data.batch_size