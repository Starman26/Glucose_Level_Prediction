import torch.nn as nn 
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import numpy as np
from models.train import fit
from models.metrics import compute_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedClassificationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedClassificationModel, self).__init__()
        hidden_dim = 32  # Puedes ajustar este valor
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return self.predictor(x)
    
    def training_step(self, batch):
        inputs, targets = batch
        targets = targets.squeeze().long()
        # Remapea: si el target es 2, se convierte a 0; si es 4, se convierte a 1.
        targets = torch.where(targets == 2, 
                            torch.tensor(0, dtype=torch.long, device=targets.device), 
                            torch.tensor(1, dtype=torch.long, device=targets.device))
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        targets = targets.squeeze().long()
        targets = torch.where(targets == 2, 
                            torch.tensor(0, dtype=torch.long, device=targets.device), 
                            torch.tensor(1, dtype=torch.long, device=targets.device))
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        return {'val_loss': loss.detach()}

    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        if (epoch+1) % 100 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))


def get_model(input_size, output_size):
    return ImprovedClassificationModel(input_size, output_size)

def cross_validate_model(dataset, input_size, output_size, k=5, epochs=1000, lr=5e-4, batch_size=22):
    """
    Realiza k-fold cross validation sobre el dataset.
    Para cada fold:
      - Se divide el dataset en entrenamiento y validación.
      - Se crea una nueva instancia del modelo.
      - Se entrena el modelo y se calculan las métricas.
    Al final se muestran las métricas promedio.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold+1}/{k}')
        
        # Crear subconjuntos para entrenamiento y validación
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Crear DataLoaders para cada subconjunto
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Crear una nueva instancia del modelo para este fold
        model = get_model(input_size, output_size)
        
        # Entrenar el modelo en este fold
        history = fit(epochs, lr, model, train_loader, val_loader)
        
        # Calcular métricas en el conjunto de validación
        acc, prec, rec, f1 = compute_metrics(model, val_loader)
        print(f'Fold {fold+1} metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')
        
        fold_results[fold] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'val_loss': history[-1]['val_loss'] if history else None
        }
        
    # Calcular las métricas promedio a lo largo de todos los folds
    avg_accuracy = np.mean([fold_results[i]['accuracy'] for i in fold_results])
    avg_precision = np.mean([fold_results[i]['precision'] for i in fold_results])
    avg_recall = np.mean([fold_results[i]['recall'] for i in fold_results])
    avg_f1 = np.mean([fold_results[i]['f1'] for i in fold_results])
    
    print("\nResultados de la validación cruzada:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    
    return fold_results