# models/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

def compute_metrics(model, loader):
    model.eval()  # Pon el modelo en modo evaluación
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            targets = targets.squeeze().long()
            # Remapea las etiquetas: por ejemplo, 2 -> 0 y 4 -> 1
            targets = torch.where(targets == 2, torch.tensor(0, device=targets.device), torch.tensor(1, device=targets.device))
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return acc, prec, rec, f1

# Nota: No llames a compute_metrics aquí.
