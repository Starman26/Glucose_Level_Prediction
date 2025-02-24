from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd


def config_train (dataset,val_percent=0.1,batch_size=22):


    " Configure the train and validation sets "
    num_rows = len(dataset)
    val_size = int(num_rows * val_percent)

    
    train_size = num_rows - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader =DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size*2)

    return train_loader, val_loader

def train_model(inputs,targets,input_size,output_size,train_loader, val_loader):
    
    num_features = input_size
    num_clases= len(torch.unique(targets))

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearRegressionModel, self).__init__()
            hidden_dim = 32  # Puedes ajustar este valor según tu problema
            
            self.predictor = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_size)
            )

        def forward(self, xb):
            out = self.predictor(xb)
            return out

        def training_step(self, batch):
            inputs, targets = batch
            targets= targets.squeeze().long()
            targets = torch.where(targets == 2, torch.tensor(0, device=targets.device), torch.tensor(1, device=targets.device))

            # Generar predicciones
            out = self(inputs)
            # Calcula la perdida
            loss = F.cross_entropy(out,targets)
            return loss

        def validation_step(self, batch):
            inputs, targets = batch
            targets= targets.squeeze().long()
            targets = torch.where(targets == 2, torch.tensor(0, device=targets.device), torch.tensor(1, device=targets.device))

            # Genera predicciones
            out = self(inputs)
            # Calcula la perdida
            loss = F.cross_entropy(out, targets)
            return {'val_loss': loss.detach()}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combina los errores de ambos
            return {'val_loss': epoch_loss.item()}

        def epoch_end(self, epoch, result, num_epochs):
            # Imprime el resultado cada 1000 epocas
            if (epoch+1) % 100 == 0 or epoch == num_epochs-1:
                print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
    model = LinearRegressionModel(num_features, num_clases)
    return model 

def evaluate(model, val_loader):
    " Evaluate the model on the validation set "
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    " Train the model using gradient descent "
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    result = evaluate(model, val_loader)
    print(result)
    return history
