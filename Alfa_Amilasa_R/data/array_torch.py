import numpy as np
import torch 
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler,random_split
def to_arrays(input_cols, output_col, data):
    " Convert data to NumPy arrays "
    inputs_array = np.array(data[input_cols])
    targets_array = np.array(data[output_col])
    return inputs_array, targets_array

def to_tensors(inputs_array, targets_array):
    " Convert arrays to PyTorch tensors "
    inputs = torch.tensor(inputs_array, dtype=torch.float32)
    targets = torch.tensor(targets_array, dtype=torch.float32)
    inputs.shape, targets.shape
    return inputs, targets

def dataset_create(inputs, targets):
    "Crea un TensorDataset y configura un DataLoader con muestreo balanceado"
    dataset = TensorDataset(inputs, targets)
    
    # Extraer todas las etiquetas (targets) para calcular su distribución
    all_targets = [dataset[i][1].item() for i in range(len(dataset))]
    classes, counts = np.unique(all_targets, return_counts=True)
    print('Class Distribution:', dict(zip(classes, counts)))
    
    # Crear un diccionario que asigne a cada clase su peso inverso
    class_weight_dict = {cls: 1. / count for cls, count in zip(classes, counts)}
    
    # Asignar a cada muestra el peso correspondiente
    samples_weights = [class_weight_dict[t] for t in all_targets]
    samples_weights = torch.DoubleTensor(samples_weights)
    
    # Crear el sampler ponderado
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
    
    # Crear el DataLoader usando el sampler
    train_loader = DataLoader(dataset, batch_size=22, sampler=sampler)
    
    return dataset, train_loader