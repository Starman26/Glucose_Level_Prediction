import torch
import numpy as np
from data.load_data import load_data, process
from data.visualization import plot, plot_lost
from data.array_torch import to_arrays, to_tensors, dataset_create
from models.train import config_train, train_model, fit
from models.metrics import compute_metrics
from torch.utils.data import random_split, DataLoader
from models.second_model import cross_validate_model
from sklearn.model_selection import KFold


def run(file_path):
    "Run the pipeline"
    # Cargar datos
    data = load_data(file_path)

    # Visualizar propiedades de los datos
    plot(data)

    # Procesar datos para separar columnas de entrada y salida
    input_cols, output_col = process(data)
    input_size = len(input_cols)
    #output_size = len(output_col)
    output_size=2

    # Convertir los datos a arrays y luego a tensores
    inputs_array, targets_array = to_arrays(input_cols, output_col, data)
    inputs, targets = to_tensors(inputs_array, targets_array)    

    # Crear el dataset con los tensores
    dataset,train_loader = dataset_create(inputs, targets)

    # Configurar los conjuntos de entrenamiento y validación
    #train_loader, val_loader = config_train(dataset, val_percent=0.2, batch_size=22)

    #"train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])"

    #"val_loader = DataLoader(val_dataset, batch_size=22, shuffle=True)"

    # Crear el modelo
    #"model = train_model(inputs, targets, input_size, output_size, train_loader, val_loader)"

    # Entrenar el modelo
    #"epochs = 1000"
    #"lr = 5e-4"
    #"history = fit(epochs, lr, model, train_loader, val_loader)"

    # Guardar el modelo entrenado
    #"torch.save(model.state_dict(), best_model.pth)"

    # Graficar la evolución de la pérdida
   #plot_lost(history)

    # Calcular las métricas de evaluación
    #"acc, prec, rec, f1 = compute_metrics(model, val_loader)"
    #"print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')"

    fold_results = cross_validate_model(dataset, input_size, output_size, k=5, epochs=1000, lr=5e-4, batch_size=22)
    
    return None


    return None

if __name__ == "__main__":
    run(r"C:\Users\lench\Desktop\Alfa_Amilasa_R\Real_Cleaned_Data.csv")
