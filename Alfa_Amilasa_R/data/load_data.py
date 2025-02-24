" Libraries to use "
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    " Load data from file_path "
    data= pd.read_csv(file_path)
    print(data.head(20))
    return data
    return pd.read_csv(file_path)

def process(data):
    num_rows = len(data.index)
    print('El total de datos es :',num_rows)
    input_cols = data.columns[0:1]
    output_col = [data.columns[1]]
    print('Las columnas de entrada son :',input_cols)
    print('La columna de salida es :',output_col)
    return input_cols, output_col