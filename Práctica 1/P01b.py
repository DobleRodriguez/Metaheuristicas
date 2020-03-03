# Práctica 01.b
# Técnicas de Búsqueda Local y Algoritmos Greedy para el Problema del Agrupamiento con Restricciones
# Metaheurísticas (MH) - Grupo A1
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import numpy as np
import pandas as pd
import pathlib as pl

def read_data(set_name):
    data = pd.read_csv(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/{set_name}_set.dat", header=None)
    const_10 = pd.read_csv(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/{set_name}_set_const_10.const", header=None) 
    const_20 = pd.read_csv(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/{set_name}_set_const_20.const", header=None) 
    return data, const_10, const_20

def greedy_copkm(data, const, k, centroid):
    RSI = np.random.RandomState.shuffle(np.arange(len(data)))
    for i in RSI:
        pass

def local_search():
    pass

# Iris
data, const_10, const_20 = read_data("iris")

    
