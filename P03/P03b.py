# Práctica 02.b
# Técnicas de Búsqueda basadas en Poblaciones para el Problema del Agrupamiento con Restricciones
# Metaheurísticas (MH) - Grupo A1
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import pathlib as pl
import time

import numpy as np

def init_data(set_name, const_percent):
    data = np.loadtxt(pl.Path(__file__).parent /
    f"Instancias y Tablas PAR 2019-20/{set_name}_set.dat", delimiter=',')
    const = np.loadtxt(pl.Path(__file__).parent /
    f"Instancias y Tablas PAR 2019-20/{set_name}_set_const_{const_percent}.const", delimiter=',')
    data_distances = np.empty(0)
    for i in np.nditer(np.arange(len(data))):
        data_distances = np.append(data_distances, np.linalg.norm(data[i] - data[i+1:], axis=1))
    # Techo de la distancia máxima (lambda) entre cantidad de restricciones (solo consideramos
    # la diagonal superior y sin contar la diagonal)
    scale_factor = np.ceil(np.amax(data_distances)) / np.count_nonzero(np.triu(const, 1))

    ml_const = np.argwhere(np.tril(const, -1) == 1)
    cl_const = np.argwhere(np.tril(const, -1) == -1)
    return data, ml_const, cl_const, scale_factor

def evaluation(data, ml_const, cl_const, solution, ncluster, scale_factor):
    centroids = np.zeros([ncluster, data.shape[1]])

    for i in np.arange(ncluster):
        centroids[i] = np.mean(data[solution == i], axis=0)

    general_desviation = 0
    for i in np.arange(ncluster):
        general_desviation += np.mean(np.linalg.norm(centroids[i] - data[solution == i], axis=1))/ncluster

    infeasibility = np.count_nonzero(solution[ml_const[:,0]] != solution[ml_const[:,1]]) + \
        np.count_nonzero(solution[cl_const[:,0]] == solution[cl_const[:,1]])

    objective = general_desviation + (infeasibility * scale_factor)
    
    return general_desviation, infeasibility, objective


def simulated_annealing(data, ml_const, cl_const, ncluster, scale_factor, seed, max_evals=100000, mu=0.3, fi=0.3, final_temp=0.001):
    randgen = np.random.default_rng(seed)
    # Generamos solución inicial
    current = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
    randgen.shuffle(current)
    general_desviation, infeasibility, objective = evaluation(data, ml_const, cl_const, current, ncluster, scale_factor)
    evals = 1

    best = current
    best_gd = general_desviation
    best_infeas = infeasibility
    best_obj = objective

    # Temperatura inicial
    initial_temp = mu * objective / -np.log(fi)
    # Si llegara a ser menor que la final, reducimos la final hasta que sea menor
    while (initial_temp <= final_temp):
        final_temp /= 10
    
    # Establecemos valores de cota
    max_neighbors = 10 * data.size
    max_successes = 0.1 * max_neighbors
    max_iters = max_evals / max_neighbors
    beta = (initial_temp - final_temp) / (max_iters * initial_temp * final_temp)

    # Asignamos valor de temperatura
    temp = initial_temp

    # Accepted vale 1 para que entre en el bucle externo
    accepted = 1
    while (temp > final_temp and accepted > 0 and evals < max_evals):
        accepted = 0
        explored = 0

        # Bucle interno
        while (explored < max_neighbors and accepted < max_successes and evals < max_evals):
            # Cuántos elementos tiene cada cluster en la solución (>= 1 para todo cluster)
            cluster_size = np.bincount(current)

            # Aislamos los clusteres con un solo elemento
            single_elem_clusters = np.flatnonzero(cluster_size == 1)

            # Escogemos uno (el que cambiaremos)
            changed = randgen.choice(np.flatnonzero(np.isin(current, single_elem_clusters, invert=True)))

            # y el valor al que cambiará
            change_to = randgen.choice(np.flatnonzero(np.isin(np.arange(ncluster), current[changed], invert=True)))
            # Generamos el vecino
            neighbor = np.copy(current)
            neighbor[changed] = change_to
            explored += 1

            
            neigh_gd, neigh_infeas, neigh_obj = evaluation(data, ml_const, cl_const, neighbor, ncluster, scale_factor)
            evals += 1
            delta = neigh_obj - objective
            if (delta < 0 or randgen.random() <= np.exp(-delta/temp)):
                accepted += 1
                current = neighbor
                general_desviation = neigh_gd
                infeasibility = neigh_infeas
                objective = neigh_obj
                if (objective < best_obj):
                    best = current
                    best_gd = general_desviation
                    best_infeas = infeasibility
                    best_obj = objective
        temp = temp / (1 + beta*temp)
    return best_gd, best_infeas, best_obj

def basic_multistart_search(data, ml_const, cl_const, ncluster, scale_factor, seed, nstarts=10, max_evals=10000):
    randgen = np.random.default_rng(seed)
    best_obj = np.inf
    for i in np.arange(nstarts):
        current = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
        general_desviation, infeasibility, objective = evaluation(data, ml_const, cl_const, current, ncluster, scale_factor)
        evals = 1
        best = False
        while (not best and evals < max_evals):
            best = True
            cluster_size = np.bincount(current)
            single_elem_clusters = np.flatnonzero(cluster_size == 1)
            possible_elements = np.flatnonzero(np.isin(current, single_elem_clusters, invert=True)) 
            neighbors = np.empty([0,2], int)
            for i in np.arange(ncluster):
                change_cluster = possible_elements[np.isin(possible_elements, np.flatnonzero(current != i))]
                neighbors = np.append(neighbors, np.stack(np.meshgrid(change_cluster, i), -1).reshape(-1, 2), axis=0)

            randgen.shuffle(neighbors)

            for index, cluster in neighbors:
                neighbor = np.copy(current)
                neighbor[index] = cluster
                neigh_gd, neigh_infeas, neigh_obj = evaluation(data, ml_const, cl_const, neighbor, ncluster, scale_factor)
                evals += 1
                if (neigh_obj < objective):
                    current = neighbor
                    general_desviation = neigh_gd
                    infeasibility = neigh_infeas
                    objective = neigh_obj
                    best = False
                    break
        if (best_obj > objective):
            solution = current
            best_obj = objective
            best_gd = general_desviation
            best_infeas = infeasibility
    return best_gd, best_infeas, best_obj 

    solutions = np.tile(np.arange(ncluster), (nstarts, 1))        

def iterated_local_search(data, ml_const, cl_const, ncluster, scale_factor, seed, niters=10, max_iters=10000):
    pass

#np.seterr(all='raise')
info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

data, ml_const, cl_const, scale_factor = init_data("ecoli", 20)
gd, infeas, obj = basic_multistart_search(data, ml_const, cl_const, 8, scale_factor, 1)
print(gd, infeas, obj)
