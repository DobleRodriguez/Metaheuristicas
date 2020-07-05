# Práctica 03.b
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


def simulated_annealing(data, ml_const, cl_const, ncluster, scale_factor, seed, solution=None, max_evals=100000, mu=0.3, fi=0.3, final_temp=0.001, adapt_factor=1):
    randgen = np.random.default_rng(seed)
    # Generamos solución inicial
    if (solution is None):
        current = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
        randgen.shuffle(current)
    else:
        current = solution
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
    max_neighbors = 10 * data.shape[0] / adapt_factor
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
    return best, best_gd, best_infeas, best_obj

def local_search(data, ml_const, cl_const, ncluster, scale_factor, seed, solution=None, max_evals=100000):
    randgen = np.random.default_rng(seed)
    if (solution is None):
        solution = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
        randgen.shuffle(solution)

    general_desviation, infeasibility, objective = evaluation(data, ml_const, cl_const, solution, ncluster, scale_factor)
    best = False
    evals = 1
    while (not best and evals < max_evals):
        best = True
        cluster_size = np.bincount(solution)
        single_elem_clusters = np.flatnonzero(cluster_size == 1)
        possible_elements = np.flatnonzero(np.isin(solution, single_elem_clusters, invert=True)) 
        neighbors = np.empty([0,2], int)
        for i in np.arange(ncluster):
            change_cluster = possible_elements[np.isin(possible_elements, np.flatnonzero(solution != i))]
            neighbors = np.append(neighbors, np.stack(np.meshgrid(change_cluster, i), -1).reshape(-1, 2), axis=0)

        randgen.shuffle(neighbors)

        for index, cluster in neighbors:
            neighbor = np.copy(solution)
            neighbor[index] = cluster
            neigh_gd, neigh_infeas, neigh_obj = evaluation(data, ml_const, cl_const, neighbor, ncluster, scale_factor)
            evals += 1
            if (neigh_obj < objective):
                solution = neighbor
                general_desviation = neigh_gd
                infeasibility = neigh_infeas
                objective = neigh_obj
                best = False
                break
    return solution, general_desviation, infeasibility, objective

def basic_multistart_search(data, ml_const, cl_const, ncluster, scale_factor, seed, nstarts=10, max_evals=10000):
    randgen = np.random.default_rng(seed)
    best_obj = np.inf
    for i in np.arange(nstarts):
        current = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
        randgen.shuffle(current)
        current, general_desviation, infeasibility, objective = local_search(data, ml_const, cl_const, ncluster, scale_factor, randgen.integers(np.iinfo(np.int32).max), current, max_evals)
        if (best_obj > objective):
            solution = current
            best_obj = objective
            best_gd = general_desviation
            best_infeas = infeasibility
    return solution, best_gd, best_infeas, best_obj 
  

def iterated_local_search(data, ml_const, cl_const, ncluster, scale_factor, seed, niters=10, max_evals=10000, hybrid_es=False, adapt_factor=1):
    randgen = np.random.default_rng(seed)
    segment_size = np.rint(data.shape[0] * 0.1)
    # Generamos solución inicial
    current = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
    randgen.shuffle(current)
    if (hybrid_es):
        current, general_desviation, infeasibility, objective = simulated_annealing(data, ml_const, cl_const, ncluster, scale_factor, randgen.integers(np.iinfo(np.int32).max), current, max_evals, adapt_factor=adapt_factor)
    else:
        current, general_desviation, infeasibility, objective = local_search(data, ml_const, cl_const, ncluster, scale_factor, randgen.integers(np.iinfo(np.int32).max), current, max_evals)
    
    for i in np.arange(niters):
        mutation = np.copy(current)
        mutation_obj = np.inf
        segment_start = randgen.integers(data.shape[0])
        segment = np.mod(np.arange(segment_start, segment_start+segment_size), data.shape[0])
        rest = np.isin(np.arange(data.shape[0]), segment, invert=True)
        missing_clusters = np.flatnonzero(np.isin(np.arange(ncluster), mutation[rest], invert=True))
        mutated_segment = np.append(missing_clusters, randgen.integers(ncluster, size=int(segment_size-missing_clusters.size)))
        randgen.shuffle(mutated_segment)
        mutation[segment.astype(int)] = mutated_segment

        if (hybrid_es):
            mutation, mutation_gd, mutation_infeas, mutation_obj = simulated_annealing(data, ml_const, cl_const, ncluster, scale_factor, randgen.integers(np.iinfo(np.int32).max), mutation, max_evals, adapt_factor=adapt_factor)
        else:
            mutation, mutation_gd, mutation_infeas, mutation_obj = local_search(data, ml_const, cl_const, ncluster, scale_factor, randgen.integers(np.iinfo(np.int32).max), mutation, max_evals)

        if (mutation_obj < objective):
            current = mutation
            objective = mutation_obj
            infeasibility = mutation_infeas
            general_desviation = mutation_gd

    return current, general_desviation, infeasibility, objective


#np.seterr(all='raise')
info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

sets = np.array(["iris", "rand", "ecoli", "newthyroid"])
nclusters = np.array([3, 3, 8, 3])
percents = np.array([10, 20])
seeds = np.array([1, 112, 241, 27, 472])

values = np.stack(np.meshgrid(percents, sets, seeds), -1).reshape(-1,3)
sets, set_repeats = np.unique(values[:,1], return_counts=True)

set_repeats = np.repeat(nclusters, set_repeats)
values = np.concatenate((values, np.array([set_repeats]).T), axis=-1)


# Funcionan: 1, 112, 241, 27, 472
info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

with open(pl.Path(__file__).parent / f"solutions_P03b.txt", 'w+') as sol_file:
    sol_file.write(
        f"{info_names[0]:>14} {info_names[1]:>10} {info_names[2]:>15} {info_names[3]:>7} {info_names[4]:>14} {info_names[5]:>20} {info_names[6]:>13} {info_names[7]:>20} {info_names[8]:>23}\n")

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = local_search(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    toc = time.perf_counter()
    func_name = "BL"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = simulated_annealing(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    toc = time.perf_counter()
    func_name = "ES"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = basic_multistart_search(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    toc = time.perf_counter()
    func_name = "BMB"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = iterated_local_search(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    toc = time.perf_counter()
    func_name = "ILS"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = iterated_local_search(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed), hybrid_es=True)
    toc = time.perf_counter()
    func_name = "ILS-ES"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    solution, general_desviation, infeasibility, objective = iterated_local_search(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed), hybrid_es=True, adapt_factor=10)
    toc = time.perf_counter()
    func_name = "ILS-ES A"
    with open(pl.Path(__file__).parent / f"solutions_P03b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

