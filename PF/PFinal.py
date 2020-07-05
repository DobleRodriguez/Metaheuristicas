# Práctica Final    
# Metaheurística Original inspirada en la adaptación de estrategias en juegos competitivos.
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

def evaluation(solution, data, ml_const, cl_const, ncluster, scale_factor):
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

def refine(strat, score, max_variation, adopters, data, ml_const, cl_const, ncluster, scale_factor, randgen, max_evals=500):
    variation_size = randgen.integers(1, max_variation, adopters)
    #print(variation_size)
    adaptations = np.tile(strat, (adopters, 1))
    #input(adaptations.shape)
    scores = np.empty((0, 3))
    t_evals = 0
    #print(strat)
    for size, sol in zip(variation_size, adaptations):
        general_desviation = score[0]
        infeasibility = score[1]
        objective = score[2]
        best = False
        evals = 0
        variation = randgen.choice(strat.size, size=size, replace=False)
        #print(variation)
        while (not best and evals < max_evals):
            best = True
            cluster_size = np.bincount(sol)
            single_elem_clusters = np.flatnonzero(cluster_size == 1)
            possible_elements = np.intersect1d(variation, np.flatnonzero(np.isin(sol, single_elem_clusters, invert=True)))
            neighbors = np.empty([0,2], int)
            for i in np.arange(ncluster):
                change_cluster = possible_elements[np.isin(possible_elements, np.flatnonzero(sol != i))]
                neighbors = np.append(neighbors, np.stack(np.meshgrid(change_cluster, i), -1).reshape(-1, 2), axis=0)
            randgen.shuffle(neighbors)
            
            for index, cluster in neighbors:
                neighbor = np.copy(sol)
                neighbor[index] = cluster
                #print(f"{np.flatnonzero(sol != neighbor)}")
                neigh_gd, neigh_infeas, neigh_obj = evaluation(neighbor, data, ml_const, cl_const, ncluster, scale_factor)
                evals += 1
                if (neigh_obj < objective):
                    sol[index] = cluster
                    general_desviation = neigh_gd
                    infeasibility = neigh_infeas
                    objective = neigh_obj
                    best = False
                    break
                #else:
                    #print(f"{neigh_obj}   {objective}")
        #print(objective)
        #print(sol)
        #print("p")
        #input(f"{score[2]}  {objective}")
        scores = np.vstack((scores, np.array([general_desviation, infeasibility, objective])))
        t_evals += evals
    #input(adaptations.shape)
    #input(adaptations)
    #adaptations = np.unique(adaptations, axis=0)
    #input(adaptations.shape)
    
    return adaptations, scores, evals

def mutate(strat, max_variation, adopters, ncluster, randgen):
    variation_size = randgen.integers(1, max_variation, adopters)
    adaptations = np.tile(strat, (adopters, 1))
    for size, sol in zip(variation_size, adaptations):
        variation = randgen.choice(strat.size, size=size, replace=False)
        rest = np.isin(np.arange(strat.size), variation, invert=True)
        missing_clusters = np.flatnonzero(np.isin(np.arange(ncluster), strat[rest], invert=True))
        mutated_segment = np.append(missing_clusters, randgen.integers(ncluster, size=int(variation.size - missing_clusters.size)))
        randgen.shuffle(mutated_segment)
        sol[variation.astype(int)] = mutated_segment

    return adaptations

def basic_competitive_adaptation(data, ml_const, cl_const, ncluster, scale_factor, seed=None, tournament_size=50, max_evals=100000):
    randgen = np.random.default_rng(seed)
    strat_size = data.shape[0]
    podium_size = 3

    # Cuántos jugadores adoptan cada estrategia
    adopters = np.rint([tournament_size * 0.5 - 1, tournament_size * 0.3, tournament_size * 0.1])
    new_adopters = np.rint(tournament_size * 0.1)

    # Cuánto de una estrategia puede llegar a mutar
    max_variation = np.rint(strat_size * 0.5)

    # Crear población inicial:
    evals = 0
    insurance = np.tile(np.arange(ncluster), (tournament_size, 1))
    tournament = randgen.integers(ncluster, size=(tournament_size, strat_size-ncluster))
    tournament = np.concatenate((tournament, insurance), axis=1)
    randgen.shuffle(tournament, axis=1)    
    scores = np.apply_along_axis(evaluation, 1, tournament, data, ml_const, cl_const, ncluster, scale_factor)
    evals += tournament_size
    podium = np.argpartition(scores[:,2], np.arange(podium_size))[:podium_size]
    scores = scores[podium]
    podium = tournament[podium]
    while(evals < max_evals):
        # El primer lugar permanece
        tournament = np.copy(podium[0])
        # Igual a la construcción del torneo inicial, en forma de one-liner (y tamaño igual a cantidad de newcomers)
        newcomers = np.concatenate((randgen.integers(ncluster, size=(new_adopters.astype(int), strat_size-ncluster)), np.tile(np.arange(ncluster), (new_adopters.astype(int), 1))), axis=1)
        randgen.shuffle(newcomers, axis=1)
        tournament = np.vstack((tournament, newcomers))
        for i in np.arange(podium_size):
            tournament = np.vstack((tournament, mutate(podium[i], max_variation.astype(int), adopters[i].astype(int), ncluster, randgen)))
        #print(scores[0])
        scores = np.vstack((scores[0], np.apply_along_axis(evaluation, 1, tournament[1:], data, ml_const, cl_const, ncluster, scale_factor)))
        evals += tournament_size-1
        podium = np.argpartition(scores[:,2], np.arange(podium_size))[:podium_size]
        #print(scores[podium])
        #input(podium)
        scores = scores[podium]
        podium = tournament[podium]
        #print(scores)
        x, unique = np.unique(tournament, axis=0, return_counts=True)
        #print(unique.size)

    gd, infeas, objective = np.split(scores[0], podium_size)
    return podium[0], gd[0], infeas[0], objective[0]


def refined_competitive_adaptation(data, ml_const, cl_const, ncluster, scale_factor, seed=None, tournament_size=50, max_evals=100000):
    randgen = np.random.default_rng(seed)
    strat_size = data.shape[0]
    podium_size = 3

    # Cuántos jugadores adoptan cada estrategia
    adopters = np.rint([tournament_size * 0.5, tournament_size * 0.3, tournament_size * 0.1])
    new_adopters = np.rint(tournament_size * 0.1)

    # Cuánto de una estrategia puede llegar a mutar
    max_variation = np.rint(strat_size * 0.5)

    # Crear población inicial:
    evals = 0
    insurance = np.tile(np.arange(ncluster), (tournament_size, 1))
    tournament = randgen.integers(ncluster, size=(tournament_size, strat_size-ncluster))
    tournament = np.concatenate((tournament, insurance), axis=1)
    randgen.shuffle(tournament, axis=1)    
    scores = np.apply_along_axis(evaluation, 1, tournament, data, ml_const, cl_const, ncluster, scale_factor)
    evals += tournament_size
    podium = np.argpartition(scores[:,2], np.arange(podium_size))[:podium_size]
    scores = scores[podium]
    podium = tournament[podium]
    #print(podium)
    #input(scores)
    solved_meta = np.unique(podium, axis=0).size == 1
    while(evals < max_evals and not solved_meta):
        #print(np.sort(scores[:,2]))
        # El primer lugar permanece
        #tournament = np.copy(podium[0])
        # Igual a la construcción del torneo inicial, en forma de one-liner (y tamaño igual a cantidad de newcomers)
        podium_scores = np.copy(scores)
        tournament = np.concatenate((randgen.integers(ncluster, size=(new_adopters.astype(int), strat_size-ncluster)), np.tile(np.arange(ncluster), (new_adopters.astype(int), 1))), axis=1)
        randgen.shuffle(tournament, axis=1)
        scores = np.apply_along_axis(evaluation, 1, tournament, data, ml_const, cl_const, ncluster, scale_factor)
        evals += new_adopters
        #input(podium_scores)
        #scores = np.empty((0, 3))
        for i in np.arange(podium_size):
            t_results, s_results, e_results = refine(podium[i], podium_scores[i], max_variation.astype(int), adopters[i].astype(int), data, ml_const, cl_const, ncluster, scale_factor, randgen)
            tournament = np.vstack((tournament, t_results))
            scores = np.vstack((scores, s_results))
            evals += e_results
        #input(tournament)

        #print(scores)
        podium = np.argpartition(scores[:,2], np.arange(podium_size))[:podium_size]
        #print(scores[podium])
        #input(podium)
        scores = scores[podium]
        #input(scores[:,2])        
        podium = tournament[podium]
        #print(scores)
        solved_meta = np.unique(podium, axis=0).shape[0] == 1

    gd, infeas, objective = np.split(scores[0], podium_size)
    return podium[0], gd[0], infeas[0], objective[0]


info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

sets = np.array(["iris", "rand", "ecoli", "newthyroid"])
nclusters = np.array([3, 3, 8, 3])
percents = np.array([10, 20])
seeds = np.array([1, 112, 241, 27, 472])
#
#sets = np.array(["ecoli"])
#nclusters = np.array([8])
#percents = np.array([10])
#seeds = np.array([1])
#
values = np.stack(np.meshgrid(percents, sets, seeds), -1).reshape(-1,3)
sets, set_repeats = np.unique(values[:,1], return_counts=True)

set_repeats = np.repeat(nclusters, set_repeats)
values = np.concatenate((values, np.array([set_repeats]).T), axis=-1)


# Funcionan: 1, 112, 241, 27, 472
info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

with open(pl.Path(__file__).parent / f"solutions_PFinal.txt", 'w+') as sol_file:
    sol_file.write(
        f"{info_names[0]:>14} {info_names[1]:>10} {info_names[2]:>15} {info_names[3]:>7} {info_names[4]:>14} {info_names[5]:>20} {info_names[6]:>13} {info_names[7]:>20} {info_names[8]:>23}\n")

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    #solution, general_desviation, infeasibility, objective = competitive_adaptation(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))

    #solution, general_desviation, infeasibility, objective = basic_competitive_adaptation(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    #print(general_desviation, infeasibility, objective)
    #toc = time.perf_counter()
    #func_name = "BCA"
    #with open(pl.Path(__file__).parent / f"solutions_PFinal.txt",'a+') as sol_file:
    #    sol_file.write(
    #        f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
    #    )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    tic = time.perf_counter()
    #solution, general_desviation, infeasibility, objective = competitive_adaptation(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))

    solution, general_desviation, infeasibility, objective = refined_competitive_adaptation(data, ml_const, cl_const, int(ncluster), scale_factor, int(seed))
    #print(general_desviation, infeasibility, objective)
    toc = time.perf_counter()
    func_name = "RCA"
    with open(pl.Path(__file__).parent / f"solutions_PFinal.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )
