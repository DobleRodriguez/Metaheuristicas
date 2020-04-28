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

##############################################################################################
# ALGORITMOS GENÉTICOS (AG)

def evaluation(data, ml_const, cl_const, population, ncluster, scale_factor):
    centroids = np.zeros([population.shape[0], ncluster, data.shape[1]])
    general_desviation = np.zeros(population.shape[0])
    infeasibility = np.empty(population.shape[0])
    objective = np.empty(population.shape[0])

    for i in np.arange(population.shape[0]):

        for j in np.arange(ncluster):
            centroids[i,j] = np.mean(data[population[i] == j], axis=0)

        for j in np.arange(ncluster):
            general_desviation[i] += np.mean(np.linalg.norm(centroids[i,j] - data[population[i] == j], axis=1))/ncluster

        infeasibility[i] = np.count_nonzero(population[i, ml_const[:,0]] != population[i, ml_const[:,1]]) + \
            np.count_nonzero(population[i, cl_const[:,0]] == population[i, cl_const[:,1]])

        objective[i] = general_desviation[i] + (infeasibility[i] * scale_factor)
    
    return general_desviation, infeasibility, objective

# ESTACIONARIO 
def stationary_genetic(data, ml_const, cl_const,ncluster, scale_factor, randgen, uniform=True, max_evals=100000, 
        population_size=50, mutation_odds=0.001):

    # Generación de la población
    # Matriz de tpoblacion x tdato (tcromosoma)
    evals = 0
    population = np.empty([population_size, data.shape[0]])
    population = randgen.integers(ncluster, size=population.shape)
    for i in np.arange(population_size):
        ordering = np.bincount(population[i], minlength=ncluster)
        check = np.flatnonzero(ordering == 0)
        if check.size > 0:
            single_elem_clusters = np.flatnonzero(ordering == 1)
            changed = randgen.choice(np.flatnonzero(np.isin(population[i], single_elem_clusters, invert=True)), check.size, replace=False)
            population[i, changed] = check
            

    general_desviation, infeasibility, objective = evaluation(data, ml_const, cl_const, population, ncluster, scale_factor)
    evals += population_size

    chromo_size = population.shape[1]
    # Mutación a nivel de cromosoma
    mutations = mutation_odds * chromo_size
    while evals < max_evals:
        #print(np.amin(infeasibility))
        #print(evals)
        # TORNEO
        # Utilizamos este mecanismo para permitir que un mismo dato participe en más de un torneo
        # pero nunca compita contra sí mismo
        parents = np.empty(2)
        for j in np.arange(2):
            tournament = randgen.choice(population_size, 2, replace=False)
            parents[j] = tournament[np.argmin(objective[tournament])]
        #print(np.sort(objective))

        # CRUCE
        # Hacemos los cruces
        genex = np.copy(population[parents.astype(int)])

        for j in np.arange(2):
            segment_size = randgen.integers(chromo_size) * int(not uniform)
            segment_start = randgen.integers(chromo_size)
            # Copiar segmento
            # AQUí CAMBIA SEGÚN SI ES UNIFORME O SEGMENTO FIJO
            # Selección de la mitad de índices de elementos del cromosoma NO PERTENECIENTES al segmento fijo, aleatorios sin reemplazo. 

            segment = np.mod(np.arange(segment_start, segment_start+segment_size), chromo_size)
            valid_points = np.flatnonzero(np.isin(np.arange(chromo_size), segment, invert=True))
            crossing_points = randgen.choice(valid_points, int(np.rint(valid_points.size/2)), replace=False)

            # Creación del hijo donde nos quedamos con el gen de cada padre respectivamente según si está o no en los puntos de cruce
            genex[j, crossing_points] = population[parents[int(not j)].astype(int), crossing_points]

            # REPARAR HIJOS
            check = np.isin(np.arange(ncluster), genex[j].astype(int), invert=True)
            empty_clusters = np.flatnonzero(check)
            if (empty_clusters.size > 0):
                cluster_amount = np.bincount(genex[j].astype(int), minlength=ncluster)
                single_elem_clusters = np.flatnonzero(cluster_amount == 1)
                # No quiero dejar otro cluster vacío
                changed = randgen.choice(np.flatnonzero(np.isin(genex[j], single_elem_clusters, invert=True)), empty_clusters.size, replace=False)
                genex[j, changed] = empty_clusters

        # MUTACIÓN
        dicerolls = randgen.random(2)
        #print(dicerolls)
        #print(mutations)
        mutated = np.flatnonzero(dicerolls < mutations)
        #input(mutated)
        for j in mutated:
            #input("Muta")
            cluster_amount = np.bincount(genex[int(j)].astype(int), minlength=ncluster)
            single_elem_clusters = np.flatnonzero(cluster_amount == 1)
            possible_elements = np.flatnonzero(np.isin(genex[int(j)], single_elem_clusters, invert=True)) 
            gen = randgen.choice(possible_elements)
            possible_mutations = np.flatnonzero(np.isin(np.arange(ncluster), genex[int(j), int(gen)] ,invert=True))
            genex[int(j), int(gen)] = randgen.choice(possible_mutations)


        # COMPETENCIA HIJOS
        #print(genex)
        new_gd, new_infeas, new_objective = evaluation(data, ml_const, cl_const, genex, ncluster, scale_factor)
        #print(new_objective)
        evals += 2
        #print(objective)
        weakest = np.argpartition(objective, -2)[-2:]
        #print(objective[weakest])
        #print(weakest)
        contestants = np.concatenate((genex, population[weakest]))
        m_obj = np.append(new_objective, objective[weakest])
        m_infeas = np.append(new_infeas, infeasibility[weakest])
        m_gd = np.append(new_gd, general_desviation[weakest])

        #print(m_obj)
        #print(m_infeas)
        #print(m_gd)
        winners = np.argpartition(m_obj, 2)[:2]
        #print(genex)
        #print()
        #print(contestants)
        #print()
        #print(population[weakest])
        #print("\n")
        #print(winners)
        #print(weakest)
        #input(np.argpartition(m_obj, 2))
        #print(objective[weakest])
        #print(m_obj)
        population[weakest] = contestants[winners]
        infeasibility[weakest] = m_infeas[winners]
        general_desviation[weakest] = m_gd[winners]
        objective[weakest] = m_obj[winners]
        #print(objective[weakest])
        #input()
    
    best_solution = np.argmin(objective)
    return general_desviation[best_solution], infeasibility[best_solution], objective[best_solution]


# GENERACIONAL CON ELITISMO 

def generational_genetic(data, ml_const, cl_const,ncluster, scale_factor, randgen, uniform=True, max_evals=100000, 
        population_size=50, crossover_odds=0.7, mutation_odds=0.001):

    # Generación de la población
    # Matriz de tpoblacion x tdato (tcromosoma)
    evals = 0
    population = np.empty([population_size, data.shape[0]])
    population = randgen.integers(ncluster, size=population.shape)
    for i in np.arange(population_size):
        ordering = np.bincount(population[i], minlength=ncluster)
        check = np.flatnonzero(ordering == 0)
        if check.size > 0:
            single_elem_clusters = np.flatnonzero(ordering == 1)
            changed = randgen.choice(np.flatnonzero(np.isin(population[i], single_elem_clusters, invert=True)), check.size, replace=False)
            population[i, changed] = check
            

    general_desviation, infeasibility, objective = evaluation(data, ml_const, cl_const, population, ncluster, scale_factor)
    evals += population_size

    crossovers = np.rint(crossover_odds * population_size/2)
    mutations = np.rint(mutation_odds * population_size * population.shape[1])
    while evals < max_evals:
        #print(evals)
        # TORNEO
        # Utilizamos este mecanismo para permitir que un mismo dato participe en más de un torneo
        # pero nunca compita contra sí mismo
        parents = np.empty(population_size)
        for j in np.arange(population_size):
            tournament = randgen.choice(population_size, 2, replace=False)
            parents[j] = tournament[np.argmin(objective[tournament])]

        # CRUCE
        genex = np.copy(population[parents.astype(int)])
        # Hacemos los cruces
        chromo_size = population.shape[1]

        for j in np.arange(crossovers):
            for k in np.arange(2):
                segment_size = randgen.integers(chromo_size) * int(not uniform)
                segment_start = randgen.integers(chromo_size)
                # Copiar segmento
                # AQUí CAMBIA SEGÚN SI ES UNIFORME O SEGMENTO FIJO
                # Selección de la mitad de índices de elementos del cromosoma NO PERTENECIENTES al segmento fijo, aleatorios sin reemplazo. 

                segment = np.mod(np.arange(segment_start, segment_start+segment_size), chromo_size)
                valid_points = np.flatnonzero(np.isin(np.arange(chromo_size), segment, invert=True))
                crossing_points = randgen.choice(valid_points, int(np.rint(valid_points.size/2)), replace=False)

                # Creación del hijo donde nos quedamos con el gen de cada padre respectivamente según si está o no en los puntos de cruce
                genex[int(2*j + k), crossing_points] = population[parents[int(2*j + (not k))].astype(int), crossing_points]

                # REPARAR HIJOS
                check = np.isin(np.arange(ncluster), genex[int(2*j + k)].astype(int), invert=True)
                empty_clusters = np.flatnonzero(check)
                if (empty_clusters.size > 0):
                    cluster_amount = np.bincount(genex[int(2*j + k)].astype(int), minlength=ncluster)
                    single_elem_clusters = np.flatnonzero(cluster_amount == 1)
                    # No quiero dejar otro cluster vacío
                    changed = randgen.choice(np.flatnonzero(np.isin(genex[int(2*j + k)], single_elem_clusters, invert=True)), empty_clusters.size, replace=False)
                    genex[int(2*j + k), changed] = empty_clusters

        # MUTACIÓN
        mutated = randgen.choice(population_size, size=int(mutations))
        for j in mutated:
            cluster_amount = np.bincount(genex[int(j)].astype(int), minlength=ncluster)
            single_elem_clusters = np.flatnonzero(cluster_amount == 1)
            possible_elements = np.flatnonzero(np.isin(genex[int(j)], single_elem_clusters, invert=True)) 
            gen = randgen.choice(possible_elements)
            prev = np.copy(genex[int(j), int(gen)])
            possible_mutations = np.flatnonzero(np.isin(np.arange(ncluster), genex[int(j), int(gen)] ,invert=True))
            genex[int(j), int(gen)] = randgen.choice(possible_mutations)
            if (prev == genex[int(j), int(gen)]):
                print("Fuck")


        # ELITISMO
        new_gd, new_infeas, new_objective = evaluation(data, ml_const, cl_const, genex, ncluster, scale_factor)
        evals += population_size
        champion = np.argmin(objective)
        if (not np.any(np.equal(population[champion], genex).all(axis=1))):
            weakest = np.argmax(new_objective)
            genex[weakest] = population[champion]
            new_gd[weakest] = general_desviation[champion]
            new_infeas[weakest] = infeasibility[champion]
            new_objective[weakest] = objective[champion]
        population = genex
        general_desviation = new_gd
        infeasibility = new_infeas
        objective = new_objective
    
    best_solution = np.argmin(objective)
    return general_desviation[best_solution], infeasibility[best_solution], objective[best_solution]


#################################################################################################




#####################################################################################################
np.seterr(all='raise')
info_names = ["Algoritmo", "Dataset", "% Restricciones", "Semilla", "N° clústeres", "Desviación general", 
    "Infeasibility", "Función objetivo", "Tiempo de ejecución (s)"]

data, ml_const, cl_const, scale_factor = init_data("iris", 10)
stationary_genetic(data, ml_const, cl_const, 3, scale_factor, np.random.default_rng(1))

sets = np.array(["iris", "rand", "ecoli", "newthyroid"])
nclusters = np.array([3, 3, 8, 3])
percents = np.array([10, 20])
seeds = np.array([1, 112, 241, 27, 472])

values = np.stack(np.meshgrid(percents, sets, seeds), -1).reshape(-1,3)
sets, set_repeats = np.unique(values[:,1], return_counts=True)

set_repeats = np.repeat(nclusters, set_repeats)
values = np.concatenate((values, np.array([set_repeats]).T), axis=-1)


with open(pl.Path(__file__).parent / f"solutions_P02b.txt", 'w+') as sol_file:
    sol_file.write(
        f"{info_names[0]:>14} {info_names[1]:>10} {info_names[2]:>15} {info_names[3]:>7} {info_names[4]:>14} {info_names[5]:>20} {info_names[6]:>13} {info_names[7]:>20} {info_names[8]:>23}\n"
    )

for percent,dataset,seed,ncluster in values:
    data, ml_const, cl_const, scale_factor = init_data(dataset, percent)
    randgen = np.random.default_rng(int(seed))
    tic = time.perf_counter()
    general_desviation, infeasibility, objective = stationary_genetic(data, ml_const, cl_const, int(ncluster), scale_factor, randgen, uniform=True)
    toc = time.perf_counter()
    func_name = "AGE-UN"
    with open(pl.Path(__file__).parent / f"solutions_P02b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )

    tic = time.perf_counter()
    general_desviation, infeasibility, objective = stationary_genetic(data, ml_const, cl_const, int(ncluster), scale_factor, randgen, uniform=False)
    toc = time.perf_counter()
    func_name = "AGE-SF"
    with open(pl.Path(__file__).parent / f"solutions_P02b.txt",'a+') as sol_file:
        sol_file.write(
            f"{func_name:>14} {dataset:>10} {percent:>15} {seed:>7} {ncluster:>14} {general_desviation:>20} {infeasibility:>13} {objective:>20} {toc - tic:>23.4f}\n"
        )
