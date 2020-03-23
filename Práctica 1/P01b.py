# Práctica 01.b
# Técnicas de Búsqueda Local y Algoritmos Greedy para el Problema del Agrupamiento con Restricciones
# Metaheurísticas (MH) - Grupo A1
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import numpy as np
import pathlib as pl
import time

# Read data y hacer lambda AFUERA
# Pasar como parámetros

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

    return data, const, scale_factor



def local_search(data, const, ncluster, scale_factor, randgen):
    # Aseguramos que al menos cada cluster tiene asignado una instancia 
    solution = np.append(np.arange(ncluster), randgen.integers(ncluster, size=len(data)-ncluster))
    randgen.shuffle(solution)

    data_distances = np.empty(0)
    for i in np.nditer(np.arange(len(data))):
        data_distances = np.append(data_distances, np.linalg.norm(data[i] - data[i:], axis=1))
    # Techo de la distancia máxima (lambda)
    scale_factor = np.ceil(np.amax(data_distances))

    # Centroides (calculados según los datos de cada clúster)
    centroids = np.empty([ncluster, len(data[0])])
    for i in np.nditer(np.arange(ncluster)):
        centroids[i] = np.mean(data[np.flatnonzero(solution == i)], axis=0)


    # Distancias intraclusteres
    intra_cluster_distances = np.empty(ncluster)
    for i in np.nditer(np.arange(ncluster)):
        intra_cluster_distances[i] = np.mean(np.linalg.norm(centroids[i] - data[np.flatnonzero(solution == i)], axis=1))

    # Infeasibility generada por cada elemento de la solución, y su suma
    infeasibility = 0
    for i in np.nditer(np.arange(len(solution))):
        ml_conflicting_data = np.flatnonzero(const[i, :i] == 1)
        #print(ml_conflicting_data)
        cl_conflicting_data = np.flatnonzero(const[i, :i] == -1)
        #input(print(cl_conflicting_data))
        infeasibility += np.count_nonzero(solution[i] != solution[ml_conflicting_data]) \
            + np.count_nonzero((solution[i] == solution[cl_conflicting_data]))
    
    # Desviación general de la solución
    general_desviation = np.mean(intra_cluster_distances)

    # Valor de la función objetivo
    objective = general_desviation + infeasibility * scale_factor
    best_solution = False
    while(not best_solution):
        best_solution = True
        # Todo el proceso de construcción de vecindario

        # Generación del vecindario virtual
        # Que no dejen un cluster vacío
        cluster_size = np.bincount(solution)
        single_elem_clusters = np.flatnonzero(cluster_size == 1)
        possible_elements = np.flatnonzero(~np.isin(solution, single_elem_clusters)) 


        # Y que excluyan la solución (haya un cambio)
        neighbors = np.empty([0,2], int)
        for i in np.nditer(np.arange(ncluster)):
            change_cluster = possible_elements[np.isin(possible_elements, np.flatnonzero(solution != i))]
            neighbors = np.append(neighbors, np.stack(np.meshgrid(change_cluster, i), -1).reshape(-1, 2), axis=0)


        randgen.shuffle(neighbors)

        counter = 0
        for index, cluster in neighbors:
            counter += 1

            new_possibility = np.copy(solution)
            new_possibility[index] = cluster

            new_centroids = np.copy(centroids)
            new_intra_cluster = np.copy(intra_cluster_distances)

            ml_conflicting_data = np.flatnonzero(const[index] == 1)
            cl_conflicting_data = np.flatnonzero(const[index] == -1)

            original_infeas = np.count_nonzero(solution[index] != solution[ml_conflicting_data]) \
            + np.count_nonzero((solution[index] == solution[cl_conflicting_data]))

            infeas_change = np.count_nonzero(cluster != new_possibility[ml_conflicting_data]) \
            + np.count_nonzero((cluster == new_possibility[cl_conflicting_data]))

            new_infeas = infeasibility - original_infeas + infeas_change

            changed_clusters = np.array([cluster, solution[index]])
            for i in np.nditer(changed_clusters):
                new_centroids[i] = np.mean(data[np.flatnonzero(new_possibility == i)], axis=0)
                new_intra_cluster[i] = np.mean(np.linalg.norm(new_centroids[i] - data[np.flatnonzero(new_possibility == i)], axis=1))
            
            new_desviation = np.mean(new_intra_cluster)

            new_objective = new_desviation + new_infeas * scale_factor
            if (new_objective < objective):
                solution = new_possibility
                centroids = new_centroids
                intra_cluster_distances = new_intra_cluster
                infeasibility = new_infeas
                general_desviation = new_desviation
                objective = new_objective
                best_solution = False
                break 
    return general_desviation, infeasibility, objective



def weak_const_kmeans(data, const, ncluster, scale_factor, randgen):
    # mins, maxs = vector con los valores mínimos y máximos de los datos para cada dimensión, respectivamente
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    # centroids = vector de tamaño ncluster donde cada elemento es el centroide de su respectivo cluster,
    # asignando aleatoriamente cada dimensión entre los valores de mins y maxs 
    centroids = randgen.uniform(mins, maxs, (ncluster, len(mins)))    
    # RSI = orden de lectura de los datos, aleatorio
    RSI = np.arange(len(data))
    # solution = vector solución, inicializado a -1, para representar que no pertenecen a ningún cluster
    # sol_update = booleano para representar si el vector solución cambia
    cluster_update = True
    # n_iters = contador de iteraciones
    n_iters = 0
    old_centroids = np.empty(ncluster)
    # Ciclo principal: mientras haya algún tipo de actualización
    randgen.shuffle(RSI)
    while(cluster_update):
        solution = np.full(len(data), -1, int)
        n_iters += 1
        cluster_update = False
        # Valor de infeasibility total
        infeasibility = 0
        # Ciclo secundario: para cada dato (barajado)
        for i in np.nditer(RSI):
            # assigned_data = datos que estén contenidos en algún cluster
            assigned_data = np.flatnonzero(solution != -1)
            # ml(cl)_conflicting_data = índices de datos con restricciones ML(CL) respecto a i
            ml_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == 1))
            cl_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == -1))
            # ml(cl)_cluster_data = cluster al que pertenecen los datos en ml(cl)_conflicting_data
            ml_cluster_data = np.take(solution, ml_conflicting_data)
            cl_cluster_data = np.take(solution, cl_conflicting_data)            
            # infeas_increase = vector de incremento de infeasibility asociado a cada cluster 
            ml_cluster_data = np.bincount(ml_cluster_data, minlength=ncluster)
            infeas_increase = np.full(ncluster, np.sum(ml_cluster_data))
            infeas_increase -= ml_cluster_data
            infeas_increase += np.bincount(cl_cluster_data, minlength=ncluster)
            # min_infeas_increase = incremento mínimo de infeasibility
            min_infeas = np.amin(infeas_increase)
            # actualizamos la infeasibility total
            infeasibility += min_infeas
            # considered_clusters = clusteres que incrementen al mínimo la infeasibility
            considered_clusters = np.flatnonzero(infeas_increase == min_infeas)
            if (len(considered_clusters) > 1):
                # data_cluster_distances = distancia (euclidiana) entre el dato i y los clústeres considerados
                # calculada con la norma l2
                data_cluster_distances = np.linalg.norm(data[i] - centroids[considered_clusters,:], axis=1)
                #print(f"{data_cluster_distances} dcd")
                # selected_cluster = cluster con mínima distancia al dato (de entre los considerados)
                # en caso de empate, selecciona el primero en orden de lectura
                selected_cluster = considered_clusters[np.argmin(data_cluster_distances)]
            else:
                selected_cluster = considered_clusters[0]
            # Actualización del dato
            solution[i] = selected_cluster
        # Actualización de clusteres
        # Puede optimizarse?
        for i in np.nditer(np.arange(ncluster)):
            #print(data[np.flatnonzero(solution==i)])
            #print(np.mean(data[np.flatnonzero(solution == i)],axis=0))
            
            centroids[i] = np.mean(data[np.flatnonzero(solution == i)], axis=0)

        if (not np.array_equal(old_centroids, centroids)):
            cluster_update = True
            old_centroids = np.copy(centroids)
        
        inter_cluster_distances = np.empty(ncluster)
        for i in np.nditer(np.arange(ncluster)):
            inter_cluster_distances[i] = np.mean(np.linalg.norm(centroids[i] - data[np.flatnonzero(solution == i)], axis=1))
        general_desviation = np.mean(inter_cluster_distances)
        #input(print(f"{solution} solution \n {np.bincount(solution)} bincount \n {centroids} centroids \
            #\n {general_desviation} general desviation \n {infeasibility} infeasibility"))
    return general_desviation, infeasibility, general_desviation + infeasibility * scale_factor


np.seterr(all='raise')
sets = np.array(["iris", "rand", "ecoli"])
nclusters = np.array([3, 3, 8])
percents = np.array([10, 20])
seeds = np.array([1123, 9145, 420, 174, 745])

values = np.stack(np.meshgrid(percents, sets, seeds), -1).reshape(-1,3)
sets, set_repeats = np.unique(values[:,1], return_counts=True)

set_repeats = np.repeat(nclusters, set_repeats)
values = np.concatenate((values, np.array([set_repeats]).T), axis=-1)

open(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/solutions.txt", 'w+').close()
for percent,dataset,seed,ncluster in values:
    data, const, scale_factor = init_data(dataset, percent)
    randgen = np.random.default_rng(int(seed))
    tic_ls = time.perf_counter()
    general_desviation, infeasibility, objective = local_search(data, const, int(ncluster), int(scale_factor), randgen)
    toc_ls = time.perf_counter()
    with open(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/solutions.txt",'a+') as sol_file:
        sol_file.write(
f"""Algoritmo Búsqueda Local para {dataset} con {percent}% de restricciones, semilla {seed} y {ncluster} clústeres
Desviación general: {general_desviation}
Restricciones incumplidas: {infeasibility}
Función Objetivo: {objective}
Tiempo de ejecución: {toc_ls - tic_ls: 0.4f}s
""")
    tic_kmeans = time.perf_counter()
    general_desviation, infeasibility, objective = weak_const_kmeans(data, const, int(ncluster), int(scale_factor), randgen)
    toc_kmeans = time.perf_counter()
    with open(pl.Path(__file__).parent / f"Instancias y Tablas PAR 2019-20/solutions.txt",'a+') as sol_file:
        sol_file.write(
f"""Algoritmo Greedy k-medias para {dataset} con {percent}% de restricciones, semilla {seed} y {ncluster} clústeres
Desviación general: {general_desviation}
Restricciones incumplidas: {infeasibility}
Función Objetivo: {objective}
Tiempo de ejecución: {toc_kmeans - tic_kmeans: 0.4f}s
""")
