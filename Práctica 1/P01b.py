# Práctica 01.b
# Técnicas de Búsqueda Local y Algoritmos Greedy para el Problema del Agrupamiento con Restricciones
# Metaheurísticas (MH) - Grupo A1
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import numpy as np
import scipy as sp
import pathlib as pl

def read_data(set_name, const_percent):
    data = np.loadtxt(pl.Path(__file__).parent /
    f"Instancias y Tablas PAR 2019-20/{set_name}_set.dat", delimiter=',')
    const = np.loadtxt(pl.Path(__file__).parent /
    f"Instancias y Tablas PAR 2019-20/{set_name}_set_const_{const_percent}.const", delimiter=',')
    return data, const

"""
def local_search(data, const, k, max_eval=100000):
    n = len(data)
    #cluster_assignment = np.random.Generator.shuffle(np.concatenate(np.arange(1,k+1), np.random.Generator.random()))

    pass
"""

# K-medias Restringido Débil 
# Calcular el incremento de infeasibility para cada dato en cada cluster
# De entre los que tengan menor incremento de infeasibility, tomar el centroide más cercano
# Actualizar centroides


########### Ideas
# Barajar en cada iteración
# Centroides iniciales entre el rango min-max de cada dimensión


# Creación clusteres iniciales
# Centroides tomando datos aleatorios entre valores mínimo y máximo de cada dimensión
"""def init_data(set_name="iris", ncluster=5, const_percent=10):
    data, const = read_data(set_name, const_percent)
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    centroids = np.random.default_rng().uniform(mins, maxs, (ncluster, len(mins)))
    print(centroids)
    return data, const
"""

def const_check(const, clust_assign, i, j):
    return ((const[i,j] == 1 and clust_assign[i] != clust_assign[j]) or 
        (const[i,j] == -1 and clust_assign[i] == clust_assign[j]))
    

def weak_const_kmeans(set_name="iris", ncluster=3, const_percent=10, seed=1):
#################################################################################################
    """
    # Trozo test: no le importa nada de los datos reales, no cree en Dios
    # y odia a la policía
    # 14 datos
    # sol = Asignación dato/cluster (índice/valor)
    sol = np.array([1,1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,1])
    const = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,1])
    # test1 = elementos YA en cluster (aplicando nonzero a sol)
    test1 = np.array([0,1,4,13])
    print(f"{test1} Test1")
    # test2 = índice de los elementos que tienen restricciones ML con el dato i
    test2 = np.take(test1, np.flatnonzero(const[test1] == 1))
    print(f"{test2} Test2")
    # test3 = número de cluster al que pertenecen los datos con restricción ML y cluster
    test3 = np.take(sol, test2)
    print(f"{test3} Test3")
    # test4 = Cantidad de elementos con restricciones ML respecto a i en cada cluster 
    test4 = np.bincount(test3, minlength=3)
    print(f"{test4} Test4")
    # test5 = Sumar dichos aumentos a la infeasibility actual (necesario? probs para combinar ML y CL)
    test5 = np.zeros(3, int)
    test5 += test4
    print(f"{test5} Test5")
    # test6 = Determinar el vector de mínimos entre estos (clusteres con mínima infeasibility)
    test6 = np.flatnonzero(test5 == np.amin(test5))
    print(f"{test6} Test6")
    #test7 = Valor de distancias entre el dato y los centroides de cada uno de los clústeres válidos
    test7 = np.linalg.norm(data[0] - centroids[test6,:], axis=1)
    print(f"{test7} Test7")
    # test8 = Número de cluster con mínima distancia respecto al dato (de entre mínima infeasibility)
    test8 = np.argmin(test7)
    print(f"{test8} Test8")
    # test9 = actualización del vector solución asignando cluster al dato i
    test9 = sol
    test9[0] = test6[test8] 
    print(f"{test9} Test9")
    """
########################################################################################################

        # Hablemos. Qué coño de la madre quieres hacer, Javier. 
        # Quiero terminar con un vector, de tamaño ncluster, tal que en cada posición
        # tenga el incremento de infeasibility causado por poner el dato en el cluster
        # con ese índice. Es decir, infes_increase[ncluster] donde cada dato está en [0, inf)

        # Para eso, necesito comprobar la información relativa a las restricciones del valor i
        # contra los elementos YA PERTENECIENTES a un cluster.
        # Si no perteneces a un cluster, te jodes y esperas tu turno.

        # Para eso, necesito:
        # a) Saber qué elementos ya están en un cluster
        # b) Saber qué restricciones involucran esos datos
        # c) Determinar en cuánto aumenta la infeasibility en cada asignación a cluster, con esto en mente

        # Cómo hago cada una de estas cosas. 
        # Coño mi bro excelente pregunta, a ver
        # a) es fácil. np.nonzero(solution)
        # b) Se me ocurre un select, capaz 2 (si descubro como particionar cada condición en un array cada una)
        # c) Verga. Tengo que cruzar ambos datos, y no estoy seguro de cómo. 

        # Una vez tienes ese vector es relativamente sencillo. Sacas el índice de los valores mínimos y después
        # determinas el mínimo entre las distancias de cada uno de esos clústeres al dato
        # y actualizas toda la información (vector solución e infeasibility)

        # Quiero es: contar la cantidad de ocurrencias de cada elemento y sumársela a su índice/los otros

            #ml_conflicting_data = assigned_data[np.flatnonzero(const[assigned_data] == 1)]

            # Aquí ni siquiera tengo que comprobar nada, solo sumo +1 a todas las posiciones que no sean esas
            # Igual con CL, pero sumo +1 a las posiciones que coincidan. 


            # Qué coño tienes, Javier. Tanto peo pa'no terminar
            # test2 da los índices de los datos con cluster asignado que tienen restricciones con el dato i
            # test3 da los índices de los clústeres a los que dichos datos pertenecen
            # Para cada elemento de test3, sumo 1 al RESTO de elementos en infes_increase 
            # INDIZO CON test3
            # Eso me sirve para las CL. Sumo uno a esos valores. Para las ML necesito sumar todos los otros
            # valores

            # Ahora tengo el vector de infeasibilites, donde en cada posición está el incremento asociado
            # al cluster de dicho índice


            # LA IDEA
            # Un np.where donde la condición sea que el índice coincida con elemento del array
            # la acción sea +1 y el default sea igual
            # EL problema es establecer la condición


    # randgen = Generador de números aleatorios según la semilla
    randgen = np.random.default_rng(seed)
    # data, const = matriz de datos y restricciones, respectivamente (leídas según
    # nombre y número dado)
    data, const = read_data(set_name, const_percent)
    # mins, maxs = vector con los valores mínimos y máximos de los datos para cada dimensión, respectivamente
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    # centroids = vector de tamaño ncluster donde cada elemento es el centroide de su respectivo cluster,
    # asignando aleatoriamente cada dimensión entre los valores de mins y maxs 
    centroids = randgen.uniform(mins, maxs, (ncluster, len(mins)))    
    # RSI = orden de lectura de los datos, aleatorio
    RSI = np.arange(len(data))
    randgen.shuffle(RSI)
    # solution = vector solución, inicializado a -1, para representar que no pertenecen a ningún cluster
    solution = np.full(len(data), -1, int)
    # sol_update = booleano para representar si el vector solución cambia
    sol_update = True
    # n_iters = contador de iteraciones
    n_iters = 0
    # Ciclo principal: mientras haya algún tipo de actualización
    while(sol_update):
        n_iters += 1
        sol_update = False
        # Valor de infeasibility total
        infeasibility = 0
        # Ciclo secundario: para cada dato (barajado)
        for i in np.nditer(RSI):
            # assigned_data = datos que estén contenidos en algún cluster
            assigned_data = np.flatnonzero(solution != -1)
            print(f"{assigned_data} : assigned_data")
            # ml(cl)_conflicting_data = índices de datos con restricciones ML(CL) respecto a i
            ml_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == 1))
            cl_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == -1))
            # ml(cl)_cluster_data = cluster al que pertenecen los datos en ml(cl)_conflicting_data
            ml_cluster_data = np.take(solution, ml_conflicting_data)
            cl_cluster_data = np.take(solution, cl_conflicting_data)
            # infeas_increase = vector de incremento de infeasibility asociado a cada cluster 
            # (inicialmente 0)
            infeas_increase = np.zeros(ncluster, int)
            # Para las restricciones cl, sumamos +1 en infeasibility a los clusteres donde el dato
            # restringido pertenezca (pues de estar i ahí, compartirían cluster y violaría el CL)
            infeas_increase += np.bincount(cl_cluster_data, minlength=ncluster)
            # ml_count_const = vector "contrario" a ml_cluster_data, con el índice de todos los clústeres
            # que NO contengan un elemento con restricción ML respecto a i (aquellos en ml_cluster_data)
            ml_count_const = np.arange(ncluster)
            np.delete(ml_count_const, ml_cluster_data)
            # Para las restricciones ml, sumamos +1 en infeasibility a los clusteres donde el dato 
            # restringido NO pertenezca (pues de estar i ahí, no compartirían cluster y violaría el ML)
            infeas_increase += np.bincount(ml_count_const, minlength=ncluster)
            # min_infeas_increase = incremento mínimo de infeasibility
            min_infeas = np.amin(infeas_increase)
            # considered_clusters = clusteres que incrementen al mínimo la infeasibility
            considered_clusters = np.flatnonzero(infeas_increase == min_infeas)
            # actualizamos la infeasibility total
            infeasibility += min_infeas
            # data_cluster_distances = distancia (euclidiana) entre el dato i y los clústeres considerados
            # calculada con la norma l2
            data_cluster_distances = np.linalg.norm(i - centroids[considered_clusters,:], axis=1)
            # selected_cluster = cluster con mínima distancia al dato (de entre los considerados)
            # en caso de empate, selecciona el primero en orden de lectura
            selected_cluster = np.argmin(data_cluster_distances)
            # Actualización del dato
            if (solution[i] != selected_cluster):
                sol_update = True
                solution[i] = selected_cluster

            input("Iteración del for")
        print(f"{solution} Solución en iteración {n_iters} con infeasibility {infeasibility}")
# Iris
#print(len(data[0]))
weak_const_kmeans()