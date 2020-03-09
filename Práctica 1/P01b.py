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
    # Establecemos el generador de números aleatorios, para la semilla dada
    randgen = np.random.default_rng(seed)
    # Leemos los datos de los ficheros de datos y restricciones, según nombre y porcentaje
    # especificado
    data, const = read_data(set_name, const_percent)
    # Determinamos, para cada dimensión de los datos, los valores máximos y mínimos
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    # Asignamos los centroides como k tuplas con dimensiones aleatorias entre el máximo y 
    # mínimo de cada de dimensión, respectivamente
    centroids = randgen.uniform(mins, maxs, (ncluster, len(mins)))




    # Asignamos cada dato a un cluster. Las restricciones se comprueban respecto a datos siguientes
    #print(np.flatnonzero(const[0,1:])+1)

    x = np.arange(10)
    condlist = [x<3, x>5]
    choicelist = [x, x**2]
    #print(np.select(condlist,choicelist))




    
    # Asignamos orden de recorrido aleatorio para los datos
    RSI = np.arange(len(data))
    randgen.shuffle(RSI)
    clusters = np.zeros(ncluster, int)
    solution = np.zeros(len(data), int)

    for i in np.nditer(RSI):

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

        assigned_data = np.flatnonzero(solution) # a)

        test1 = np.array([0,4,13])
        #print(test1)
        #print(const[0,test1])

        # Alexa play El Futuro Funciona by La Vida Bohème
        test2 = np.take(test1, np.flatnonzero(const[0,test1] == 1))
        print(test2)
        ml_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == 1))
        #ml_conflicting_data = assigned_data[np.flatnonzero(const[assigned_data] == 1)]

        # Aquí ni siquiera tengo que comprobar nada, solo sumo +1 a todas las posiciones que no sean esas
        # Igual con CL, pero sumo +1 a las posiciones que coincidan. 












        # No sirve, los índices cambian cuando hago el take, pierdo correlación.
        ml_indices = np.flatnonzero(np.take(const[i], assigned_data)) 



    for i in np.nditer(RSI): # Índices de datos, randomizados (0..len(data)-1)
        #infeasibility = 0
        for j in np.nditer(np.arange(ncluster)): # Índices de clústeres (0..k-1)
            #infeasibility_increase = np.zeros(ncluster)
            #for k in np.nditer(np.flatnonzero(solution)): # Índices de datos con cluster asignado
                #pass
                #const_pair_indices = np.flatnonzero(np.take(const[i]) # Índices de datos restringidos respecto a i
                #ml = np.select([const[i] == 1], [const[i]])
                #infeasibility_increase[]
        #sp.spatial.distance.euclidean()
        #condlist = np.ndarray()
        #np.select()
            pass
        pass


# Iris
#print(len(data[0]))
weak_const_kmeans()