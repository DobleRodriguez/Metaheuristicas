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

        # Quiero es: contar la cantidad de ocurrencias de cada elemento y sumársela a su índice/los otros


        assigned_data = np.flatnonzero(solution) # a)

        test1 = np.arange(14)
        #test1 = np.array([0,1,4,13])
        #print(test1)
        #print(const[0,test1])

        sol = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,1])

        # Alexa play El Futuro Funciona by La Vida Bohème
        test2 = np.take(test1, np.flatnonzero(const[0,test1] == 1))
        #print(test2)
        #print(test2)
        test3 = np.take(sol, test2)
        #print(test3)
        #print(test3)

        # Te amo NumPy dame un np.hijo
        test4 = np.bincount(test3, minlength=3)
        #print(test4)
        
        test5 = np.zeros(3, int)
        test5 += test4
        print(test5)

        test6 = np.flatnonzero(test5 == np.amin(test5))
        print(test6)

        #print(centroids)
        #print(np.asarray([0,1]))
        print(centroids[test6,:])

        print(data[0])
        print(data[0] - centroids[test6,:])
        test7 = np.linalg.norm(data[0] - centroids[test6,:], axis=1)
        print("Distancias dato/centroides válidos")
        print(test7)

        test8 = np.argmin(test7)
        print(test8)

        test9 = np.zeros(10, int)
        print(test9)
        test9[0] = test6[test9]
        print(test9)

        # MOSCA: cluster0, válido? ¿Cuándo y dónde uso flatnonzero?


        #data_cluster_distance = np.linalg.norm(data[0] - clusters[test6])
        #print(data_cluster_distance)        

        #test4 = np.zeros(5, int)
        #np.where(test4)
        #test4[test3] += 1
        #test4 = np.where(test4 == test3, test4, test4+1)
        #print(test4)
        #print(test4[test3])
        #print(test3)


        ml_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == 1))
        cl_conflicting_data = np.take(assigned_data, np.flatnonzero(const[i, assigned_data] == -1))

        ml_cluster_data = np.take(solution, ml_conflicting_data)
        cl_cluster_data = np.take(solution, cl_conflicting_data)
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


        # Ahora necesito calcular distancias del dato a cada uno de esos centroides

        infes_increase = np.zeros(ncluster, int)
        infes_increase += np.bincount(cl_cluster_data, minlength=ncluster)

        ml_count_const = np.arange(ncluster)
        np.delete(ml_count_const, ml_cluster_data)

        infes_increase += np.bincount(ml_count_const, minlength=ncluster)

        considered_clusters = np.flatnonzero(infes_increase == np.amin(infes_increase))

       # np.linalg.norm(data[i] - clusters[considered_clusters])

        #min_const_clusters = np.flatnonzero(np.amin(infes_increase))




        # No sirve, los índices cambian cuando hago el take, pierdo correlación.
        #ml_indices = np.flatnonzero(np.take(const[i], assigned_data)) 



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