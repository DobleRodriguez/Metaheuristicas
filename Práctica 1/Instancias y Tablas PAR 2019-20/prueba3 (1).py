# enconding: utf-8

import numpy as np
import random
import math
import time
import pandas as pd
import os
from os import listdir

k=3
MAX_ITER = 100

#semillas = np.array([1, 7, 15, 16, 12])

"""

    FUNCIONES AUXILIARES PARA LOS ALGORITMOS GREEDY Y BUSQUEDA LOCAL

"""


"""

    FUNCIONES DE CARGA DE DATOS, INICIALIZACION DE VECTORES Y COMPROBACIONES

"""


# Funcion que carga lo datos. Tanto el set de valores como la matriz de restricciones
# Se devuelve 4 parametros, la matriz de datos, la matriz de restricciones, y el numero de filas
# y columnas de nuestra matriz de datos

def cargar_datos(f_datos, f_restricciones):
    # Read data separated by commmas in a text file
    # with open("text.txt", "r") as filestream:

    f = open(f_datos, "r")
    f1 = open(f_restricciones, "r")

    # MATRIZ DATOS 

    currentline_datos = []
    matrix_datos = []


    # print(line)
    # We need to change the input text into something we can work with (such an array)
    # currentline contains all the integers listed in the first line of "text.txt"
    # string.split(separator, maxsplit)
    # https://stackoverflow.com/questions/4319236/remove-the-newline-character-in-a-list-read-from-a-file

    for line in f: 
    #for line in f: 
        
        currentline_datos = line.rstrip('\n').split(",")

        matrix_datos.append(currentline_datos)

    
    # n == filas y d == columnas
    n = len(matrix_datos)
    d = len(currentline_datos)

    # MATRIZ RESTRICCIONES

    for i in range (n):
        for j in range (d):
            matrix_datos[i][j] = float(matrix_datos[i][j])


    currentline_const = []
    matrix_const = []

    for line in f1:
        # https://stackoverflow.com/questions/4319236/remove-the-newline-character-in-a-list-read-from-a-file
        currentline_const = line.rstrip("\n").split(",")

        matrix_const.append(currentline_const)


    return matrix_datos, matrix_const, n, d




# Funcion que genera un vector de tamaño n con valores aleatorios tomados de 1 a k

def generar_solucion_aleatoria(k,n):
    solucion = np.zeros(n)
    sol_valida = 0


    for i in range(n):
        solucion[i] = random.randint(1,k)

    return solucion




# Funcion para comprobar que ningun cluster queda vacio

def cluster_vacio(v_solucion, cluster):
    for i in range(len(v_solucion)):
        if v_solucion[i] == cluster:
            return 1

    return 0




"""

    FUNCIONES DE CALCULO DE CENTROIDES

"""


# Funcion que genera ALEATORIAMENTE los valores de los centroides

def calcular_centroides():
    #numpy.zeros(shape, dtype=float, order='C')
    #Return a new array of given shape and type, filled with zeros.

    centroides = np.zeros((k, d))

    for i in range(k):
        for j in range(d):
            #Multiplicamos *10 para que los valores vayan de 0 a 10
            centroides[i][j] = random.random()*10

    return centroides




# Funcion que recalcula los centroides
# Devuelve 1 si los centroides han cambiado y 0 si no lo han hecho

def recalcular_centroides(datos, vector_asignados, centroides, d):
    
    ha_cambiado = 0
    for i in range(k):
        nuevo_centroide = np.zeros(d)
        num_total = 0

        for j in range (n):
            #Si este punto pertenece a este cluster
            if vector_asignados[j] == i + 1:
                for l in range (d):
                    nuevo_centroide[l] = nuevo_centroide[l] + datos[j][l]

            
                num_total = num_total + 1

        for j in range(d):
            if num_total != 0:
                nuevo_centroide[j] = nuevo_centroide[j]/num_total


        # Comprobamos si han cambiado los centroides

        for j in range(d):
            if centroides[i][j] != nuevo_centroide[j]:
                ha_cambiado = 1 #HA CAMBIADO 
            


        centroides[i] = nuevo_centroide


    return ha_cambiado




"""

    FUNCIONES DE CALCULO DE INFEASIBILITY

"""


# Funcion que calcula la infeasibility.
# Si en la matriz de restricciones marca un 1 y dos instancias NO estan en el mismo cluster
# o si en la matriz de restricciones hay un -1 y estan en el mismo, sumamos +1 el valor de la
# infeasibility ya que sera una RESTRICCION INCUMPLIDA


def calcular_infeasibility(datos, valores_asignados, restricciones):
    infeasability = 0

    for i in range(len(valores_asignados)):

        if valores_asignados[i] != 0:
            for j in range(i, len(valores_asignados)):
                if valores_asignados[j] != 0:
                    if restricciones[i][j] == '-1' and valores_asignados[i] == valores_asignados[j]:
                        infeasability = infeasability + 1
                    
                    elif restricciones[i][j] == '1' and valores_asignados[i] != valores_asignados[j]:
                        infeasability = infeasability + 1

    return infeasability
        





"""

    FUNCIONES PARA LA DISTANCIA MEDIA INTRA-CLUSTER

"""


# Funcion que calcula la distancia

def distancia(datos, pos, centroides, d, vector_asignados):
    dif = 0
    contador = 0
   
    for j in range (len(datos)):
        if vector_asignados[j] == pos + 1:
            for i in range(d):
                dif = dif + math.pow((centroides[pos][i] - datos[j][i]),2)
            contador = contador + 1

    dist = math.sqrt(dif)

    return dist, contador



# Funcion para calcular la distancia media intra-cluster haciendo uso de la funcion "distancia"

def distancia_media(datos, pos, centroides, d, valores_asignados):
    dist, num_elem_dist = distancia(datos, pos, centroides, d, valores_asignados)

    distancia_media_ic = dist / num_elem_dist

    return distancia_media_ic




"""

    FUNCIONES PARA LA FUNCION OBJETIVO DE LA BUSQUEDA LOCAL

"""


# Funcion que calcula cual es la distancia maxima de entre todas las instancias

def calcular_dmaxima(datos, n):
    maximo = float("-inf")
    dist = 0

    for i in range(n):
        for j in range(i+1, n):
            for c in range(d):
                dist = dist + math.pow((datos[i][c] - datos[j][c]),2)

            distancia = math.sqrt(dist)

            if(distancia > maximo):
                maximo = distancia

    return maximo




# Funcion que calcula el numero de restricciones de la matriz de restricciones

def calcular_nrestricciones(restricciones, n):
    num_restricciones = 0

    for i in range(n):
        for j in range(n):
            if (restricciones[i][j] == "1") or (restricciones[i][j] == "-1"):
                num_restricciones = num_restricciones + 1

    return num_restricciones




# Funcion que calcula el valor de lambda de la funcion objetivo 

def calcular_lambda(datos, restricciones, n):
    distancima_maxima = calcular_dmaxima(datos,n)

    la_distancia = distancima_maxima 

    num_restricciones = calcular_nrestricciones(restricciones, n)

    lamda = (int(la_distancia))/ num_restricciones

    return lamda




# Funcion que calcula la desviacion general haciendo la media de las distancias intra cluster

def desviacion(datos, centroides, d, valores_asignados):
    desv = 0
    cont = 0

    for i in range(k):
        desv = desv + distancia_media(datos, i, centroides, d, valores_asignados)
        cont = cont + 1

    desviacion_general = desv / cont

    return desviacion_general




# FUNCION OBJETIVO DE LA BUSQUEDA LOCAL    

def f_objetivo(datos, valores_asignados, restricciones, n, d, centroides, valor_lambda):
    """
    desvv = desviacion(datos, centroides, d, valores_asignados)
    print("DESVIACION ", desvv)
    infff = calcular_infeasibility(datos, valores_asignados, restricciones)
    print("VALOR INF ", infff)
    print("VALOR LAMBDA ", valor_lambda)

    print("MULTIPLICACION ", infff * valor_lambda)
    f_objj = desvv + (infff * valor_lambda)
    print("VALOR FUNCION OBJETIVO", f_objj)
    """

    return desviacion(datos, centroides, d, valores_asignados) + (calcular_infeasibility(datos, valores_asignados, restricciones) * valor_lambda)



def cambio_solucion(vector_asignados, anterior_asignados):
    hay_cambio = False

    for i in range(len(vector_asignados)):
        if vector_asignados[i] != anterior_asignados[i]:
            hay_cambio = True
            break

    return hay_cambio




"""

        ALGORITMO GREEDY

"""


def Greedy(datos, restricciones, centroides, n, d):

    start = time.time()

    #Ordenes de exploración distintos
    #producen particiones con distintos valores para
    #𝒊𝒏𝒇𝒆𝒂𝒔𝒊𝒃𝒊𝒍𝒊𝒕𝒚 y para 𝑪 

    #hay_cambio = 1
    cambio = True

    #vectores_asignados =[0] * n
    vector_asignados = np.zeros(n)
    anterior_asignados = np.zeros(n)

    distancias = np.zeros(k)


    contador = 0

    rsi = np.arange(n)

    np.random.shuffle(rsi)

    valor_lambda = calcular_lambda(datos, restricciones, n)
    
    while (cambio and contador < MAX_ITER):
        anterior_asignados = vector_asignados.copy()

        vector_asignados = np.zeros(n)
        contador += 1

        for i in rsi:
            #valores_infeasibilipollas = np.zeros(k)
            min = float("inf")
            
       
            valores_infeas = np.zeros(k)

            for j in range(k):
                
                vector_asignados[i] = j + 1 # pongo +1 porque no existe el cluster 0
                v_infeasibility = calcular_infeasibility(datos, vector_asignados, restricciones)

                #GUARDO EN UN VECTOR TODOS LOS VALORES DE INFEASIBILITY DE ASIGNAR UN Xi A CADA CLUSTER
                valores_infeas[j] =v_infeasibility

                #DE LOS VALORES IFEASIBILITY ME QUEDO CON LAS POSICIONES DE LOS VALORES MINIMOS
                #EJEMPLO: PARTIENDO DE [3,5,0,1,0] -> [2,4]

            #input(print(f"Valores infeas = {valores_infeas}"))
            min_aux = float("inf")
            pos_aux = 0

            v_minimos = []

            for c in range(k):
                if valores_infeas[c] < min_aux:
                    min_aux = valores_infeas[c]
                    pos_aux = c
                    v_minimos = [pos_aux]

                if (valores_infeas[c] == min_aux) and (c != pos_aux):
                    v_minimos.append(c)

            #input(print(f"v minimos = {v_minimos}"))

            #De los centroides elegidos nos quedamos con aquel que tenga la menos distancia intra-cluster
            v_dist_aux = np.zeros(len(v_minimos))
            
            num_elem_dist = 0 

            for c in range(len(v_minimos)):
                dist = 0
                for dim in range(d):
                    dist += math.pow(datos[i][dim] - centroides[c][dim], 2)
                dist = math.sqrt(dist)
                #print(dist)
                #dist, num_elem_dist = distancia(datos, v_minimos[c], centroides, d, vector_asignados)  

                v_dist_aux[c] = dist

            #input(print(f"v_dist_aux = {v_dist_aux}"))
            #Nos quedamos con la menor distancia

            centroid_cercano = 0

            for c in range(len(v_dist_aux)):
                if v_dist_aux[c] < min:
                    min = v_dist_aux[c]
                    centroid_cercano = v_minimos[c] 
                    #print("EL CLUSTER CON CENTROIDE MAS CERCANO ES ", centroid_cercano)
                    
            #input(print(f"centroid cercano = {centroid_cercano}"))

            #print("cluster del centroide mas cercano ", centroid_cercano)
            #Obtenemos a que cluster pertecene esa distancia
            
           # print("CUSTER CON CENTROIDE MAS CERCANO, ", centroid_cercano+1)
            vector_asignados[i] = centroid_cercano + 1

            #print("VECTOR ASGINADOS")
            #input(print(vector_asignados))
           

        #RECALCULO CENTROIDES
        #NO HACERLE CASO A LA VARIABLE BOOLEANA QUE DEVUELVE LA FUNCION
        print(vector_asignados)
        recalcular_centroides(datos, vector_asignados, centroides, d)
        #input(print(f"Nuevos centroides = {centroides}"))

        cambio = cambio_solucion(vector_asignados, anterior_asignados)

    
    desviacion_general = desviacion(datos, centroides, d, vector_asignados)
    valor_inf = calcular_infeasibility(datos, vector_asignados, restricciones)
    valor_fobjetivo = f_objetivo(datos, vector_asignados, restricciones, n, d, centroides, valor_lambda)
    print("VALOR FUNCION OBJETIVO GREEDY")
    print(valor_fobjetivo)

    return vector_asignados, desviacion_general, valor_inf, valor_fobjetivo





"""

        BUSQUEDA LOCAL

"""


def LocalSearch(datos, restricciones, n, d):
    start = time.time()
    contador = 0
    hay_cambio = 1

    v_solucion = generar_solucion_aleatoria(k, n)
    vecinos = []

    centroides = np.zeros((k,d))
    v_centroides = []
    for i in range(n*100):
        v_centroides.append(np.zeros((k,d)))
        
    centroides_cambiados = recalcular_centroides(datos, v_solucion, centroides, d)
    valor_lambda = calcular_lambda(datos, restricciones, n)

    while(hay_cambio == 1) and (contador < MAX_ITER):
        contador = contador + 1
        vecinos = []
        for i in range(n):
            for j in range(k):
            
                if (float(j + 1)) != v_solucion[i]:

                    vecinos.append(v_solucion.copy())                  
                    vecinos[-1][i] = j + 1

                    if (cluster_vacio(v_solucion, vecinos[-1][i] ) == 0):
                        vecinos.pop()

        np.random.shuffle(vecinos)
        #print("SHAPEEEEEEEEEE", len(v_centroides))
        for c in range(len(vecinos)):
            recalcular_centroides(datos, vecinos[c], v_centroides[c], d)
        #Para n vecinos
        #Cada vecino tiene k centroides de dimension d
            
        for v in range(len(vecinos)):
            sol1 = f_objetivo(datos, vecinos[v], restricciones, n, d, v_centroides[v], valor_lambda)
            sol2 = f_objetivo(datos, v_solucion, restricciones, n, d, centroides, valor_lambda)
            if sol1 < sol2 :
                v_solucion = vecinos[v]
                centroides = v_centroides[v]
                hay_cambio = 1
                break
            else:
                hay_cambio = 0
                    
                    

    desviacion_general = desviacion(datos, centroides, d, v_solucion)
    valor_inf = calcular_infeasibility(datos, v_solucion, restricciones)/n
    valor_fobjetivo = f_objetivo(datos, v_solucion, restricciones, n, d, centroides, valor_lambda)
    print("VALOR FUNCION OBJETIVO BUSQUEDA LOCAL")
    print(valor_fobjetivo)

    return v_solucion, desviacion_general, valor_inf, valor_fobjetivo




f = "iris_set.dat"
f2 = "iris_set_const_10.const"
seed = np.random.default_rng(1)
datos, restricciones, n, d = cargar_datos(f, f2)

centroides = calcular_centroides()

print("INICIO DEL GREEDY")

start = time.time()
vector_asignados, valor_desviacion_g, inf_g, f_obj_g = Greedy(datos, restricciones, centroides, n, d)
start2 = time.time() - start

print("EL vector solucion es:")
print(vector_asignados)
print("TIEMPO ", start2)
   
print("INFESEABILITY", inf_g)

vector_greedy = (valor_desviacion_g, inf_g, f_obj_g, start2)

#dataframe = pd.DataFrame(vector)
#dataframe.to_excel("datos.xls")
   
"""print("INICIO DE LA BUSQUEDA LOCAL")

start_init_bl = time.time()
v_solucion, valor_desviacion_bl, inf_bl, f_obj_bl = LocalSearch(datos, restricciones,n, d)
start_fin_bl = time.time() - start_init_bl

print("EL vector solucion es:")
print(v_solucion)
print("TIEMPO ", start_fin_bl)
print("INFESEABILITY", inf_bl)

vector_bl = (valor_desviacion_bl, inf_bl, f_obj_bl, start_fin_bl)
"""