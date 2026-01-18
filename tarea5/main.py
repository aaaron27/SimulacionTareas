import numpy as np
from itertools import product
from math import inf
from dataclasses import dataclass
from random import random
import pandas as pd

PATH_MUESTRA_POISSON = 'data/muestra_poisson.txt'

EMPLEADOS_TOTALES = 12
REPETICIONES_TOTALES = 1000

MEDIA_CAJAS = 2.5
MEDIA_REFRESCOS = 0.75
MEDIA_FREIDORA = 3
MEDIA_POSTRES = 0.667
MEDIA_POLLO = 10

# refrescos, freidora, postres, pollo
PROBABILIDADES = [0.9, 0.7, 0.25, 0.3]
    
@dataclass
class Config:
    cajas: int
    refrescos: int
    freidora: int
    postres: int
    pollo: int

def get_cant_ordenes(n: int, p: float):
    return np.random.binomial(n, p)

def get_random_tiempo_cajas():
    return np.random.exponential(MEDIA_CAJAS)

def get_estaciones():
    estaciones_deseadas = [False]*len(PROBABILIDADES)

    for i in range(len(PROBABILIDADES)):
        if random() <= PROBABILIDADES[i]:
            estaciones_deseadas[i] = True
        
    return estaciones_deseadas

def generar_muestra_poisson(minutos_limite: float):
    c = 0
    res = ''
    while (c <= minutos_limite):
        poisson = np.random.poisson(3)
        c += poisson
        res += str(poisson) + '\n'
    
    with open(PATH_MUESTRA_POISSON, 'w') as f:
        f.write(res)

def generar_permutaciones(empleados: int) -> list[list[int]]:
    values = np.arange(1, empleados)
    return [
        c for c in product(values, repeat=5)
        if sum(c) == empleados
    ]

def calc_hora_llegada() -> list[int]:
    hora_llegada = []
    with open(PATH_MUESTRA_POISSON, 'r') as f:
        nums = f.read().split('\n')
        hora_llegada.append(int(nums[0]))
        for i in range(1, len(nums)-1):
            hora_llegada.append(hora_llegada[i-1] + int(nums[i]))
        
    return hora_llegada

def get_servidor_disponible(servidores: list[float], tiempo: float) -> int:
    min_value = inf
    min_index = -1
    for i, time in enumerate(servidores):
        if time <= min_value:
            min_index = i
            min_value = time
        if tiempo >= time:
            return i
    return min_index

def etapa_1(n_cajas: int):
    hora_llegada = calc_hora_llegada()
    servidores_cajas_disponibles = [0]*n_cajas
    servidor_usado = [0]*len(hora_llegada)
    hora_inicio_atencion = [0]*len(hora_llegada)
    tiempo_atencion = [0]*len(hora_llegada)
    tiempo_sistema1 = [0]*len(hora_llegada)
    hora_fin = [0]*len(hora_llegada)

    # primer fila
    aux_cajas = get_random_tiempo_cajas()
    tiempo_atencion[0] = aux_cajas
    hora_inicio_atencion[0] = hora_llegada[0]
    tiempo_sistema1[0] = aux_cajas
    servidor_usado[0] = 0
    hora_fin[0] = tiempo_sistema1[0] + hora_inicio_atencion[0]
    servidores_cajas_disponibles[0] = hora_fin[0]

    for i in range(1, len(hora_llegada)):
        tiempo_atencion[i] = get_random_tiempo_cajas()
        servidor_usado[i] = get_servidor_disponible(servidores_cajas_disponibles, hora_llegada[i])

        hora_inicio_atencion[i] = max(
            servidores_cajas_disponibles[servidor_usado[i]], 
            hora_llegada[i]
        )
        servidores_cajas_disponibles[servidor_usado[i]] = hora_inicio_atencion[i] + tiempo_atencion[i]
        hora_fin[i] = hora_llegada[i] + tiempo_atencion[i]
        tiempo_sistema1[i] = hora_fin[i] - hora_llegada[i]
    
    df = pd.DataFrame({
        "Hora llegada": hora_llegada,
        "Hora inicio atención": hora_inicio_atencion,
        "Servidor usado": servidor_usado,
        "Tiempo atención": tiempo_atencion,
        "Tiempo sistema": tiempo_sistema1,
        "Hora fin": hora_fin
    })

    print(df)

    return hora_fin

def etapa_2(hora_llegada: list[int]) -> list[int]:
    raise NotImplementedError('h')

def simular(permutation: list[int]) -> tuple[float, float]:
    hora_fin_etapa_1 = etapa_1(permutation[0])
    hora_sistema_total = etapa_2(hora_fin_etapa_1)

    return np.mean(hora_sistema_total), np.var(hora_sistema_total)

def minimizar(permutaciones: list[list[int]]) -> tuple[float, float, Config, Config]:
    media_result = [-1]*5
    varianza_result = [-1]*5
    media = inf
    varianza = inf

    #for _ in range(REPETICIONES_TOTALES):

    simular([6,2,2,1,1])
    # for p in permutaciones:
    #     media_simulada, varianza_simulada = simular(p)
    #     if media > media_simulada:
    #         media = media_simulada
    #         media_result = p
    #     if varianza > varianza_simulada:
    #         varianza = varianza_simulada
    #         varianza_result = p
    
    return media, varianza, Config(*media_result), Config(*varianza_result)

def main():
    permutaciones = generar_permutaciones(EMPLEADOS_TOTALES)

    media, varianza, config_media, config_varianza = minimizar(permutaciones)

    print("Minimizacion media:", media)
    print(f"\tCajas: {config_media.cajas}")
    print(f"\tRefrescos: {config_media.refrescos}")
    print(f"\tFreidora: {config_media.freidora}")
    print(f"\tPostres: {config_media.postres}")
    print(f"\tPollo: {config_media.pollo}")

    print("Minimizacion varianza:", varianza)
    print(f"\tCajas: {config_varianza.cajas}")
    print(f"\tRefrescos: {config_varianza.refrescos}")
    print(f"\tFreidora: {config_varianza.freidora}")
    print(f"\tPostres: {config_varianza.postres}")
    print(f"\tPollo: {config_varianza.pollo}")

    return 0

if __name__ == '__main__':
    main()