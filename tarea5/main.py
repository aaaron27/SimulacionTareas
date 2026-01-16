import numpy as np
from itertools import product
from math import inf
from dataclasses import dataclass

PATH_MUESTRA_POISSON = 'data/muestra_poisson.txt'

EMPLEADOS_TOTALES = 12

MEDIA_CAJAS = 2.5
MEDIA_REFRESCOS = 0.75
MEDIA_FREIDORA = 3
MEDIA_POSTRES = 0.667
MEDIA_POLLO = 10

PROB_REFRESCOS = 0.9
PROB_FREIDORA = 0.7
PROB_POSTRES = 0.7
PROB_POLLO = 0.3
    
@dataclass
class Config:
    cajas: int
    refrescos: int
    freidora: int
    postres: int
    pollo: int

def get_cant_ordenes(n: int, p: float):
    return np.random.binomial(5, 0.4)

def generar_muestra_poisson(minutos_limite: float) -> None:
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

def simular(permutation: list[int]) -> tuple[float, float]:
    media = inf
    varianza = inf

    hora_llegada = calc_hora_llegada()

    raise NotImplementedError("hola")

    return media, varianza

def minimizar(permutaciones: list[list[int]]) -> tuple[float, float, Config, Config]:
    media_result = []
    varianza_result = []
    media = inf
    varianza = inf

    for p in permutaciones:
        media_simulada, varianza_simulada = simular(p)
        if media > media_simulada:
            media = media_simulada
            media_result = p
        if varianza > varianza_simulada:
            varianza = varianza_simulada
            varianza_result = p
    
    return media, varianza, Config(media_result), Config(varianza_result)

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