from simpson import simpson
import numpy as np
from itertools import product
from math import inf

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

def generar_muestra_poisson(minutos_limite: float):
    c = 0
    res = ''
    while (c <= minutos_limite):
        poisson = np.random.poisson(3)
        c += poisson
        res += str(poisson) + '\n'
    
    with open("muestra_poisson.txt", 'w') as f:
        f.write(res)

def generar_permutaciones(empleados):
    values = np.arange(1, empleados)
    return [
        c for c in product(values, repeat=5)
        if sum(c) == empleados
    ]

def simular(permutation: list, minimizar_media: bool):
    media = inf

    if minimizar_media:
        raise NotImplementedError("")
    else:
        raise NotImplementedError("")

    return media

def minimizar_varianza(permutaciones):
    result = []
    varianza = inf
    for p in permutaciones:
        varianza_simulada = simular(p, False)
        if varianza > varianza_simulada:
            varianza = varianza_simulada
            result = p

    result.insert(0, varianza)
    return result

def minimizar_media(permutaciones):
    result = []
    media = inf
    for p in permutaciones:
        media_simulada = simular(p, True)
        if media > media_simulada:
            media = media_simulada
            result = p

    result.insert(0, media)
    return result

def main():
    permutaciones = generar_permutaciones(EMPLEADOS_TOTALES)

    media, cajas_config, refrescos_config, freidora_config, postres_config, pollo_config = minimizar_media(permutaciones)

    print("Minimizacion media:", media)
    print(f"\tCajas: {cajas_config}")
    print(f"\tRefrescos: {refrescos_config}")
    print(f"\tFreidora: {freidora_config}")
    print(f"\tPostres: {postres_config}")
    print(f"\tPollo: {pollo_config}")

    varianza, cajas_config, refrescos_config, freidora_config, postres_config, pollo_config = minimizar_varianza(permutaciones)

    print("Minimizacion varianza:", varianza)
    print(f"\tCajas: {cajas_config}")
    print(f"\tRefrescos: {refrescos_config}")
    print(f"\tFreidora: {freidora_config}")
    print(f"\tPostres: {postres_config}")
    print(f"\tPollo: {pollo_config}")

    return 0

if __name__ == '__main__':
    main()