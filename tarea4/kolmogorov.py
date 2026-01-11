import numpy as np
from scipy import stats
from math import sqrt, exp
from collections import Counter
from itertools import accumulate

POISSON = 1
UNIFORME = 2
GENERADORA = 3
EXPONENCIAL = 4

def poisson2(numbers: list):
    lambda_observado = np.mean(numbers)
    fo = dict()
    n = len(numbers)

    for i in numbers:
        fo[i] = fo.get(i, 0) + 1

    fo_ordenado = sorted(fo.keys())

    foa = [0] * len(fo)
    j = 0
    for i in fo_ordenado:
        if not j:
            foa[j] = fo[i]
        else:
            foa[j] = fo[i] + foa[j-1]
        j += 1
    
    poa = [0] * len(fo)
    for i in range(len(fo)):
        poa[i] = foa[i] / n

    pe = [0] * len(fo)
    for i in range(len(fo)):
        pe[i] = stats.poisson.cdf(fo_ordenado[i], lambda_observado)
    
    difs = [0] * len(fo)
    for i in range(len(fo)):
        difs[i] = abs(poa[i] - pe[i])
    
    d = max(difs)

    valor_critico = 1.36 / sqrt(n)

    print("K-S: Poisson")
    print("\tlambda:", lambda_observado)
    print("\tD:", d)
    print("\tValor critico:", valor_critico)
    print("\tAcepta H0:", d < valor_critico)

def uniforme(numbers: list, a=1, b=999):
    fo = dict()
    n = len(numbers)

    for i in numbers:
        fo[i] = fo.get(i, 0) + 1

    fo_ordenado = sorted(fo.keys())

    foa = [0] * len(fo)
    j = 0
    for i in fo_ordenado:
        if not j:
            foa[j] = fo[i]
        else:
            foa[j] = fo[i] + foa[j-1]
        j += 1

    poa = [0] * len(fo)
    for i in range(len(fo)):
        poa[i] = foa[i] / n

    pe = [0] * len(fo)
    values = b - a + 1

    for i in range(len(fo)):
        pe[i] = (fo_ordenado[i] - a + 1) / values
    
    difs = [0]*len(fo)
    for i in range(len(fo)):
        difs[i] = abs(poa[i] - pe[i])
    
    d = max(difs)

    valor_critico = 1.36 / sqrt(n)

    print("K-S: Uniforme")
    print("\tD:", d)
    print("\tValor critico:", valor_critico)
    print("\tAcepta H0:", d < valor_critico)

def generador_aleatorio(numbers: list):
    minimo = min(numbers)
    maximo = max(numbers)

    normalizados = [(i - minimo) / (maximo - minimo) for i in numbers]
    numeros_ordenados = sorted(normalizados)
    n = len(numbers)

    f = [0]*n
    for i in range(n):
        f[i] = (i+1) / n
    
    difs = [0]*n
    for i in range(n):
        difs[i] = abs(numeros_ordenados[i] - f[i])
    d = max(difs)

    valor_critico = 1.36 / sqrt(n)

    print("K-S: Generador Aleatorio")
    print("\tD:", d)
    print("\tValor critico:", valor_critico)
    print("\tAcepta H0:", d < valor_critico)

def exponencial(numbers: list):
    n = len(numbers)
    media = np.mean(numbers)
    lambda_esperado = 1 / media

    numeros_ordenados = sorted(numbers)

    f = [(i+1)/n for i in range(n)]

    ft = [stats.expon.cdf(numeros_ordenados[i], scale=1/lambda_esperado) for i in range(n)]

    difs = [abs(f[i] - ft[i]) for i in range(n)]

    d = max(difs)
    valor_critico = 1.36 / sqrt(n)

    print("K-S: Exponencial")
    print("\tD:", d)
    print("\tValor critico:", valor_critico)
    print("\tAcepta H0:", d < valor_critico)

def calc_fo(numbers, discreto):
    n = len(numbers)
    fo = dict()
    lit = []
    if discreto:
        fo = dict(Counter(numbers))
    else:
        num_intervalos = round(sqrt(n))
        minimo = min(numbers)
        maximo = max(numbers)
        calc = (maximo - minimo) / num_intervalos

        lit = [(minimo + i*calc, minimo + (i+1)*calc) for i in range(num_intervalos)]

        for i in lit:
            fo[i] = 0
        
        for i in numbers:
            x = min(int((i - minimo) / calc), num_intervalos - 1)
            fo[lit[x]] += 1

    return fo, lit

def calc_foa(fo, fo_ordenado):
    foa = [0] * len(fo)
    j = 0
    for i in fo_ordenado:
        if not j:
            foa[j] = fo[i]
        else:
            foa[j] = fo[i] + foa[j-1]
        j += 1
    
    return foa

def calc_poa(foa, n):
    poa = [0] * len(foa)
    for i in range(len(foa)):
        poa[i] = foa[i] / n
    
    return poa

def calc_pe(distribucion, fo, media):
    pe = [0]*len(fo)
    
    match distribucion:
        case 1:
            for i in range(len(fo)):
                pe[i] = stats.poisson.pmf(len(fo), media)
        case 2:
            for i in range(len(fo)):
                pe[i] = fo[i] / 100000
        case _:
            return -1
    return pe

def calc_pea(pe):
    return list(accumulate(pe))

def calc_dif_poa_pea(poa, pea):
    difs = [0]*len(poa)
    for i in range(len(poa)):
        difs[i] = abs(poa[i] - pea[i])
    return difs

def calc_dif_gen(sorted_numbers):
    n = len(sorted_numbers)
    f = [0]*n
    for i in range(n):
        f[i] = (i+1) / n
    
    difs = [0]*n
    for i in range(n):
        difs[i] = abs(sorted_numbers[i] - f[i])

    return difs

def calc_pea_exp(limite_superior, cant_intervalos, lambda_esperado):
   return [stats.expon.cdf(limite_superior[i], scale=1/lambda_esperado) for i in range(cant_intervalos)]

def pruebas_ks(numbers: list, muestra, discreto):
    print("\nMuestra:", muestra)
    
    n = len(numbers)
    media = np.mean(numbers)
    fo, limites_exp = calc_fo(numbers, discreto)
    fo_ordenado = sorted(fo.keys())
    foa = calc_foa(fo, fo_ordenado)
    poa = calc_poa(foa, n)
    pe1 = calc_pe(1, fo_ordenado, media)
    pe2 = calc_pe(2, fo_ordenado, media)

    pea1 = calc_pea(pe1)
    pea2 = calc_pea(pe2)

    pea4 = []
    difs4 = []
    d4 = None
    if not discreto:
        pea4 = calc_pea_exp([i for _,i in limites_exp], len(fo), 1/media)
        difs4 = calc_dif_poa_pea(poa, pea4)
        d4 = max(difs4)

    difs1 = calc_dif_poa_pea(poa, pea1)
    difs2 = calc_dif_poa_pea(poa, pea2)
    difs3 = calc_dif_gen(sorted(numbers))

    d1 = max(difs1)
    d2 = max(difs2)
    d3 = max(difs3)

    valor_limite = 1.36 / sqrt(n)

    print("Valor critico:", valor_limite)

    if not discreto:
        print("\nPoisson")

        print("\tD:", d1)
        print("\tAcepta H0:", d1 < valor_limite)

    print("Uniforme")
    
    print("\tD:", d2)
    print("\tAcepta H0:", d2 < valor_limite)

    print("Generadora aleatoria")

    print("\tD:", d3)
    print("\tAcepta H0:", d3 < valor_limite)

    if not discreto:
        print("Exponencial")

        print("\tD:", d4)
        print("\tAcepta H0:", d4 < valor_limite)