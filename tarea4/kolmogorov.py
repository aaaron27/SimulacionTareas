import numpy as np
from scipy import stats
from math import sqrt, exp
from collections import Counter
from itertools import accumulate

def poisson(numbers: list):
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

    return fo

def calc_foa(fo):
    foa = [0] * len(fo)
    j = 0
    for i in fo:
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

def calc_pe():
    return 0

def calc_pea(pe):
    return accumulate(pe)

def calc_dif_poa_pea(poa, pea):
    difs = [0]*len(poa)
    for i in range(len(poa)):
        difs[i] = abs(poa[i] - pea[i])
    return difs

def pruebas_ks(numbers: list, muestra, discreto):
    print("Muestra:", muestra)
    
    n = len(numbers)
    fo = calc_fo(numbers, discreto)
    fo_ordenado = sorted(fo.keys())
    foa = calc_foa(fo_ordenado)
    poa = calc_poa(foa, n)
    pe = calc_pe()
    pea = calc_pea(pe)
    difs = calc_dif_poa_pea(poa, pea)

    d = max(difs)
    valor_limite = 1.36 / sqrt(n)
    
    poisson(numbers)
    uniforme(numbers)
    generador_aleatorio(numbers)
    exponencial(numbers)