import numpy as np
from scipy import stats
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt 

# continuos

def generador(numbers):
    n = len(numbers)
    
    maximo = max(numbers)
    minimo = min(numbers)

    norm = [(x - minimo) / (maximo - minimo) for x in numbers]
    sorted_numbers = sorted(norm)

    fn = [(i+1)/n for i in range(n)]
    f = sorted_numbers

    d = max(abs(fn[i] - f[i]) for i in range(n))
    
    return d

def exponencial(numbers):
    n = len(numbers)
    sorted_numbers = sorted(numbers)
    media = np.mean(sorted_numbers)

    fn = [(i + 1) / n for i in range(n)]
    f = [stats.expon.cdf(x, scale=media) for x in sorted_numbers]
    d = max(abs(fn[i] - f[i]) for i in range(n))

    return d

def uniforme_continua(numbers, a=None, b=None):
    n = len(numbers)
    sorted_numbers = sorted(numbers)

    if a is None:
        a = min(numbers)
    if b is None:
        b = max(numbers)

    fn = [(i+1) / n for i in range(n)]

    # CDF teorica
    f = [(i - a) / (b-a) for i in sorted_numbers]

    d = max(abs(fn[i] - f[i]) for i in range(n))
    
    return d

def normal(numbers):
    n = len(numbers)
    sorted_numbers = sorted(numbers)

    media = np.mean(sorted_numbers)
    sigma = np.std(sorted_numbers, ddof=1)

    # CDF empirica
    fn = [(i+1) / n for i in range(n)]

    # CDF teorica
    f = [stats.norm.cdf(x, media, sigma) for x in sorted_numbers]

    d = max(abs(fn[i] - f[i]) for i in range(n))
    
    return d

# discretos

def uniforme_discreta(numbers, a=None, b=None):
    n = len(numbers)
    fo = Counter(numbers)
    values = sorted(fo.keys())

    if a is None:
        a = min(values)
    if b is None:
        b = max(values)
    
    foa = []
    acumulada = 0
    for i in values:
        acumulada += fo[i]
        foa.append(acumulada)
    
    poa = [i / n for i in foa]

    # CDF teorica discreta
    pea = [(i - a + 1) / (b - a + 1) for i in values]

    d = max(abs(poa[i] - pea[i]) for i in range(len(values)))

    return d

def poisson(numbers):
    n = len(numbers)
    fo = Counter(numbers)
    values = sorted(fo.keys())
    media = np.mean(numbers)

    # CDF empirica acumulada
    foa = []
    c = 0
    for i in values:
        c += fo[i]
        foa.append(c)
    
    poa = [i / n for i in foa]

    # CDF teorica
    pea = [stats.poisson.cdf(v, media) for v in values]

    d = max(abs(poa[i] - pea[i]) for i in range(len(values)))

    return d

def pruebas_ks(numbers: list, muestra, discreta):
    print(f"\nMuestra: {muestra}")

    n = len(numbers)
    critico = 1.36 / sqrt(n)

    print("Valor crítico:", critico)
    print("Media:", np.mean(numbers))
    print("Desviacion estandar:", np.std(numbers))

    if discreta:
        d = uniforme_discreta(numbers)
        print("Uniforme")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = poisson(numbers)
        print("Poisson")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

    else:
        d = uniforme_continua(numbers)
        print("Uniforme")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = generador(numbers)
        print("Generador")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = exponencial(numbers)
        print("Exponencial")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = normal(numbers)
        print("Normal")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)