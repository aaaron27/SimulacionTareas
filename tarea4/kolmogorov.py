import numpy as np
from scipy import stats
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt 


def grafico_normal(numbers):
    mu = np.mean(numbers)
    sigma = np.std(numbers)

    x = np.linspace(min(numbers), max(numbers), 1000)
    pdf = stats.norm.pdf(x, mu, sigma)

    plt.hist(numbers, bins=50, density=True, alpha=0.6, label="Muestra")
    plt.plot(x, pdf, linewidth=2, label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})")

    plt.title("Distribución Normal")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid()
    plt.show()

def grafico_exponencial(numbers):
    media = np.mean(numbers)

    x = np.linspace(0, max(numbers), 1000)
    pdf = stats.expon.pdf(x, scale=media)

    plt.hist(numbers, bins=50, density=True, alpha=0.6, label="Muestra")
    plt.plot(x, pdf, linewidth=2, label=f"Exponencial(media={media:.3f})")

    plt.title("Distribución Exponencial")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid()
    plt.show()

def grafico_uniforme_continua(numbers):
    a = min(numbers)
    b = max(numbers)

    x = np.linspace(a, b, 1000)
    pdf = stats.uniform.pdf(x, loc=a, scale=b-a)

    plt.hist(numbers, bins=50, density=True, alpha=0.6, label="Muestra")
    plt.plot(x, pdf, linewidth=2, label=f"Uniforme([{a:.2f},{b:.2f}])")

    plt.title("Distribución Uniforme Continua")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid()
    plt.show()

def grafico_poisson(numbers):
    lambda_ = np.mean(numbers)

    valores, frec = np.unique(numbers, return_counts=True)
    frec_rel = frec / len(numbers)

    pmf = stats.poisson.pmf(valores, lambda_)

    plt.bar(valores, frec_rel, alpha=0.6, label="Muestra")
    plt.plot(valores, pmf, 'o-', linewidth=2, label=f"Poisson(λ={lambda_:.3f})")

    plt.title("Distribución Poisson")
    plt.xlabel("k")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.grid()
    plt.show()

def grafico_uniforme_discreta(numbers):
    valores, frec = np.unique(numbers, return_counts=True)
    n = len(numbers)

    a = min(valores)
    b = max(valores)

    frec_rel = frec / n
    p = 1 / (b - a + 1)

    plt.bar(valores, frec_rel, alpha=0.6, label="Muestra")
    plt.hlines(p, a, b, colors='r', label=f"Uniforme discreta [{a},{b}]")

    plt.title("Distribución Uniforme Discreta")
    plt.xlabel("k")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.grid()
    plt.show()

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

# discretos

def uniforme(numbers, a=None, b=None):
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
    pea = [(i - a + 1) / (b - a + 1) for i in values]

    d = max(abs(poa[i] - pea[i]) for i in range(len(values)))

    return d

def poisson(numbers):
    n = len(numbers)
    fo = Counter(numbers)
    values = sorted(fo.keys())
    media = np.mean(numbers)

    foa = []
    c = 0
    for i in values:
        c += fo[i]
        foa.append(c)
    
    poa = [i / n for i in foa]
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
        d = uniforme(numbers)
        print("Uniforme")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = poisson(numbers)
        print("Poisson")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

    else:
        d = generador(numbers)
        print("Generador")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)

        d = exponencial(numbers)
        print("Exponencial")
        print("\tD:", d)
        print("\tAcepta H0:", d < critico)