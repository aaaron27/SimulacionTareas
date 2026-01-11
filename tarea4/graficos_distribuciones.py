import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def grafico_uniforme(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)

    if bins is None:
        bins = int(np.sqrt(n))

    a = numbers.min()
    b = numbers.max()

    # histograma normalizado
    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    # PDF teórica
    x = np.linspace(a, b, 300)
    pdf = np.ones_like(x) / (b - a)

    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"Distribución Uniforme U({a:.3f}, {b:.3f})")
    plt.show()

def grafico_exponencial(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)

    if bins is None:
        bins = int(np.sqrt(n))

    beta = np.mean(numbers)

    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    x = np.linspace(0, numbers.max(), 300)
    pdf = (1 / beta) * np.exp(-x / beta)

    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"Distribución Exponencial (β = {beta:.3f})")
    plt.show()

def grafico_normal(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)

    if bins is None:
        bins = int(np.sqrt(len(numbers)))

    mu = np.mean(numbers)
    sigma = np.std(numbers, ddof=1)

    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    x = np.linspace(numbers.min(), numbers.max(), 300)
    pdf = stats.norm.pdf(x, mu, sigma)

    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"Distribución Normal (μ={mu:.3f}, σ={sigma:.3f})")
    plt.show()

def grafico_poisson(numbers):
    numbers = np.asarray(numbers, dtype=int)
    lambda_ = np.mean(numbers)

    valores, conteos = np.unique(numbers, return_counts=True)
    frec_rel = conteos / len(numbers)

    plt.figure()
    plt.bar(valores, frec_rel, alpha=0.5)

    x = np.arange(0, valores.max() + 1)
    pmf = stats.poisson.pmf(x, lambda_)

    plt.plot(x, pmf, marker='o')
    plt.xlabel("x")
    plt.ylabel("Probabilidad")
    plt.title(f"Distribución Poisson (λ = {lambda_:.3f})")
    plt.show()
