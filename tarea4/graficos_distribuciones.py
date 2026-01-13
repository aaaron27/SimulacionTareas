import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def analizar_muestra_uniforme(numbers, nombre="muestra"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))
    sigma = np.std(numbers, ddof=1)
    
    # Parámetros
    a = numbers.min()
    b = numbers.max()
    
    # Crear figura con dos subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6, 
            color='lightcoral', edgecolor='black', label='Histograma')
    
    # PDF teórica (línea horizontal)
    x_pdf = np.linspace(a, b, 300)
    pdf_teorica = np.ones_like(x_pdf) / (b - a)
    
    ax1.plot(x_pdf, pdf_teorica, 'r-', linewidth=3, 
             label=f'PDF Teórica: $f(x) = \\frac{{1}}{{b-a}}$')
    
    # Líneas verticales en los límites
    ax1.axvline(a, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(b, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title(f'PDF - Distribución Uniforme ({nombre})\n$a = {a:.4f}$, $b = {b:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    
    # CDF teórica
    F_teorica = (numbers_sorted - a) / (b - a)
    F_teorica = np.clip(F_teorica, 0, 1)  # Limitar entre 0 y 1
    
    ax2.step(numbers_sorted, Fn, where='post', linewidth=2, 
             label='CDF Empírica', color='blue')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5, 
             label=f'CDF Teórica: $F(x) = \\frac{{x-a}}{{b-a}}$')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probabilidad Acumulada', fontsize=12)
    ax2.set_title(f'CDF - Distribución Uniforme ({nombre})\n$a = {a:.4f}$, $b = {b:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analizar_muestra_exponencial(numbers, nombre="m16"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))
    
    # Parámetros
    beta = np.mean(numbers)
    lambd = 1 / beta
    sigma = np.std(numbers, ddof=1)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6, 
             color='skyblue', edgecolor='black', label='Histograma')
    
    x_pdf = np.linspace(0, numbers.max(), 300)
    pdf_teorica = (1 / beta) * np.exp(-x_pdf / beta)
    
    ax1.plot(x_pdf, pdf_teorica, 'r-', linewidth=2.5, 
             label=f'PDF Teórica: $f(x) = \\frac{{1}}{{β}} e^{{-x/β}}$')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title(f'PDF - Distribución Exponencial ({nombre})\n$β = {beta:.4f}$, $λ = {lambd:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    F_teorica = 1 - np.exp(-numbers_sorted / beta)
    
    ax2.step(numbers_sorted, Fn, where='post', linewidth=2, 
             label='CDF Empírica', color='blue')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5, 
             label=f'CDF Teórica: $F(x) = 1 - e^{{-x/β}}$')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probabilidad Acumulada', fontsize=12)
    ax2.set_title(f'CDF - Distribución Exponencial ({nombre})\n$β = {beta:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analizar_muestra_normal(numbers, nombre="m8"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))
    
    # Parámetros
    mu = np.mean(numbers)
    sigma = np.std(numbers, ddof=1)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6, 
             color='lightgreen', edgecolor='black', label='Histograma')
    
    x_pdf = np.linspace(numbers.min(), numbers.max(), 300)
    pdf_teorica = stats.norm.pdf(x_pdf, mu, sigma)
    
    ax1.plot(x_pdf, pdf_teorica, 'r-', linewidth=2.5, 
             label=f'PDF Teórica: $f(x) = \\frac{{1}}{{σ\\sqrt{{2π}}}} e^{{-\\frac{{(x-μ)^2}}{{2σ^2}}}}$')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title(f'PDF - Distribución Normal ({nombre})\n$μ = {mu:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    F_teorica = stats.norm.cdf(numbers_sorted, mu, sigma)
    
    ax2.step(numbers_sorted, Fn, where='post', linewidth=2, 
             label='CDF Empírica', color='blue')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5, 
             label='CDF Teórica')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probabilidad Acumulada', fontsize=12)
    ax2.set_title(f'CDF - Distribución Normal ({nombre})\n$μ = {mu:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analizar_muestra_gamma(numbers, nombre="m13"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))
    sigma = np.std(numbers, ddof=1)
    
    # Estimar parámetros usando MLE
    shape, loc, scale = stats.gamma.fit(numbers, floc=0)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6, 
             color='salmon', edgecolor='black', label='Histograma')
    
    x_pdf = np.linspace(0, numbers.max(), 300)
    pdf_teorica = stats.gamma.pdf(x_pdf, shape, loc, scale)
    
    ax1.plot(x_pdf, pdf_teorica, 'r-', linewidth=2.5, 
             label=f'PDF Teórica: $f(x) = \\frac{{x^{{α-1}} e^{{-x/β}}}}{{Γ(α)β^α}}$')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title(f'PDF - Distribución Gamma ({nombre})\n$α = {shape:.4f}$, $β = {scale:.4f}$, $σ = {sigma:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    F_teorica = stats.gamma.cdf(numbers_sorted, shape, loc, scale)
    
    ax2.step(numbers_sorted, Fn, where='post', linewidth=2, 
             label='CDF Empírica', color='blue')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5, 
             label='CDF Teórica')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probabilidad Acumulada', fontsize=12)
    ax2.set_title(f'CDF - Distribución Gamma ({nombre})\n$α = {shape:.4f}$, $β = {scale:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
def cdf_piecewise(x):
    x = np.asarray(x)
    F = np.zeros_like(x, dtype=float)

    F[(0 <= x) & (x <= 1)] = 0.5 * x[(0 <= x) & (x <= 1)]
    F[(1 < x) & (x <= 1.5)] = 0.5
    F[(1.5 < x) & (x <= 1.875)] = np.exp(x[(1.5 < x) & (x <= 1.875)] - 1.5) - 0.5
    F[x > 1.875] = np.exp(1.875 - 1.5) - 0.5

    return F

def pdf_piecewise(x):
    x = np.asarray(x)
    f = np.zeros_like(x, dtype=float)

    f[(0 <= x) & (x <= 1)] = 0.5
    f[(1.5 < x) & (x <= 1.875)] = np.exp(1.72*(x[(1.5 < x) & (x <= 1.875)] - 1.5))

    return f


def analizar_muestra_piecewise(numbers, nombre="muestra"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6,
             edgecolor='black', label='Histograma')

    x_pdf = np.linspace(0, 2, 400)
    ax1.plot(x_pdf, pdf_piecewise(x_pdf), 'r-', linewidth=2.5,
             label='PDF teórica')

    ax1.set_xlabel('x')
    ax1.set_ylabel('Densidad de Probabilidad')
    ax1.set_title(f'PDF - Función por partes ({nombre})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    F_teorica = cdf_piecewise(numbers_sorted)

    ax2.step(numbers_sorted, Fn, where='post',
             linewidth=2, label='CDF Empírica')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5,
             label='CDF Teórica')

    ax2.set_xlabel('x')
    ax2.set_ylabel('Probabilidad Acumulada')
    ax2.set_title(f'CDF - Función por partes ({nombre})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analizar_muestra_geometrica(numbers, nombre="muestra"):
    numbers = np.asarray(numbers, dtype=int)
    n = len(numbers)
    
    # Parámetro
    p = 1 / np.mean(numbers)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PMF ---
    valores_unicos, frecuencias = np.unique(numbers, return_counts=True)
    probabilidades_empiricas = frecuencias / n
    
    ax1.bar(valores_unicos, probabilidades_empiricas, alpha=0.6, 
            color='mediumpurple', edgecolor='black', label='Frecuencias Empíricas', width=0.6)
    
    x_pmf = np.arange(1, numbers.max() + 1)
    pmf_teorica = p * (1 - p)**(x_pmf - 1)
    
    ax1.plot(x_pmf, pmf_teorica, 'ro-', linewidth=2, markersize=6,
             label=f'PMF Teórica: $P(X=k) = p(1-p)^{{k-1}}$')
    ax1.set_xlabel('k (número de ensayos)', fontsize=12)
    ax1.set_ylabel('Probabilidad', fontsize=12)
    ax1.set_title(f'PMF - Distribución Geométrica ({nombre})\n$p = {p:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    
    ax2.step(numbers_sorted, Fn, where='post', linewidth=2, 
             label='CDF Empírica', color='blue')
    
    # CDF teórica
    x_cdf = np.arange(1, numbers.max() + 1)
    cdf_teorica = 1 - (1 - p)**x_cdf
    ax2.step(x_cdf, cdf_teorica, where='post', color='r', linewidth=2.5,
             label=f'CDF Teórica: $F(k) = 1 - (1-p)^k$')
    
    ax2.set_xlabel('k (número de ensayos)', fontsize=12)
    ax2.set_ylabel('Probabilidad Acumulada', fontsize=12)
    ax2.set_title(f'CDF - Distribución Geométrica ({nombre})\n$p = {p:.4f}$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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

def grafico_gamma(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)

    if bins is None:
        bins = int(np.sqrt(n))

    beta = np.mean(numbers)

    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    x = np.linspace(0, numbers.max(), 300)
    shape, loc, scale = stats.gamma.fit(numbers, floc=0)
    pdf = stats.gamma.pdf(x, shape, loc, scale)
    
    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"Distribución gamma (β = {beta:.3f})")
    plt.show()

def grafico_cdf_exponencial(numbers):
    numbers = np.sort(numbers)
    n = len(numbers)

    beta = np.mean(numbers)

    Fn = np.arange(1, n + 1) / n
    
    F = 1 - np.exp(-numbers / beta)

    plt.figure()
    plt.step(numbers, Fn, where='post', label='Empírica')
    plt.plot(numbers, F, label='Teórica', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title(f"CDF Empírica vs CDF Exponencial (β = {beta:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
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

def cdf_piecewise(x):
    x = np.asarray(x)
    F = np.zeros_like(x, dtype=float)

    F[(0 <= x) & (x <= 1)] = 0.5 * x[(0 <= x) & (x <= 1)]
    F[(1 < x) & (x <= 1.5)] = 0.5
    F[(1.5 < x) & (x <= 1.875)] = np.exp(x[(1.5 < x) & (x <= 1.875)] - 1.5) - 0.5
    F[x > 1.875] = np.exp(1.875 - 1.5) - 0.5

    return F

def pdf_piecewise(x):
    x = np.asarray(x)
    f = np.zeros_like(x, dtype=float)

    f[(0 <= x) & (x <= 1)] = 0.5
    f[(1.5 < x) & (x <= 1.875)] = np.exp(x[(1.5 < x) & (x <= 1.875)] - 1.5)

    return f

def analizar_muestra_piecewise(numbers, nombre="muestra"):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    bins = int(np.sqrt(n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- PDF ---
    ax1.hist(numbers, bins=bins, density=True, alpha=0.6,
             edgecolor='black', label='Histograma')

    x_pdf = np.linspace(0, 2, 400)
    ax1.plot(x_pdf, pdf_piecewise(x_pdf), 'r-', linewidth=2.5,
             label='PDF teórica')

    ax1.set_xlabel('x')
    ax1.set_ylabel('Densidad de Probabilidad')
    ax1.set_title(f'PDF - Función por partes ({nombre})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- CDF ---
    numbers_sorted = np.sort(numbers)
    Fn = np.arange(1, n + 1) / n
    F_teorica = cdf_piecewise(numbers_sorted)

    ax2.step(numbers_sorted, Fn, where='post',
             linewidth=2, label='CDF Empírica')
    ax2.plot(numbers_sorted, F_teorica, 'r-', linewidth=2.5,
             label='CDF Teórica')

    ax2.set_xlabel('x')
    ax2.set_ylabel('Probabilidad Acumulada')
    ax2.set_title(f'CDF - Función por partes ({nombre})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def grafico_geometrica(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)

    if bins is None:
        bins = int(np.sqrt(n))

    beta = np.mean(numbers)

    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    x = np.linspace(0, numbers.max(), 300)
    p = 1 / np.mean(numbers)
    pdf = p * (1 - p)**(x - 1)
    
    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"Distribución Geometrica (β = {beta:.3f})")
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

def grafico_pdf_uniforme(numbers):
    numbers = np.asarray(numbers)
    bins = int(np.sqrt(len(numbers)))

    a = numbers.min()
    b = numbers.max()

    # histograma
    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    # PDF teórica
    x = np.linspace(a, b, 300)
    pdf = np.ones_like(x) / (b - a)

    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"PDF Uniforme U({a:.3f}, {b:.3f})")
    plt.show()

def grafico_pdf_exponencial(numbers):
    numbers = np.asarray(numbers)
    
    bins = int(np.sqrt(len(numbers)))

    beta = np.mean(numbers)

    plt.figure()
    plt.hist(numbers, bins=bins, density=True, alpha=0.5)

    x = np.linspace(0, numbers.max(), 300)
    pdf = (1 / beta) * np.exp(-x / beta)

    plt.plot(x, pdf)
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.title(f"PDF Exponencial (β = {beta:.3f})")
    plt.show()

def grafico_cdf_uniforme(numbers):
    numbers = np.sort(numbers)
    n = len(numbers)

    a = numbers.min()
    b = numbers.max()

    Fn = np.arange(1, n + 1) / n
    F = (numbers - a) / (b - a)

    plt.figure()
    plt.step(numbers, Fn, where='post', label='Empírica')
    plt.plot(numbers, F, label='Teórica')
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("CDF Empírica vs CDF Uniforme")
    plt.legend()
    plt.show()