import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt 

# continuos

def generador(numbers):
    n = len(numbers)
    sorted_numbers = sorted(numbers)
    # CDF empírica
    fn = [(i+1)/n for i in range(n)]
    
    # CDF teórica
    f = sorted_numbers
    
    d = max(abs(fn[i] - f[i]) for i in range(n))
    
    return d

def exponencial_continua(numbers, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))

    media = np.mean(numbers)
    
    fo, bordes = np.histogram(numbers, bins=bins)

    foa = np.cumsum(fo)
    poa = foa / n
    
    pea = [stats.expon.cdf(bordes[i+1], scale=media) for i in range(len(bordes)-1)]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'Intervalo': [f"[{bordes[i]:.3f}, {bordes[i+1]:.3f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa,
        'pea': pea,
        '|poa-pea|': diferencias
    })
    
    return d, tabla, media

def uniforme_continua(numbers, a=None, b=None, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    if a is None:
        a = min(numbers)
    if b is None:
        b = max(numbers)
    
    min_val = min(numbers)
    max_val = max(numbers)
    amplitud = (max_val - min_val) / bins
    
    bordes = [min_val + i * amplitud for i in range(bins + 1)]
    bordes[-1] = max_val + 0.01
    
    fo, _ = np.histogram(numbers, bins=bordes)
    
    foa = np.cumsum(fo)
    
    poa = foa / n
    
    pea = [(bordes[i+1] - a) / (b - a) for i in range(len(bordes)-1)]

    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'Intervalo': [f"[{bordes[i]:.3f}, {bordes[i+1]:.3f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa,
        'pea': pea,
        '|poa-pea|': diferencias
    })
    
    return d, tabla, a, b

def normal_continua(numbers, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    mu = np.mean(numbers)
    sigma = np.std(numbers, ddof=1)
    
    fo, bordes = np.histogram(numbers, bins=bins)
    
    foa = np.cumsum(fo)
    
    poa = foa / n
    
    pea = [stats.norm.cdf(bordes[i+1], mu, sigma) for i in range(len(bordes)-1)]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'Intervalo': [f"[{bordes[i]:.3f}, {bordes[i+1]:.3f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa,
        'pea': pea,
        '|poa-pea|': diferencias
    })
    
    return d, tabla, mu, sigma

# discretos

def exponencial_discreta(numbers, bins=None, beta=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    if beta is None:
        beta = np.mean(numbers)

    min_val = min(numbers)
    max_val = max(numbers)
    amplitud = (max_val - min_val) / bins
    
    bordes = [min_val + i * amplitud for i in range(bins + 1)]
    bordes[-1] = max_val + 0.01
    
    fo, _ = np.histogram(numbers, bins=bordes)
    foa = np.cumsum(fo)

    poa = foa / n    
    pea = [1 - np.exp(-bordes[i+1]/beta) for i in range(len(bordes)-1)]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'No.': range(1, bins + 1),
        'Intervalo': [f"[{bordes[i]:7.2f}, {bordes[i+1]:7.2f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': [round(x, 4) for x in pea],
        '|poa-pea|': [round(x, 4) for x in diferencias]
    })
    
    return d, tabla, beta, bordes

def uniforme_discreta(numbers, a=None, b=None):
    n = len(numbers)
    fo_dict = Counter(numbers)
    valores = sorted(fo_dict.keys())
    
    if a is None:
        a = min(valores)
    if b is None:
        b = max(valores)
    
    fo = [fo_dict[v] for v in valores]
    foa = np.cumsum(fo)

    poa = foa / n    
    pea = [(v - a + 1) / (b - a + 1) for v in valores]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'x': valores,
        'fo': fo,
        'foa': foa,
        'poa': poa,
        'pea': pea,
        '|poa-pea|': diferencias
    })
    
    return d, tabla, a, b

def poisson_discreta(numbers):
    n = len(numbers)
    fo_dict = Counter(numbers)
    valores = sorted(fo_dict.keys())
    
    lambda_ = np.mean(numbers)
    
    fo = [fo_dict[v] for v in valores]
    foa = np.cumsum(fo)
    
    poa = foa / n
    pea = [stats.poisson.cdf(v, lambda_) for v in valores]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'x': valores,
        'fo': fo,
        'foa': foa,
        'poa': poa,
        'pea': pea,
        '|poa-pea|': diferencias
    })
    
    return d, tabla, lambda_

def pruebas_ks(numbers, muestra_id, tipo):
    print("-" * 70)
    print(f"Pruebas Kolmogorov - {muestra_id}")
    print("-" * 70)
    
    n = len(numbers)
    critico = 1.36 / sqrt(n)
    
    print(f"Tamaño de muestra (n): {n}")
    print(f"Valor crítico d(0.05, {n}): {critico:.4f}")
    print(f"Media: {np.mean(numbers):.4f}")
    print(f"Desviación estándar: {np.std(numbers, ddof=1):.4f}")
    print()

    
    resultados = {}
    
    if tipo:
        print("-" * 70)
        print("Distribuciones continuas")
        print("-" * 70)
        
        print("\n1. Distribucion normal")
        d, tabla, mu, sigma = normal_continua(numbers)
        print(f"   Parámetros: μ = {mu:.4f}, σ = {sigma:.4f}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        print(f"\n   Tabla de intervalos:")
        print(tabla.to_string(index=False))
        resultados['normal'] = {'d': d, 'acepta': d < critico}
        
        print("\n\n2. Distribucion exponencial")
        d, tabla, media = exponencial_continua(numbers)
        print(f"   Parámetro: β (media) = {media:.4f}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        print(f"\n   Tabla de intervalos:")
        print(tabla.to_string(index=False))
        resultados['exponencial'] = {'d': d, 'acepta': d < critico}
        
        print("\n\n3. Distribucion Uniforme")
        d, tabla, a, b = uniforme_continua(numbers)
        print(f"   Parámetros: a = {a:.4f}, b = {b:.4f}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        print(f"\n   Tabla de intervalos:")
        print(tabla.to_string(index=False))
        resultados['uniforme_continua'] = {'d': d, 'acepta': d < critico}
        
        print("\n\n4. Generador aleatorio")
        d = generador(numbers)
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        resultados['generador'] = {'d': d, 'acepta': d < critico}
        
    else:  # discreta
        print("-" * 70)
        print("Distribuciones discretas")
        print("-" * 70)
        
        print("\n1. Distribucion Poisson")
        d, tabla, lambda_ = poisson_discreta(numbers)
        print(f"   Parámetro: λ = {lambda_:.4f}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        print(f"\n   Tabla de frecuencias:")
        print(tabla.to_string(index=False))
        resultados['poisson'] = {'d': d, 'acepta': d < critico}

        print("\n1. DISTRIBUCIÓN EXPONENCIAL")
        d, tabla, beta, _ = exponencial_discreta(numbers)
        print(f"   Parámetro: β = {beta:.4f}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico}")
        print(f"\n   Tabla:")
        print(tabla.to_string(index=False))
        resultados['exponencial'] = {'d': d, 'acepta': d < critico}
        
        print("\n\n2. Distribucion Uniforme")
        d, tabla, a, b = uniforme_discreta(numbers)
        print(f"   Parámetros: a = {a}, b = {b}")
        print(f"   D = {d:.4f}")
        print(f"   Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
        print(f"\n   Tabla de frecuencias:")
        print(tabla.to_string(index=False))
        resultados['uniforme_discreta'] = {'d': d, 'acepta': d < critico}
    
    print("\n" + "="*70)
    print("Resumen")
    print("="*70)
    for dist, res in resultados.items():
        simbolo = "✓" if res['acepta'] else "✗"
        print(f"{simbolo} {dist:20s}: D = {res['d']:.4f} - {'Acepta H0' if res['acepta'] else 'Rechaza H0'}")
    
    if resultados:
        mejor = min(resultados.items(), key=lambda x: x[1]['d'])
        print(f"\nMejor acercamiento: {mejor[0].upper()} (D = {mejor[1]['d']:.4f})")
    
    return resultados