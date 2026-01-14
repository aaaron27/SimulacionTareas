import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt
from collections import Counter

def generador(numbers):
    n = len(numbers)
    sorted_numbers = sorted(numbers)
    
    fn = [(i+1)/n for i in range(n)]
    
    diferencias = [abs(sorted_numbers[i] - fn[i]) for i in range(n)]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'i': range(1, n + 1),
        'x(i)': [round(x, 4) for x in sorted_numbers],
        'i/n': [round(x, 4) for x in fn],
        '|x(i) - i/n|': [round(x, 4) for x in diferencias]
    })
    
    return d, tabla

def ks_exponencial(numbers, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))

    beta = np.mean(numbers)
    
    min_val = min(numbers)
    max_val = max(numbers)
    amplitud = (max_val - min_val) / bins
    
    bordes = [min_val + i * amplitud for i in range(bins + 1)]
    bordes[-1] = max_val + 0.01
    
    fo, _ = np.histogram(numbers, bins=bordes)

    n -= fo[-1]
    fo = fo[:-1]
    bordes = bordes[:-1]

    foa = np.cumsum(fo)
    poa = foa / n
    
    pea = [1 - np.exp(-bordes[i+1]/beta) for i in range(len(bordes)-1)]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    
    d, p = stats.kstest(numbers, 'expon', args=(0, beta))
    
    tabla = pd.DataFrame({
        'No.': range(1, len(fo) + 1),
        'Intervalo': [f"[{bordes[i]:7.2f}, {bordes[i+1]:7.2f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': [round(x, 4) for x in pea],
        '|poa-pea|': [round(x, 4) for x in diferencias]
    })
    
    return d, tabla, beta, p

def ks_uniforme(numbers, a=None, b=None, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    if a is None:
        a = min(numbers)
    if b is None:
        b = max(numbers)
    
    bordes = np.linspace(a, b, bins + 1)
    
    fo, _ = np.histogram(numbers, bins=bordes)

    foa = np.cumsum(fo)
    poa = foa / n

    calc = b - a
    pea = (bordes[1:] - a) / calc

    diff = np.abs(poa - pea)
    d, p = stats.kstest(numbers, 'uniform')

    tabla = pd.DataFrame({
        'Intervalo': [
            f"[{bordes[i]:.4f}, {bordes[i+1]:.4f})"
            for i in range(bins)
        ],
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': pea.round(4),
        '|poa-pea|': diff.round(4)
    })

    return d, tabla, a, b, p

def ks_poisson_discreta(numbers):
    n = len(numbers)
    lambda_ = np.mean(numbers)

    fo_dict = Counter(numbers)
    values = np.array(sorted(fo_dict.keys()))

    fo = np.array([fo_dict[i] for i in values])
    foa = np.cumsum(fo)
    poa = foa / n

    pea = stats.poisson.cdf(values, lambda_)

    diff = np.abs(poa - pea)
    d = diff.max()

    tabla = pd.DataFrame({
        'x': values,
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': pea.round(4),
        '|poa-pea|': diff.round(4)
    })

    return d, tabla, lambda_

def ks_poisson(numbers, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    lambda_ = np.mean(numbers)
    
    min_val = min(numbers)
    max_val = max(numbers)
    amplitud = (max_val - min_val) / bins
    
    bordes = [min_val + i * amplitud for i in range(bins + 1)]
    bordes[-1] = max_val + 0.01
    
    fo, _ = np.histogram(numbers, bins=bordes)

    foa = np.cumsum(fo)
    poa = foa / n
    
    pea = [stats.poisson.cdf(int(np.floor(bordes[i+1])), lambda_) for i in range(len(bordes)-1)]
    
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
    
    return d, tabla, lambda_

def ks_normal(numbers, bins=None):
    numbers = np.asarray(numbers, dtype=float)
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    mu = np.mean(numbers)
    sigma = np.std(numbers, ddof=1)
    
    min_val = min(numbers)
    max_val = max(numbers)
    bordes = np.linspace(min_val, max_val, bins + 1)
    
    fo, _ = np.histogram(numbers, bins=bordes)
    
    foa = np.cumsum(fo)
    poa = foa / n
    
    pea = stats.norm.cdf(bordes[1:], loc=mu, scale=sigma)
    
    diff = np.abs(poa - pea)
    d = diff.max()
    
    d_scipy, p = stats.kstest(numbers, 'norm', args=(mu, sigma))
    
    tabla = pd.DataFrame({
        'No.': range(1, bins + 1),
        'Intervalo': [
            f"[{bordes[i]:7.4f}, {bordes[i+1]:7.4f})"
            for i in range(bins)
        ],
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': pea.round(4),
        '|poa-pea|': diff.round(4)
    })
    
    return d, tabla, mu, sigma, p

def ks_geometrica(numbers, bins=None):
    n = len(numbers)
    
    if bins is None:
        bins = int(np.sqrt(n))
    
    media = np.mean(numbers)
    p_param = 1 / media
    
    min_val = min(numbers)
    max_val = max(numbers)
    amplitud = (max_val - min_val) / bins
    
    bordes = [min_val + i * amplitud for i in range(bins + 1)]
    bordes[-1] = max_val + 0.01
    
    fo, _ = np.histogram(numbers, bins=bordes)
    
    n -= fo[-1]
    fo = fo[:-1]
    bordes = bordes[:-1]
 
    foa = np.cumsum(fo)
    poa = foa / n
    
    pea = [stats.geom.cdf(int(np.floor(bordes[i+1])), p_param) for i in range(len(bordes)-1)]
    
    diferencias = [abs(poa[i] - pea[i]) for i in range(len(poa))]
    d = max(diferencias)
    
    tabla = pd.DataFrame({
        'No.': range(1, len(fo) + 1),
        'Intervalo': [f"[{bordes[i]:7.2f}, {bordes[i+1]:7.2f})" for i in range(len(bordes)-1)],
        'fo': fo,
        'foa': foa,
        'poa': poa.round(4),
        'pea': [round(x, 4) for x in pea],
        '|poa-pea|': [round(x, 4) for x in diferencias]
    })
    
    return d, tabla, p_param

def pruebas_ks(numbers, muestra_id, discreta):
    print("=" * 70)
    print(f"Pruebas Kolmogorov-Smirnov - {muestra_id}")
    print("=" * 70)
    
    n = len(numbers)
    critico = 1.36 / sqrt(n)
    
    print(f"\nTamaño de muestra (n): {n}")
    print(f"Valor crítico d(0.05, {n}): {critico:.4f}")
    print(f"Media: {np.mean(numbers):.4f}")
    print(f"Desviación estándar: {np.std(numbers, ddof=1):.4f}")
    print(f"Número de intervalos: {int(np.sqrt(n))}")
    print()
    
    resultados = {}
    
    print("\n" + "-" * 70)
    print("2. Distribucion exponencial")
    print("-" * 70)
    d, tabla, beta, p = ks_exponencial(numbers)
    print(f"Parámetro: β (media) = {beta:.4f}")
    print(f"D = {d:.4f}")
    print(f"Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f}) {p} >= 0.05")
    print(f"p-valor = {p:.4f}")
    print(f"Acepta H0 (p-valor): {p >= 0.05}")
    print(f"Acepta H0 (valor crítico): {d < critico}")
    print("\nTabla de intervalos:")
    print(tabla.to_string(index=False))
    resultados['exponencial'] = {'d': d, 'acepta': d < critico}
    
    print("\n" + "-" * 70)
    print("3. Distribucion Uniforme")
    print("-" * 70)
    d, tabla, a, b, p = ks_uniforme(numbers)
    print(f"Parámetros: a = {a:.4f}, b = {b:.4f}")
    print(f"D = {d:.4f}")
    print(f"Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f}) {p} >= 0.05")
    print(f"p-valor = {p:.4f}")
    print(f"Acepta H0 (p-valor): {p >= 0.05}")
    print(f"Acepta H0 (valor crítico): {d < critico}")
    print("\nTabla de intervalos:")
    print(tabla.to_string(index=False))
    resultados['uniforme'] = {'d': d, 'acepta': d < critico}
    
    print("\n" + "-" * 70)
    print("4. Distribucion Poisson")
    print("-" * 70)
    d, tabla, lambda_ = ks_poisson_discreta(numbers) if discreta else ks_poisson(numbers)
    print(f"Parámetro: λ = {lambda_:.4f}")
    print(f"D = {d:.4f}")
    print(f"Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
    print("\nTabla de intervalos:")
    print(tabla.to_string(index=False))
    resultados['poisson'] = {'d': d, 'acepta': d < critico}
    
    print("\n" + "-" * 70)
    print("5. Generador Aleatorio")
    print("-" * 70)
    d, tabla = generador(numbers)
    print(f"D = {d}")
    print(f"Acepta H0: {d < critico} (d {'<' if d < critico else '>='} {critico:.4f})")
    print("\nTabla de intervalos:")
    #print(tabla.to_string(index=False))
    resultados['generador'] = {'d': d, 'acepta': d < critico}
    
    d, tabla, mu, sigma, p = ks_normal(numbers)
    print("=" * 70)
    print("Prueba Kolmogorov-Smirnov - Distribución Normal")
    print("=" * 70)
    print(f"\nTamaño de muestra (n): {n}")
    print(f"Valor crítico d(0.05, {n}): {critico:.4f}")
    print(f"\nParámetros estimados:")
    print(f"  μ (media) = {mu:.4f}")
    print(f"  σ (desv. estándar) = {sigma:.4f}")
    print(f"\nEstadístico D = {d:.4f}")
    print(f"P-valor = {p:.4f}")
    print(f"\nDecisión:")
    print(f"  Acepta H0 (valor crítico): {d < critico} (D {'<' if d < critico else '>='} {critico:.4f})")
    print(f"  Acepta H0 (p-valor): {p >= 0.05} (p {'≥' if p >= 0.05 else '<'} 0.05)")
    print("\nTabla de intervalos:")
    print(tabla.to_string(index=False))

    d2, tabla2, p_estimado2 = ks_geometrica(numbers)
    
    print(f"\nTamaño de muestra (n): {n}")
    print(f"Valor crítico d(0.05, {n}): {critico:.4f}")
    print(f"Número de intervalos: {int(np.sqrt(n))}")
    print(f"\nParámetro estimado:")
    print(f"  p = {p_estimado2:.4f}")
    print(f"\nEstadístico D = {d2:.4f}")
    print(f"\nDecisión:")
    print(f"  Acepta H0: {d2 < critico} (D {'<' if d2 < critico else '>='} {critico:.4f})")
    print("\nTabla de intervalos:")
    print(tabla2.to_string(index=False))

    print("\n" + "=" * 70)
    print("Resumen")
    print("=" * 70)
    for dist, res in resultados.items():
        simbolo = "✓" if res['acepta'] else "✗"
        print(f"{simbolo} {dist:20s}: D = {res['d']:.4f} - {'ACEPTA H0' if res['acepta'] else 'RECHAZA H0'}")
    
    if resultados:
        mejor = min(resultados.items(), key=lambda x: x[1]['d'])
        print(f"\nMEJOR AJUSTE: {mejor[0].upper()} (D = {mejor[1]['d']:.4f})")
    
    return resultados