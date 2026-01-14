import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

from kolmogorov import pruebas_ks
from graficos_distribuciones import *

M8_PATH = "./muestras/m8.txt"
M13_PATH = "./muestras/m13.txt"
M15_PATH = "./muestras/m15.txt"
M16_PATH = "./muestras/m16.txt"

m8 = []
m13 = []
m15 = []
m16 = []

def init_numbers():
    global m8, m13, m15, m16

    with open(M8_PATH, "r") as f:
        m8 = [float(i) for i in f.read().splitlines()]

    with open(M13_PATH, "r") as f:
        m13 = [int(i) for i in f.read().splitlines()]

    with open(M15_PATH, "r") as f:
        m15 = [float(i) for i in f.read().splitlines()]

    with open(M16_PATH, "r") as f:
        m16 = [float(i) for i in f.read().splitlines()]


import numpy as np
import scipy.stats as stats


def prueba_chi_cuadrado(datos, nombre_muestra):
    datos = np.array(datos)
    N = len(datos)
    nombre_lower = nombre_muestra.lower()

    print(f"\n--- ANALIZANDO: {nombre_muestra} ---")  # Título para separar salidas

    # Variables de configuración
    dist_nom = ""
    k_params = 0
    cdf = None

    # --- LÓGICA DE DISTRIBUCIONES ---
    if "m15" in nombre_lower:
        dist_nom = "Función por partes"
        cdf = lambda x: cdf_piecewise(x)
        k_params = 0

        # M15 no estima parámetros, pero podemos ver sus estadísticas básicas
        print(f"  [Info] Tipo: {dist_nom}")
        print(f"  [Datos] Media Muestral: {np.mean(datos):.4f}")
        print(f"  [Datos] Varianza Muestral: {np.var(datos):.4f}")
        print(f"  [Params] No se estiman parámetros (fijos en cdf_piecewise).")

    elif "m16" in nombre_lower:
        dist_nom = "Gamma"

        media = np.mean(datos)
        var = np.var(datos)
        std = np.std(datos)

        alpha_est = (media / std) ** 2
        scale_est = var / media

        cdf = lambda x: stats.gamma.cdf(x, a=alpha_est, loc=0, scale=scale_est)
        k_params = 2

        # IMPRIMIR VARIABLES M16
        print(f"  [Info] Tipo: {dist_nom}")
        print(f"  [Datos] Media: {media:.4f}")
        print(f"  [Datos] Varianza: {var:.4f}")
        print(f"  [Params] Alpha estimado (forma): {alpha_est:.4f}")
        print(f"  [Params] Scale estimado (escala): {scale_est:.4f}")

    elif "m8" in nombre_lower:
        dist_nom = "Normal"
        mu, sigma = np.mean(datos), np.std(datos, ddof=1)
        cdf = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
        k_params = 2

        # IMPRIMIR VARIABLES M8
        print(f"  [Info] Tipo: {dist_nom}")
        print(f"  [Params] Mu (Media): {mu:.4f}")
        print(f"  [Params] Sigma (Desv. Std): {sigma:.4f}")

    elif "m13" in nombre_lower:
        dist_nom = "Geométrica"

        conteo_999 = np.sum(datos == 999)
        N_original = N
        N = N - conteo_999

        datos_validos = datos[datos != 999]
        media_valida = np.mean(datos_validos)
        p = 1 / media_valida

        cdf = lambda x: stats.geom.cdf(x, p=p, loc=0)
        k_params = 1

        # IMPRIMIR VARIABLES M13
        print(f"  [Info] Tipo: {dist_nom}")
        print(f"  [Ajuste] 999s ignorados: {conteo_999}")
        print(f"  [Ajuste] N ajustado: {N} (de {N_original})")
        print(f"  [Datos] Media (sin 999): {media_valida:.4f}")
        print(f"  [Params] p estimado: {p:.4f}")

    else:
        print(f"Distribución no identificada para {nombre_muestra}.")
        return

    # --- RESTO DEL CÓDIGO (Histograma y Chi2) ---
    num_bins = 7
    obs, bordes = np.histogram(datos, bins=num_bins)

    if "m13" in nombre_lower:
        obs[-1] = obs[-1] - conteo_999
        if obs[-1] < 0: obs[-1] = 0

    chi_total = 0

    for i in range(num_bins):
        lim_inf = bordes[i]
        lim_sup = bordes[i + 1]

        Oi = obs[i]

        Pe = cdf(lim_sup) - cdf(lim_inf)
        if Pe <= 0: Pe = 1e-9
        Ei = N * Pe

        if Ei > 0:
            chi_parcial = ((Oi - Ei) ** 2) / Ei
        else:
            chi_parcial = 0.0

        chi_total += chi_parcial


    gl = max(1, num_bins - 1 - k_params)
    alpha = 0.05
    valor_critico = stats.chi2.ppf(1 - alpha, gl)
    p_valor = stats.chi2.sf(chi_total, gl)

    print("-" * 60)
    print(f"RESUMEN FINAL {nombre_muestra}:")
    print(f"  Grados Libertad (gl): {gl}")
    print(f"  Chi2 Calculado:       {chi_total:.4f}")
    print(f"  Chi2 Crítico:         {valor_critico:.4f}")
    print(f"  P-Valor:              {p_valor:.4e}")



def main():
    init_numbers()
    if not m8: return

    muestras = [
        (m15, "M15", False),  # Uniforme
        (m16, "M16", False),  # GAMMA
        (m8, "M8", False),  # Normal
        (m13, "M13", True)  # Geometrica
    ]

    for datos, nombre, _ in muestras:
        prueba_chi_cuadrado(datos, nombre)

    for datos, nombre, discreta in muestras:
        pruebas_ks(datos, nombre, discreta)

    #grafico_normal(m8)

    #grafico_exponencial(m16)
    #grafico_gamma(m16)

    #grafico_exponencial(m13)
    #grafico_cdf_exponencial(m13)
    #grafico_geometrica(m13)
    #grafico_poisson(m8)

    #grafico_pdf_uniforme(m15)
    #grafico_cdf_uniforme(m15)

    analizar_muestra_piecewise(m15,'m15')
    analizar_muestra_gamma(m16, 'm16')
    analizar_muestra_normal(m8, 'm8')
    analizar_muestra_exponencial(m13, 'm13')
    analizar_muestra_geometrica(m13, 'm13')
    analizar_muestra_uniforme(m15, 'm15')

if __name__ == '__main__':
    main()