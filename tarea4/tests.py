import scipy.stats as stats
import numpy as np

from kolmogorov import pruebas_ks, grafico_poisson

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


def prueba_chi_cuadrado(datos, nombre_muestra):
    datos = np.array(datos)
    N = len(datos)
    nombre_lower = nombre_muestra.lower()

    if "m15" in nombre_lower:
        dist_nom = "Uniforme"
        a, b = np.min(datos), np.max(datos) - np.min(datos)
        cdf = lambda x: stats.uniform.cdf(x, loc=a, scale=b)
        k_params = 2

    elif "m16" in nombre_lower:
        dist_nom = "Exponencial"
        media = np.mean(datos)
        cdf = lambda x: stats.expon.cdf(x, scale=media)
        k_params = 1

    elif "m8" in nombre_lower:
        dist_nom = "Normal"
        mu, sigma = np.mean(datos), np.std(datos, ddof=1)
        cdf = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
        k_params = 2

    elif "m13" in nombre_lower:
        dist_nom = "Geométrica"
        # Geometrica p = 1/media
        p = 1 / np.mean(datos)
        cdf = lambda x: stats.geom.cdf(x, p=p, loc=0)
        k_params = 1
    else:
        print("Distribución no identificada.")
        return

    # Generar 7 Intervalos de igual amplitud
    num_bins = 7
    obs, bordes = np.histogram(datos, bins=num_bins)

    # Calcular Chi-Cuadrado
    chi_total = 0

    for i in range(num_bins):
        lim_inf = bordes[i]
        lim_sup = bordes[i + 1]

        Oi = obs[i]

        #PE:
        Pe = cdf(lim_sup) - cdf(lim_inf)

        #evitar P=0 en colas extremas
        if Pe <= 0: Pe = 1e-9

        Ei = N * Pe

        # Chi Parcial
        chi_parcial = ((Oi - Ei) ** 2) / Ei
        chi_total += chi_parcial

    # Calcular P-Valor
    gl = num_bins - 1 - k_params  # Grados de libertad
    alpha = 0.05
    valor_critico = stats.chi2.ppf(1 - alpha, gl)
    p_valor = 1 - stats.chi2.cdf(chi_total, gl)

    print(f"RESUMEN CH2, {nombre_muestra}:")
    print(f"  Chi2 Calc:   {chi_total:.4f}")
    print(f"  Chi2 Crít:   {valor_critico:.4f}")
    print(f"  P-Valor:     {p_valor}")


def main():
    init_numbers()
    if not m8: return

    muestras = [
        (m15, "M15", False),  # Uniforme
        (m16, "M16", False),  # Exponencial
        (m8, "M8", False),  # Normal
        (m13, "M13", True)  # Geometrica
    ]

    for datos, nombre, _ in muestras:
        prueba_chi_cuadrado(datos, nombre)
    
    for datos, nombre, discreta in muestras:
        pruebas_ks(datos, nombre, discreta)
        grafico_poisson(datos)


if __name__ == '__main__':
    main()