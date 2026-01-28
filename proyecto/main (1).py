import numpy as np
from math import inf, sqrt, exp, pi
from dataclasses import dataclass
from random import random
import pandas as pd
from numpy import floating
from pandas import DataFrame
from scipy import stats
from simpson import simpson
import matplotlib.pyplot as plt

COSTOS = {
    'cajas': 500,
    'refrescos': 750,
    'freidora': 200,
    'pollo': 100
}

PROBABILIDADES = [0.9, 0.7, 0.3]

PATH_MUESTRA_POISSON = 'data/muestra_poisson.txt'

REPETICIONES_TOTALES = 10

MEDIA_CAJAS = 2.5
MEDIA_REFRESCOS = 0.75
MEDIA_FREIDORA = 3
MEDIA_POLLO = 10

@dataclass
class Config:
    cajas: int
    refrescos: int
    freidora: int
    pollo: int
    costo: int = 0

def realizar_pruebas(nombre, muestras, dist_teorica, params):
    print(f"\n--- {nombre.upper()} ---")

    _, ks_p = stats.ks_1samp(muestras, dist_teorica(*params).cdf)
    print(f"KS Test: p-valor = {ks_p:.4f} -> {'ACEPTADO' if ks_p > 0.05 else 'RECHAZADO'}")

    obs, edges = np.histogram(muestras, bins=10)

    cdf_val = dist_teorica(*params).cdf(edges)
    prob_esperada = np.diff(cdf_val)

    prob_esperada = prob_esperada / np.sum(prob_esperada)
    esperados = prob_esperada * len(muestras)

    _, chi_p = stats.chisquare(obs, f_exp=esperados)
    print(f"Chi2 Test: p-valor = {chi_p:.4f} -> {'ACEPTADO' if chi_p > 0.05 else 'RECHAZADO'}")

def validar_distribuciones():
    N = 1000

    muestras_cajas = [get_random_tiempo_cajas() for _ in range(N)]
    realizar_pruebas("Cajas", muestras_cajas, stats.expon, (0, MEDIA_CAJAS))

    muestras_ref = [get_random_tiempo_refrescos() for _ in range(N)]
    realizar_pruebas("Refrescos", muestras_ref, stats.expon, (0, MEDIA_REFRESCOS))

    muestras_frei = [get_random_tiempo_freidora() for _ in range(N)]
    realizar_pruebas("Freidora", muestras_frei, stats.norm, (3, 1))

    muestras_pollo = [get_random_tiempo_pollo() for _ in range(N)]
    realizar_pruebas("Pollo", muestras_pollo, stats.geom, (0.1,))

    muestras_llegadas = np.loadtxt(PATH_MUESTRA_POISSON).astype(int).tolist()
    realizar_pruebas("Llegadas", muestras_llegadas, stats.poisson, (3,))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

def normal_pdf(x, mu, sigma):
    part1 = 1 / (sigma * sqrt(2 * pi))
    part2 = exp(-0.5 * ((x - mu) / sigma)**2)
    return part1 * part2

def normal_cdf(x, mu=3, sigma=1):
    return simpson(lambda t: normal_pdf(t, mu, sigma), mu - 5 * sigma, x, n=50)

def get_cant_ordenes(n: int, p: float):
    return np.random.binomial(n, p)

def get_random_tiempo_cajas():
    return np.random.exponential(MEDIA_CAJAS)

def get_random_tiempo_refrescos():
    return np.random.exponential(MEDIA_REFRESCOS)

def get_random_tiempo_freidora():
    u = random()
    low, high = 0, 10
    for _ in range(15):
        mid = (low + high) / 2
        if normal_cdf(mid) < u:
            low = mid
        else:
            high = mid
    return max(1, low)

def get_random_tiempo_pollo():
    return np.random.geometric(p=0.1)

def get_estaciones():
    return [random() <= p for p in PROBABILIDADES]

def generar_configuraciones(presupuesto: int) -> list[Config]:
    configuraciones = []
    for c in range(1, (presupuesto // COSTOS['cajas']) + 1):
        for r in range(1, (presupuesto // COSTOS['refrescos']) + 1):
            for f in range(1, (presupuesto // COSTOS['freidora']) + 1):
                for p in range(1, (presupuesto // COSTOS['pollo']) + 1):

                    costo_total = (c * COSTOS['cajas'] +
                                   r * COSTOS['refrescos'] +
                                   f * COSTOS['freidora'] +
                                   p * COSTOS['pollo'])

                    if costo_total <= presupuesto:
                        configuraciones.append(Config(
                            cajas=c,
                            refrescos=r,
                            freidora=f,
                            pollo=p,
                            costo=costo_total
                        ))
    return configuraciones

def calc_hora_llegada() -> list[int]:
    minutos_limite = 480 # 8h = 480min
    c = 0
    hora_llegada = []
    while (c < minutos_limite):
        c += np.random.poisson(3)

        if c > minutos_limite:
            break

        hora_llegada.append(c)

    return hora_llegada

def get_servidor_disponible(servidores: list[float], tiempo: float) -> int:
    min_value = inf
    min_index = -1
    for i, time in enumerate(servidores):
        if time <= min_value:
            min_index = i
            min_value = time
        if tiempo >= time:
            return i
    return min_index

def etapa_1(n_cajas: int):
    hora_llegada = calc_hora_llegada()
    servidores_cajas_disponibles = [0]*n_cajas
    servidor_usado = [0]*len(hora_llegada)
    hora_inicio_atencion = [0]*len(hora_llegada)
    tiempo_atencion = [0]*len(hora_llegada)
    tiempo_sistema1 = [0]*len(hora_llegada)
    hora_fin = [0]*len(hora_llegada)

    # primer fila
    aux_cajas = get_random_tiempo_cajas()
    tiempo_atencion[0] = aux_cajas
    hora_inicio_atencion[0] = hora_llegada[0]
    tiempo_sistema1[0] = aux_cajas
    servidor_usado[0] = 0
    hora_fin[0] = tiempo_sistema1[0] + hora_inicio_atencion[0]
    servidores_cajas_disponibles[0] = hora_fin[0]

    for i in range(1, len(hora_llegada)):
        tiempo_atencion[i] = get_random_tiempo_cajas()
        servidor_usado[i] = get_servidor_disponible(servidores_cajas_disponibles, hora_llegada[i])

        hora_inicio_atencion[i] = max(
            servidores_cajas_disponibles[servidor_usado[i]],
            hora_llegada[i]
        )

        servidores_cajas_disponibles[servidor_usado[i]] = hora_inicio_atencion[i] + tiempo_atencion[i]
        hora_fin[i] = hora_llegada[i] + tiempo_atencion[i]
        tiempo_sistema1[i] = hora_fin[i] - hora_llegada[i]

    df = pd.DataFrame({
        "Hora llegada": hora_llegada,
        "Hora inicio atención": hora_inicio_atencion,
        "Servidor usado": servidor_usado,
        "Tiempo atención": tiempo_atencion,
        "Tiempo sistema": tiempo_sistema1,
        "Hora fin": hora_fin
    })

    return hora_fin, tiempo_sistema1, df

def etapa_2(permutacion, hora_llegada: list[float], tiempo_sis_1: list[float]):    
    grupos = []
    
    for i in range(len(hora_llegada)):
        cant_personas = get_cant_ordenes(5, 0.4)
        
        if cant_personas == 0:
            continue
        
        gustos = get_estaciones()
        ordenes_grupo = []
        
        for gusto_idx, tiene_gusto in enumerate(gustos):
            if tiene_gusto:
                for _ in range(cant_personas):
                    ordenes_grupo.append(gusto_idx)
        
        if ordenes_grupo:
            grupos.append({
                'grupo_id': i,
                'hora_llegada': hora_llegada[i],
                'tiempo_sis_1': tiempo_sis_1[i],
                'ordenes': ordenes_grupo,
                'cant_personas': cant_personas
            })
    
    servidores_refrescos = [0] * permutacion.refrescos
    servidores_freidoras = [0] * permutacion.freidora
    servidores_pollos = [0] * permutacion.pollo
    
    tiempos_refrescos = []
    tiempos_freidoras = []
    tiempos_pollos = []
    
    resultados_grupos = []
    
    for grupo in grupos:
        hora_llegada_grupo = grupo['hora_llegada']
        ordenes = grupo['ordenes']
        
        tiempos_finalizacion = []
        
        for orden in ordenes:
            match orden:
                case 0:  # Refrescos
                    tiempo_servicio = get_random_tiempo_refrescos()
                    servidor_idx = get_servidor_disponible(servidores_refrescos, hora_llegada_grupo)
                    
                    hora_inicio = max(servidores_refrescos[servidor_idx], hora_llegada_grupo)
                    hora_fin = hora_inicio + tiempo_servicio
                    servidores_refrescos[servidor_idx] = hora_fin
                    tiempos_finalizacion.append(hora_fin)
                    tiempos_refrescos.append(tiempo_servicio)
                
                case 1:  # Freidora
                    tiempo_servicio = get_random_tiempo_freidora()
                    servidor_idx = get_servidor_disponible(servidores_freidoras, hora_llegada_grupo)
                    
                    hora_inicio = max(servidores_freidoras[servidor_idx], hora_llegada_grupo)
                    hora_fin = hora_inicio + tiempo_servicio
                    servidores_freidoras[servidor_idx] = hora_fin
                    tiempos_finalizacion.append(hora_fin)
                    tiempos_freidoras.append(tiempo_servicio)
                
                case 2:  # Pollo
                    tiempo_servicio = get_random_tiempo_pollo()
                    servidor_idx = get_servidor_disponible(servidores_pollos, hora_llegada_grupo)
                    
                    hora_inicio = max(servidores_pollos[servidor_idx], hora_llegada_grupo)
                    hora_fin = hora_inicio + tiempo_servicio
                    servidores_pollos[servidor_idx] = hora_fin
                    tiempos_finalizacion.append(hora_fin)
                    tiempos_pollos.append(tiempo_servicio)
        
        hora_salida_grupo = max(tiempos_finalizacion) if tiempos_finalizacion else hora_llegada_grupo
        tiempo_total_grupo = (hora_salida_grupo - hora_llegada_grupo) + grupo['tiempo_sis_1']
        
        resultados_grupos.append({
            'grupo_id': grupo['grupo_id'],
            'cant_personas': grupo['cant_personas'],
            'hora_llegada': hora_llegada_grupo,
            'hora_salida': hora_salida_grupo,
            'cant_ordenes': len(ordenes),
            'tiempo_total': tiempo_total_grupo
        })
    
    df = pd.DataFrame(resultados_grupos)
    
    if len(resultados_grupos) > 0:
        tiempos_totales = [g['tiempo_total'] for g in resultados_grupos]
        media_tiempo_total = np.mean(tiempos_totales)
    else:
        tiempos_totales = []
        media_tiempo_total = 0
    
    # Calcular medias por estación
    media_refrescos = np.mean(tiempos_refrescos) if tiempos_refrescos else 0
    media_freidoras = np.mean(tiempos_freidoras) if tiempos_freidoras else 0
    media_pollos = np.mean(tiempos_pollos) if tiempos_pollos else 0
    
    return (tiempos_totales, df, media_tiempo_total, 
            media_refrescos, media_freidoras, media_pollos)

def simular(permutation: Config) -> tuple[float, float, DataFrame, int]:
    hora_fin_etapa_1, tiempo_sis_1, df1 = etapa_1(permutation.cajas)
    (tiempos_totales, df2, _,
     media_refrescos, media_freidoras, media_pollos) = etapa_2(
        permutation, hora_fin_etapa_1, tiempo_sis_1
    )

    return float(np.mean(tiempos_totales)), float(np.var(tiempos_totales)), df1, df2, media_refrescos, media_freidoras, media_pollos

def graficar_all(medias, freq, mejor_tiempo, mejor_config, mediana, moda, varianza, c1, c2, c3, p95, cov_ref_frei, cov_frei_pol, cov_ref_pol):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # histograma
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(medias, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(mejor_tiempo, color='red', linestyle='--', linewidth=2, label=f'Mejor: {mejor_tiempo:.2f}')
    ax1.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    ax1.axvline(moda, color='orange', linestyle='--', linewidth=2, label=f'Moda: {moda}')
    ax1.set_xlabel('Tiempo Medio (min)', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Tiempos Medios', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    bp = ax2.boxplot(medias, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Tiempo (min)', fontsize=12)
    ax2.set_title('Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # cuartiles
    ax2.text(1.15, c1, f'Q1: {c1:.2f}', fontsize=9, va='center')
    ax2.text(1.15, c2, f'Q2: {c2:.2f}', fontsize=9, va='center')
    ax2.text(1.15, c3, f'Q3: {c3:.2f}', fontsize=9, va='center')

    # frecuencias
    ax3 = fig.add_subplot(gs[1, :2])
    tiempos_ordenados = sorted(freq.keys())
    frecuencias = [freq[k] for k in tiempos_ordenados]

    bars = ax3.bar(tiempos_ordenados, frecuencias, edgecolor='black', alpha=0.7, color='coral')

    if moda in freq:
        idx_moda = tiempos_ordenados.index(moda)
        bars[idx_moda].set_color('red')

    ax3.set_xlabel('Tiempo Medio (min)', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title('Frecuencias de Tiempos (Moda en Rojo)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # tabla estadistica
    ax4 = fig.add_subplot(gs[1:, 2])
    ax4.axis('off')

    combined_text = f"""
    ESTADÍSTICAS DESCRIPTIVAS
    {'='*30}

    Media:        {np.mean(medias):.3f} min
    Mediana:      {mediana:.3f} min
    Moda:         {moda} min

    Desv. Est:    {np.sqrt(varianza):.3f} min
    Varianza:     {varianza:.3f}

    Mínimo:       {np.min(medias):.3f} min
    Máximo:       {np.max(medias):.3f} min
    Rango:        {np.max(medias) - np.min(medias):.3f} min

    Q1 (P25):     {c1:.3f} min
    Q2 (P50):     {c2:.3f} min
    Q3 (P75):     {c3:.3f} min
    P95:          {p95:.3f} min

    MEJOR CONFIGURACIÓN
    {'='*30}

    Tiempo: {mejor_tiempo:.3f} min

    Distribución:
    - Cajas:      {mejor_config.cajas} servidores
    - Refrescos:  {mejor_config.refrescos} servidores
    - Freidora:   {mejor_config.freidora} servidores
    - Pollo:      {mejor_config.pollo} servidores
    """

    ax4.text(0.1, 0.95, combined_text, transform=ax4.transAxes,
    fontsize=10, verticalalignment='top',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # covarianza
    ax5 = fig.add_subplot(gs[2, :2])

    covs = {
        'Refrescos\nvs\nFreidora': cov_ref_frei,
        'Freidora\nvs\nPollo': cov_frei_pol,
        'Refrescos\nvs\nPollo': cov_ref_pol
    }

    colors = ['green' if v > 0 else 'red' for v in covs.values()]
    bars = ax5.bar(covs.keys(), covs.values(), color=colors, edgecolor='black', alpha=0.7)

    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.set_ylabel('Covarianza', fontsize=12)
    ax5.set_title('Covarianzas entre Estaciones (Mejor Config)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, covs.values()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Título general
    fig.suptitle('Dashboard de Optimización - Sistema de Colas',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('resultados_optimizacion.png', dpi=300, bbox_inches='tight')
    plt.show()

def minimizar(presupuesto: int) -> tuple[float | floating, list[int] | None]:
    configuraciones = generar_configuraciones(presupuesto)
    mejor_tiempo = float('inf')
    mejor_config = None

    # covarianza
    cov_ref_frei = None
    cov_frei_pol = None
    cov_ref_pol = None

    freq = dict()
    rango_minimo = float('inf')
    rango_maximo = -float('inf')
    medias = []

    for p in configuraciones:
        tiempos_medios = []
        tiempos_ref = []
        tiempos_frei = []
        tiempos_pol = []

        for _ in range(REPETICIONES_TOTALES):
            media_sim, _, _, _, media_refre, media_frei, media_pol = simular(p)
            tiempos_medios.append(media_sim)
            tiempos_ref.append(media_refre)
            tiempos_frei.append(media_frei)
            tiempos_pol.append(media_pol)

        media_total = np.mean(tiempos_medios)
        medias.append(media_total)

        # rango
        rango_minimo = min(rango_minimo, media_total)
        rango_maximo = max(rango_maximo, media_total)

        # moda
        freq[round(media_total)] = freq.get(round(media_total), 0) + 1

        if media_total < mejor_tiempo:
            mejor_tiempo = media_total
            mejor_config = p
            cov_ref_frei = np.cov(tiempos_ref, tiempos_frei)[0, 1]
            cov_frei_pol = np.cov(tiempos_frei, tiempos_pol)[0, 1]
            cov_ref_pol = np.cov(tiempos_ref, tiempos_pol)[0, 1]

    # mediana
    mediana = np.median(medias)

    # varianza
    varianza = np.var(medias)

    # moda
    moda = None
    max_freq = -1
    for k, i in freq.items():
        if i > max_freq:
            max_freq =  i
            moda = k

    # cuartiles
    c1 = np.percentile(medias, 25)
    c2 = np.percentile(medias, 50)
    c3 = np.percentile(medias, 75)

    # percentiles
    p25 = c1
    p50 = c2
    p75 = c3
    p95 = np.percentile(medias, 95)

    graficar_all(medias, freq, mejor_tiempo, mejor_config, mediana, moda, varianza, c1, c2, c3, p95, cov_ref_frei, cov_frei_pol, cov_ref_pol)

    return mejor_tiempo, mejor_config

def main():
    validar_distribuciones()
    media, config_media = minimizar(3000)

    print("Minimizacion media:", media)
    print(f"\tCajas: {config_media.cajas}" )
    print(f"\tRefrescos: {config_media.refrescos}")
    print(f"\tFreidora: {config_media.freidora}")
    print(f"\tPollo: {config_media.pollo}")


if __name__ == '__main__':
    main()