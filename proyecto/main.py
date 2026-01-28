from typing import Tuple, Any, List

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

def get_random_tiempo_postres():
    return np.random.binomial(5, 0.6)

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
    if not hora_llegada: return [], [], [], pd.DataFrame()

    servidores = [0.0] * n_cajas
    espera_etapa1 = []
    hora_fin_etapa1 = []

    # Listas para el DataFrame
    h_inicio_atencion = []
    t_atencion = []
    serv_usado = []

    for llegada in hora_llegada:
        idx = get_servidor_disponible(servidores, llegada)
        inicio = max(llegada, servidores[idx])

        # CÁLCULO DE ESPERA 1
        espera = inicio - llegada
        espera_etapa1.append(espera)

        tiempo_atencion = get_random_tiempo_cajas()
        fin = inicio + tiempo_atencion
        servidores[idx] = fin

        hora_fin_etapa1.append(fin)
        h_inicio_atencion.append(inicio)
        t_atencion.append(tiempo_atencion)
        serv_usado.append(idx)

    df = pd.DataFrame({
        "Hora llegada": hora_llegada,
        "Espera": espera_etapa1,
        "Hora fin": hora_fin_etapa1
    })

    return hora_fin_etapa1, espera_etapa1, df


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


def etapa_2_2(permutacion, hora_fin_e1: list[int], esperas_e1: list[float]):
    ordenes_input = []
    # Emparejamos cada salida de caja con su espera previa
    for i in range(len(hora_fin_e1)):
        cant_ordenes = get_cant_ordenes(5, 0.4)
        gustos = get_estaciones()
        for _ in range(cant_ordenes):
            targets = [j for j, g in enumerate(gustos) if g]
            # Pasamos la hora de llegada a etapa 2 y la espera que ya trae de la caja
            if not targets:
                ordenes_input.append((hora_fin_e1[i], -1, esperas_e1[i]))
            else:
                for t in targets:
                    ordenes_input.append((hora_fin_e1[i], t, esperas_e1[i]))

    ordenes_sorted = sorted(ordenes_input, key=lambda x: x[0])

    serv_ref = [0] * permutacion.refrescos
    serv_frei = [0] * permutacion.freidora
    serv_pol = [0] * permutacion.pollo

    esperas_totales = []  # Wait1 + Wait2
    tiempos_ref, tiempos_frei, tiempos_pol = [], [], []

    for hora_llegada_e2, gusto, espera_e1 in ordenes_sorted:
        if gusto == -1:
            esperas_totales.append(espera_e1)
        else:
            # Seleccionar servidor y calcular espera en etapa 2
            if gusto == 0:  # Refrescos
                idx = get_servidor_disponible(serv_ref, hora_llegada_e2)
                inicio = max(hora_llegada_e2, serv_ref[idx])
                espera_e2 = inicio - hora_llegada_e2
                serv_ref[idx] = inicio + get_random_tiempo_refrescos()
                tiempos_ref.append(espera_e2)
            elif gusto == 1:  # Freidora
                idx = get_servidor_disponible(serv_frei, hora_llegada_e2)
                inicio = max(hora_llegada_e2, serv_frei[idx])
                espera_e2 = inicio - hora_llegada_e2
                serv_frei[idx] = inicio + get_random_tiempo_freidora()
                tiempos_frei.append(espera_e2)
            else:  # Pollo
                idx = get_servidor_disponible(serv_pol, hora_llegada_e2)
                inicio = max(hora_llegada_e2, serv_pol[idx])
                espera_e2 = inicio - hora_llegada_e2
                serv_pol[idx] = inicio + get_random_tiempo_pollo()
                tiempos_pol.append(espera_e2)

            # EL RESULTADO ES LA SUMA DE TODAS LAS ESPERAS (FILAS)
            esperas_totales.append(espera_e1 + espera_e2)

    # Medias de espera por estación para la covarianza (pueden ser 0 si no hubo pedidos)
    m_ref = np.mean(tiempos_ref) if tiempos_ref else 0
    m_frei = np.mean(tiempos_frei) if tiempos_frei else 0
    m_pol = np.mean(tiempos_pol) if tiempos_pol else 0

    return esperas_totales, None, m_ref, m_frei, m_pol

def simular(permutation: Config) -> tuple[float, float, Any, Any, floating, floating, floating]:
    hora_fin_etapa_1, tiempo_sis_1, df1 = etapa_1(permutation.cajas)
    hora_sistema_total, df2, media_total,  media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo = etapa_2(permutation, hora_fin_etapa_1, tiempo_sis_1)

    return float(np.mean(hora_sistema_total)), float(np.var(hora_sistema_total)), df1, df2, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo


def simular_2(permutation: Config) -> tuple[float, float, Any, Any, floating, floating, floating]:
    hora_fin_etapa_1, tiempo_sis_1, df1 = etapa_1(permutation.cajas)
    hora_sistema_total, df2, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo = etapa_2_2(permutation,
                                                                                                         hora_fin_etapa_1,
                                                                                                         tiempo_sis_1)
    return float(np.mean(hora_sistema_total)), float(
        np.var(hora_sistema_total)), df1, df2, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo

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

        for _ in range(REPETICIONES_TOTALES + 20):
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


def encontrar_costo(meta_minutos=3.0):
    configuraciones = []

    for c in range(1, 5):
        for r in range(1, 5):
            for f in range(1, 5):
                for p in range(1, 5):
                    costo = (c * COSTOS['cajas'] +
                             r * COSTOS['refrescos'] +
                             f * COSTOS['freidora'] +
                             p * COSTOS['pollo'])
                    configuraciones.append(Config(c, r, f, p, costo))

    # ORDENAR POR COSTO
    configuraciones.sort(key=lambda x: x.costo)

    print(f"Buscando configuración mínima para espera <= {meta_minutos} min...")
    for config in configuraciones:

        conf = simular_2(config)
        media_preliminar = conf[0]
        print(media_preliminar)
        if media_preliminar <= meta_minutos:
            return config, media_preliminar

    return None, None

def main():
    validar_distribuciones()
    print("\n")
    mejor_config, tiempo_espera = encontrar_costo(3.0)

    if mejor_config is not None:
        print(f"--- RESULTADO OBJETIVO 2.a ---")
        print(f"Costo Mínimo Necesario: ${mejor_config.costo}")
        print(f"Configuración Óptima:")
        print(f"  - Cajas: {mejor_config.cajas}")
        print(f"  - Refrescos: {mejor_config.refrescos}")
        print(f"  - Freidoras: {mejor_config.freidora}")
        print(f"  - Parrillas Pollo: {mejor_config.pollo}")
        print(f"Tiempo de espera promedio logrado: {tiempo_espera:.2f} min")
    print("\n")
    print("Medias")
    mediaa, config_mediaa = minimizar(2000)

    print("Minimizacion media 2000:", mediaa)
    print(f"\tCajas: {config_mediaa.cajas}" )
    print(f"\tRefrescos: {config_mediaa.refrescos}")
    print(f"\tFreidora: {config_mediaa.freidora}")
    print(f"\tPollo: {config_mediaa.pollo}")

    media, config_media = minimizar(3000)

    print("Minimizacion media 3000:", media)
    print(f"\tCajas: {config_media.cajas}" )
    print(f"\tRefrescos: {config_media.refrescos}")
    print(f"\tFreidora: {config_media.freidora}")
    print(f"\tPollo: {config_media.pollo}")
if __name__ == '__main__':
    main()