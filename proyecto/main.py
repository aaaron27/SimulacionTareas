import numpy as np
from math import inf, sqrt, exp, pi
from dataclasses import dataclass
from random import random
import pandas as pd
from numpy import floating
from pandas import DataFrame
from scipy import stats
from simpson import simpson

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

def etapa_2(permutacion, hora_llegada: list[int], tiempo_sis_1: list[float]) -> list[int]:
    ordenes_input = []
    for i in range(len(hora_llegada)):
        cant_ordenes = get_cant_ordenes(5, 0.4)
        gustos = get_estaciones()
        for _ in range(cant_ordenes):
            targets = [j for j, g in enumerate(gustos) if g]
            if not targets:
                ordenes_input.append((hora_llegada[i], -1, tiempo_sis_1[i]))
            else:
                for t in targets:
                    ordenes_input.append((hora_llegada[i], t, tiempo_sis_1[i]))

    hora_llegada_sorted = sorted(ordenes_input, key=lambda x: x[0])

    servidores_refrescos = [0]*permutacion.refrescos
    servidores_freidoras = [0]*permutacion.freidora
    servidores_pollos = [0]*permutacion.pollo

    tiempo_servidores_refrescos = []
    tiempo_servidores_freidoras = []
    tiempo_servidores_pollos = []

    hora_inicio_atencion = [0]*len(hora_llegada_sorted)
    servidores_usados = [0]*len(hora_llegada_sorted)
    hora_salida = [0]*len(hora_llegada_sorted)
    tiempo_atencion = [0]*len(hora_llegada_sorted)
    tiempo_sis_2 = [0]*len(hora_llegada_sorted)
    tiempo_total = [0]*len(hora_llegada_sorted)
    servicio = [0]*len(hora_llegada_sorted)

    for i in range(len(hora_llegada_sorted)):
        hora, gusto, tiempo_sis_1_i = hora_llegada_sorted[i]

        if gusto == -1:
            hora_inicio_atencion[i] = hora
            servidores_usados[i] = -1
            tiempo_atencion[i] = 0
            hora_salida[i] = hora
            tiempo_sis_2[i] = 0
            tiempo_total[i] = tiempo_sis_2[i] + tiempo_sis_1_i
            servicio[i] = None
        else:
            match gusto:
                case 0:
                    tiempo_atencion[i] = get_random_tiempo_refrescos()
                    servidores_usados[i] = get_servidor_disponible(servidores_refrescos, hora)

                    hora_inicio_atencion[i] = max(
                        servidores_refrescos[servidores_usados[i]],
                        hora
                    )

                    servidores_refrescos[servidores_usados[i]] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    hora_salida[i] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    tiempo_sis_2[i] = hora_salida[i] - hora_inicio_atencion[i]
                    tiempo_total[i] = tiempo_sis_2[i] + tiempo_sis_1_i
                    servicio[i] = 'ref'

                    tiempo_servidores_refrescos.append(tiempo_total[i])
                case 1:
                    tiempo_atencion[i] = get_random_tiempo_freidora()
                    servidores_usados[i] = get_servidor_disponible(servidores_freidoras, hora)

                    hora_inicio_atencion[i] = max(
                        servidores_freidoras[servidores_usados[i]],
                        hora
                    )

                    servidores_freidoras[servidores_usados[i]] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    hora_salida[i] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    tiempo_sis_2[i] = hora_salida[i] - hora_inicio_atencion[i]
                    tiempo_total[i] = tiempo_sis_2[i] + tiempo_sis_1_i
                    servicio[i] = 'frei'

                    tiempo_servidores_freidoras.append(tiempo_total[i])
                case 2:
                    tiempo_atencion[i] = get_random_tiempo_pollo()
                    servidores_usados[i] = get_servidor_disponible(servidores_pollos, hora)

                    hora_inicio_atencion[i] = max(
                        servidores_pollos[servidores_usados[i]],
                        hora
                    )

                    servidores_pollos[servidores_usados[i]] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    hora_salida[i] = hora_inicio_atencion[i] + tiempo_atencion[i]
                    tiempo_sis_2[i] = hora_salida[i] - hora_inicio_atencion[i]
                    tiempo_total[i] = tiempo_sis_2[i] + tiempo_sis_1_i

                    servicio[i] = 'pol'

                    tiempo_servidores_pollos.append(tiempo_total[i])

    media_tiempo_refrescos = np.mean(tiempo_servidores_refrescos)
    media_tiempo_freidora = np.mean(tiempo_servidores_freidoras)
    media_tiempo_pollo = np.mean(tiempo_servidores_pollos)

    df = pd.DataFrame({
        "hora de llegada": hora_llegada_sorted,
        "inicio atencion": hora_inicio_atencion,
        "servidor": servidores_usados,
        "servicio": servicio,
        "tiempo atencion": tiempo_atencion,
        "hora salida": hora_salida,
        "tiempoSIS2": tiempo_sis_2,
        "Total": tiempo_total,
    })

    return tiempo_total, df, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo

def simular(permutation: Config) -> tuple[float, float, DataFrame, int]:
    hora_fin_etapa_1, tiempo_sis_1, df1 = etapa_1(permutation.cajas)
    hora_sistema_total, df2, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo = etapa_2(permutation, hora_fin_etapa_1, tiempo_sis_1)

    return float(np.mean(hora_sistema_total)), float(np.var(hora_sistema_total)), df1, df2, media_tiempo_refrescos, media_tiempo_freidora, media_tiempo_pollo

def minimizar(presupuesto: int) -> tuple[float | floating, list[int] | None]:
    configuraciones = generar_configuraciones(presupuesto)
    mejor_tiempo = float('inf')
    mejor_config = None

    # covarianza
    cov_ref_frei = None
    cov_frei_pol = None
    cov_ref_pol = None

    freq = dict()
    rango_minimo = -1
    rango_maximo = inf
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
        freq[int(media_total)] = freq.get(int(media_total), 0) + 1

        if media_total < mejor_tiempo:
            mejor_tiempo = media_total
            mejor_config = p
            cov_ref_frei = np.cov(tiempos_ref, tiempos_frei)[0, 1]
            cov_frei_pol = np.cov(tiempos_frei, tiempos_pol)[0, 1]
            cov_ref_pol = np.cov(tiempos_ref, tiempos_pol)

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
    p25 = np.percentile(medias, 25)
    p50 = np.percentile(medias, 50)
    p75 = np.percentile(medias, 75)
    p95 = np.percentile(medias, 95)

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