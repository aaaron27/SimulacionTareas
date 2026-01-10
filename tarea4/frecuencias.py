import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

nombres_archivos = ['muestras\m15.txt', 'muestras\m16.txt', 'muestras\m8.txt', 'muestras\m13.txt']

def frecuencia(archivos):
    for archivo in archivos:
        nombre_clean = os.path.basename(archivo).split('.')[0]
        print(f"\n{'=' * 20} PROCESANDO {nombre_clean} {'=' * 20}")

        # Cargar datos
        try:
            # archivo no tiene header y es una sola columna
            df = pd.read_csv(archivo, header=None, names=['valor'])
            datos = df['valor']
        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo '{archivo}'. Saltando...")
            continue
        except Exception as e:
            print(f"ERROR leyendo '{archivo}': {e}")
            continue

        conteos, bordes = np.histogram(datos, bins='auto')
        # Histograma Frecuencia A
        plt.figure(figsize=(8, 5))
        plt.hist(datos, bins=bordes, color='#4F81BD', edgecolor='black', alpha=0.9)

        plt.title(f"Histograma Frecuencia Absoluta - {nombre_clean}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia Absoluta ($f_i$)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Guardar imagen
        nom_abs = f"histogramas\hist_absoluto_{nombre_clean}.png"
        plt.savefig(nom_abs, dpi=300)
        plt.show()
        print(f">> Gráfico guardado: {nom_abs}")

        #Histograma Frecuencia R
        plt.figure(figsize=(8, 5))
        pesos = np.ones_like(datos) / len(datos)
        plt.hist(datos, bins=bordes, weights=pesos, color='#C0504D', edgecolor='black', alpha=0.9)

        plt.title(f"Histograma Frecuencia Relativa - {nombre_clean}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia Relativa ($h_i$)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Guardar imagen
        nom_rel = f"histogramas\hist_relativo_{nombre_clean}.png"
        plt.savefig(nom_rel, dpi=300)
        plt.show()
        print(f">> Gráfico guardado: {nom_rel}")

        # Calculo explicito de Frecuencias Absolutas y Relativas
        tabla_frecuencias = pd.DataFrame({
            'Límite Inferior': bordes[:-1],
            'Límite Superior': bordes[1:],
            'Frecuencia Absoluta': conteos,
            'Frecuencia Relativa': conteos / len(datos)
        })

        print(f"Tabla de Frecuencias (primeras 10 filas) para {archivo}:")
        print(tabla_frecuencias.head(10).to_string(index=False))
        print(f"\nTotal de datos: {len(datos)}")
        print("-" * 50)

frecuencia(nombres_archivos)
