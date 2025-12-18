import numpy as np
import matplotlib.pyplot as plt



N = 3
P = 0.50
SIMULACIONES = int(input("Cantidad de Simulaciones: "))
print(f"Simulando Distribución Binomial (n={N}, p={P}) con {SIMULACIONES} intentos...")
resultados = np.random.binomial(N, P, SIMULACIONES)
plt.figure(figsize=(10, 6))
bins = np.arange(0, N + 2) - 0.5
plt.hist(resultados, bins=bins, color='#3A245C', edgecolor='black', rwidth=0.8, density=True)
plt.title(f'Simulacion Binomial: n={N}, p={P} ; Numero De Simulaciones: {SIMULACIONES}')
plt.xlabel('Numero de exitos')
plt.ylabel('Probabilidad (Frecuencia Relativa)')
plt.xticks(range(N + 1))
plt.grid(axis='y', linestyle='--', alpha=0.5)
unique, counts = np.unique(resultados, return_counts=True)
print("\n--- Resultados ---")
for valor, cuenta in zip(unique, counts):
    prob = cuenta / SIMULACIONES
    print(f"{valor} exitos: {cuenta} veces ({prob:.2%})")
plt.show()
