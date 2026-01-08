from math import sqrt
from itertools import pairwise, chain

JAVA_PATH = "./data/Java.txt"
PYTHON01_PATH = "./data/Python01.txt"
PYTHON02_PATH = "./data/Python02.txt"
C1_PATH = "./data/C1.txt"
C2_PATH = "./data/C2.txt"
RUST_PATH = "./data/Rust.txt"

ERLANG_PATH = "./data/Erlang.txt"
SCHEME_PATH = "./data/Scheme.txt"

java_numbers = []
python1_numbers = []
python2_numbers = []
c1_numbers = []
c2_numbers = []
rust_numbers = []
scheme_numbers = []
erlang_numbers = []

def init_numbers():
    global java_numbers, python1_numbers, python2_numbers
    global c1_numbers, c2_numbers, rust_numbers
    global scheme_numbers, erlang_numbers

    # [0,1[
    with open(JAVA_PATH, "r") as f:
        java_numbers = [float(i) for i in f.read().splitlines()]
    
    # [0,1[
    with open(PYTHON01_PATH, "r") as f:
        python1_numbers = [float(i) for i in f.read().splitlines()]
    
    # [1,6]
    with open(PYTHON02_PATH, "r") as f:
        python2_numbers = [int(i) for i in f.read().splitlines()]
    
    # [1,4]
    with open(C1_PATH, "r") as f:
        c1_numbers = [int(i) for i in f.read().splitlines()]

    # [1,8]
    with open(C2_PATH, "r") as f:
        c2_numbers = [int(i) for i in f.read().splitlines()]

    # [0,1[
    with open(RUST_PATH, "r") as f:
        rust_numbers = [float(i) for i in f.read().splitlines()]

    # # [1, 20]
    # with open(SCHEME_PATH, "r") as f:
    #     scheme_numbers = [int(i) for i in f.read().splitlines()]

    # # [0,1[
    # with open(ERLANG_PATH, "r") as f:
    #     erlang_numbers = [float(i) for i in f.read().splitlines()]

def calc_limites_promedios(media, z, desviacion_estandar, total):
    calc = desviacion_estandar / sqrt(total)
    l_inf = media - z * calc
    l_sup = media + z * calc

    return (l_inf, l_sup)

def prueba_promedios(media, alfa, desviacion_estandar, numbers):
    l_inf, l_sup = calc_limites_promedios(media, alfa, desviacion_estandar, len(numbers))
    return l_inf <= sum(numbers)/len(numbers) <= l_sup

def calc_varianza_muestral(numbers):
    promedio_numbers = sum(numbers) / len(numbers)
    res = 0
    for i in numbers:
        res += (i - promedio_numbers)**2
    return res / (len(numbers) - 1)

def prueba_varianza(numbers, varianza, l_inf, l_sup):
    x2 = (len(numbers) - 1) * calc_varianza_muestral(numbers) / varianza
    return l_inf <= x2 <= l_sup

def calc_frecuencias(columnas):
    sec = []
    for i in columnas:
        sec.extend(i)

    freq = {i: {} for i in range(10)}

    for i in range(10):
        c = 0
        vis = False

        for j in sec:
            if j == i:
                if vis:
                    freq[i][c] = freq[i].get(c, 0) + 1
                c = 0
                vis = True
            else:
                if vis: c += 1

    return freq

def get_fo(freq):
    huecos = set()
    fo = {}
    for i in range(10):
        huecos.update(freq[i].keys())
    
    for i in sorted(huecos):
        fo[i] = sum(freq[j].get(i, 0) for j in range(10))
    
    return fo

def prueba_huecos_digitos(numbers):
    columnas = []

    for i in range(len(numbers)):
        row = []
        digitos = [int(d) for d in str(numbers[i]) if d.isdigit()]

        for j in digitos[1:]:
            row.append(j)
        columnas.append(row)
    
    freq = calc_frecuencias(columnas)

    fo = get_fo(freq)

    fo1 = {i: 0 for i in range(7)}
    fo1['>=7'] = 0

    for i,j in fo.items():
        if i <= 6:
            fo1[i] += j
        else:
            fo1['>=7'] += j
        
    n = sum(fo1.values())
    res = 0

    for i in range(7):
        fe = n * (0.9 ** i)
        res += (fo1[i] - fe) ** 2 / fe
    
    fe = n * (0.9 ** 7)
    res += (fo1['>=7'] - fe) ** 2 / fe

def execute_tests():
    print("Java")
    java_media = 0.5
    java_alfa = 1.96
    java_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(java_media, java_alfa, java_desviacion_estandar, java_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(java_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos(java_numbers))
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Erlang")
    erlang_media = 0.0
    erlang_alfa = 1.96
    erlang_desviacion_estandar = 0.0

    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Python")
    python1_media = 0.5
    python1_alfa = 1.96
    python1_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(python1_media, python1_alfa, python1_desviacion_estandar, python1_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(python1_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Python 2")
    python2_media = 3.5
    python2_alfa = 1.96
    python2_desviacion_estandar = sqrt(35/12)

    print("\tPrueba de Promedio:", prueba_promedios(python2_media, python2_alfa, python2_desviacion_estandar, python2_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(python2_numbers, 35/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("C")
    c_media = 2.5
    c_alfa = 1.96
    c_desviacion_estandar = sqrt(15/12)

    print("\tPrueba de Promedio:", prueba_promedios(c_media, c_alfa, c_desviacion_estandar, c1_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(c1_numbers, 15/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("C2")
    c2_media = 4.5
    c2_alfa = 1.96
    c2_desviacion_estandar = sqrt(63/12)

    print("\tPrueba de Promedio:", prueba_promedios(c2_media, c2_alfa, c2_desviacion_estandar, c2_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(c2_numbers, 63/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Scheme")
    scheme_media = 0.0
    scheme_alfa = 1.96
    scheme_desviacion_estandar = 0.0

    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Rust")
    rust_media = 0.5
    rust_alfa = 1.96
    rust_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(rust_media, rust_alfa, rust_desviacion_estandar, rust_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(rust_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

def main():
    init_numbers()
    execute_tests()

if __name__ == '__main__':
    main()