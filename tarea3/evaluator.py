from math import sqrt
from collections import defaultdict, Counter

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

    # [1, 20]
    with open(SCHEME_PATH, "r") as f:
        scheme_numbers = [int(i) for i in f.read().splitlines()]

    # [0,1[
    with open(ERLANG_PATH, "r") as f:
        erlang_numbers = [float(i) for i in f.read().splitlines()]

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

def calc_corridas(numbers):
    signos = []

    for i in range(len(numbers)-1):
        if numbers[i] < numbers[i+1]:
            signos.append(1)
        elif numbers[i] > numbers[i+1]:
            signos.append(-1)
        # si son iguales, se ignoran

    if len(signos) == 0:
        return 0, 0

    corridas = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i-1]:
            corridas += 1

    return corridas, len(signos)
def prueba_corridas(numbers):
    a,n = calc_corridas(numbers)

    esperanza = (2 * n - 1) / 3
    desviacion =  sqrt((16 * n - 29) / 90)
    z0 =  (a - esperanza) / desviacion

    print(z0)
    return -1.96 <= z0 <= 1.96

def calc_frecuencias_enteros(sec, simbolos, max_hueco=100):
    res = [[0 for _ in simbolos] for _ in range(max_hueco)]

    for idx, simbolo in enumerate(simbolos):
        pos = [i for i, x in enumerate(sec) if x == simbolo]

        for i in range(len(pos) - 1):
            c = pos[i+1] - pos[i] - 1
            if c < max_hueco:
                res[c][idx] += 1

    return res

def calc_frecuencias(listas):
    res = [[0 for _ in range(10)] for _ in range(100)]

    for digitos in listas:
        for digito in range(10):
            pos = [i for i, x in enumerate(digitos) if x == digito]

            for i in range(len(pos) - 1):
                c = pos[i+1] - pos[i] - 1
                if c < 100:
                    res[c][digito] += 1
                
    return res

def prueba_huecos_digitos(numbers):
    columnas = [[] for _ in range(6)]

    for i in range(len(numbers)):
        number = f"{numbers[i]:.6f}".replace("0.", "").replace(".", "")
        for j in [int(k) for k in number]:
            columnas[i%6].append(j)

    res = calc_frecuencias(columnas)
    fo = [0 for _ in range(8)]
    # primeros numeros hasta el 6
    for i in range(7):
        fo[i] = sum(res[i])
    # >= 7
    for i in range(7, 100):
        fo[7] += sum(res[i])

    pe = [0.1, 0.09, 0.0810, 0.0729, 0.0656, 0.0590, 0.0531, 0.4783]
    n = sum(fo)
    fe = [i*n for i in pe]

    fofe = [(fo[i] - fe[i])**2 / fe[i] for i in range(8)]

    x2 = sum(fofe)
    punto_rechazo = 14.07
    
    return x2 <= punto_rechazo

def prueba_huecos_digitos_enteros(numbers, simbolos):
    m = len(simbolos)

    res = calc_frecuencias_enteros(numbers, simbolos)

    fo = [0]*8
    for i in range(7):
        fo[i] = sum(res[i])
    for i in range(7, len(res)):
        fo[7] += sum(res[i])

    p = 1 / m
    q = 1 - p

    pe = [q**i * p for i in range(7)]
    pe.append(q**7) # >=7

    n = sum(fo)
    fe = [n * pe[i] for i in range(8)]

    x2 = sum((fo[i] - fe[i])**2 / fe[i] for i in range(8))

    punto_rechazo = 14.07
    return x2 <= punto_rechazo

def prueba_huecos_digitos_enteros2(numbers, simbolos):
    m = len(simbolos)

    res = calc_frecuencias_enteros(numbers, simbolos)

    fo = [0]*5
    for i in range(4):
        fo[i] = sum(res[i])
    for i in range(4, len(res)):
        fo[4] += sum(res[i])

    p = 1 / m
    q = 1 - p

    pe = [
        p,
        p*q,
        q**2*p,
        q**3*p,
        q**4
    ]

    n = sum(fo)
    fe = [n * pi for pi in pe]

    x2 = sum((fo[i] - fe[i])**2 / fe[i] for i in range(5))

    punto_rechazo = 9.49
    return x2 <= punto_rechazo

def prueba_huecos_numeros(numbers, alpha, beta, prob, total_clases=20):
    indices = [i for i, x in enumerate(numbers) if alpha <= x <= beta]
    if len(indices) < 2:
        return "Fallo (Insuficientes datos en rango)"
    huecos = [indices[i+1] - indices[i] - 1 for i in range(len(indices)-1)]
    total_huecos = len(huecos)
    #Contar frecuencias observadas
    observados = defaultdict(int)
    max_idx = total_clases - 1
    for h in huecos:
        if h >= max_idx:
            observados[max_idx] += 1
        else:
            observados[h] += 1
    #Calcular Chi-Cuadrada
    chi_square = 0.0
    for k in range(total_clases):
        obs = observados[k]
        #Calcular frecuencia esperada: Total * Prob_Hueco(k)
        if k == max_idx:
            # Probabilidad acumulada para (>= k)
            # P(Hueco >= k) = (1-p)^k
            prop = pow(1 - prob, k)
        else:
            # P(Hueco = k) = p * (1-p)^k
            prop = prob * pow(1 - prob, k)
        esp = total_huecos * prop
        # Evitar n/0
        if esp > 0:
            chi_square += ((obs - esp) ** 2) / esp
    # Chi2(0.95, 19) ~= 30.144
    valor_critico = 30.144
    resultado = chi_square <= valor_critico
    return f"{resultado} (Chi2: {chi_square:.2f}, Critico: {valor_critico})"

def prueba_poker(numbers, is_int=False, max_val=1):

    probs = [0.3024, 0.5040, 0.1080, 0.0720, 0.0090, 0.0045, 0.0001]
    nombres = ["Dif", "Par", "2Par", "Trio", "Full", "Poker", "Quint"]

    conteos = [0] * 7 # para las 7 categorias

    for i in range(0, len(numbers) - 4, 5):
            hand = []
            for j in range(5):
                val = numbers[i + j]
                if is_int:
                    digit = int(val) % 10
                else:
                    digit = int(val * 10) % 10
                hand.append(digit)

            # Clasificar la mano
            counts_dict = Counter(hand)
            shape = sorted(counts_dict.values(), reverse=True)

            if shape == [1, 1, 1, 1, 1]:   conteos[0] += 1 # Todos Diferentes
            elif shape == [2, 1, 1, 1]:   conteos[1] += 1 # Un Par
            elif shape == [2, 2, 1]:      conteos[2] += 1 # Dos Pares
            elif shape == [3, 1, 1]:      conteos[3] += 1 # Trio
            elif shape == [3, 2]:         conteos[4] += 1 # Full House
            elif shape == [4, 1]:         conteos[5] += 1 # Póker
            elif shape == [5]:            conteos[6] += 1 # 5

    # Calcular Chi-Cuadrada
    chi_square = 0.0
    total = len(numbers) // 5

    for i in range(7):
        obs = conteos[i]
        esp = total * probs[i]

        # Evitar n/0
        if esp > 0:
            chi_square += ((obs - esp) ** 2) / esp

    #
    # Chi2(0.95, 6) = 12.592
    valor_critico = 12.592

    resultado = chi_square <= valor_critico
    return f"{resultado} (Chi2: {chi_square:.2f})"

def execute_tests():
    print("Java")
    java_media = 0.5
    java_alfa = 1.96
    java_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(java_media, java_alfa, java_desviacion_estandar, java_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(java_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(java_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos(java_numbers))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(java_numbers, 0.5, 1.0, 0.5))
    print("\tPrueba de Poker:", prueba_poker(java_numbers))
    print("\tPrueba de Series:")

    print("Erlang")
    erlang_media = 0.5
    erlang_alfa = 1.96
    erlang_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(erlang_media, erlang_alfa, erlang_desviacion_estandar, erlang_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(erlang_numbers, 1/12, 997228, 1002770))
    print("\tPrueba de Corridas:", prueba_corridas(erlang_numbers))
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(erlang_numbers, 0.5, 1.0, 0.5))
    print("\tPrueba de Poker:", prueba_poker(erlang_numbers))
    print("\tPrueba de Series:")

    print("Python")
    python1_media = 0.5
    python1_alfa = 1.96
    python1_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(python1_media, python1_alfa, python1_desviacion_estandar, python1_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(python1_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(python1_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos(python1_numbers))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(python1_numbers, 0.5, 1.0, 0.5))
    print("\tPrueba de Poker:", prueba_poker(python1_numbers))
    print("\tPrueba de Series:")

    print("Python 2")
    python2_media = 3.5
    python2_alfa = 1.96
    python2_desviacion_estandar = sqrt(35/12)

    print("\tPrueba de Promedio:", prueba_promedios(python2_media, python2_alfa, python2_desviacion_estandar, python2_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(python2_numbers, 35/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(python2_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos_enteros(python2_numbers, [1,2,3,4,5,6]))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(python2_numbers, 3, 3, 1/6))
    print("\tPrueba de Poker:", prueba_poker(python2_numbers, is_int=True, max_val=6))
    print("\tPrueba de Series:")

    print("C")
    c_media = 2.5
    c_alfa = 1.96
    c_desviacion_estandar = sqrt(15/12)

    print("\tPrueba de Promedio:", prueba_promedios(c_media, c_alfa, c_desviacion_estandar, c1_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(c1_numbers, 15/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(c1_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos_enteros(c1_numbers, [1,2,3,4]))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(c1_numbers, 2, 2, 0.25))
    print("\tPrueba de Poker:", prueba_poker(c1_numbers, is_int=True, max_val=4))
    print("\tPrueba de Series:")

    print("C2")
    c2_media = 4.5
    c2_alfa = 1.96
    c2_desviacion_estandar = sqrt(63/12)

    print("\tPrueba de Promedio:", prueba_promedios(c2_media, c2_alfa, c2_desviacion_estandar, c2_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(c2_numbers, 63/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(c2_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos_enteros(c2_numbers, [1,2,3,4,5,6,7,8]))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(c2_numbers, 5, 5, 1/8))
    print("\tPrueba de Poker:", prueba_poker(c2_numbers, is_int=True, max_val=8))
    print("\tPrueba de Series:")

    print("Scheme")
    scheme_media = 10.5
    scheme_alfa = 1.96
    scheme_desviacion_estandar = sqrt(399/12)

    print("\tPrueba de Promedio:", prueba_promedios(scheme_media, scheme_alfa, scheme_desviacion_estandar, scheme_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(scheme_numbers, 399/12, 997228, 1002770))
    print("\tPrueba de Corridas:", prueba_corridas(scheme_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos_enteros2(scheme_numbers, [i for i in range(1, 21)]))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(scheme_numbers, 10, 10, 0.05))
    print("\tPrueba de Poker:", prueba_poker(scheme_numbers, is_int=True, max_val=20))
    print("\tPrueba de Series:")

    print("Rust")
    rust_media = 0.5
    rust_alfa = 1.96
    rust_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(rust_media, rust_alfa, rust_desviacion_estandar, rust_numbers))
    print("\tPrueba de Varianza:", prueba_varianza(rust_numbers, 1/12, 997229, 1002769))
    print("\tPrueba de Corridas:", prueba_corridas(rust_numbers))
    print("\tPrueba de Huecos con digitos:", prueba_huecos_digitos(rust_numbers))
    print("\tPrueba de Huecos con Numeros:", prueba_huecos_numeros(rust_numbers, 0.5, 1.0, 0.5))
    print("\tPrueba de Poker:", prueba_poker(rust_numbers))
    print("\tPrueba de Series:") 

def main():
    init_numbers()
    execute_tests()

if __name__ == '__main__':
    main()