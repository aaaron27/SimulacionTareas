from math import sqrt
from itertools import pairwise, chain

JAVA_PATH = "./data/Java.txt"
PYTHON01_PATH = "./data/Python01.txt"
PYTHON02_PATH = "./data/Python02.txt"
C1_PATH = "./data/C1.txt"
C2_PATH = "./data/C2.txt"
RUST_PATH = "./data/Rust.txt"

#ERLANG_PATH = "./data/Erlang.txt"
#SCHEME_PATH = "./data/Scheme.txt"

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
    #global scheme_numbers, erlang_numbersb

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

    '''
    # [1, 20]
    with open(SCHEME_PATH, "r") as f:
        scheme_numbers = [int(i) for i in f.read().splitlines()]

    # [0,1[
    with open(ERLANG_PATH, "r") as f:
        erlang_numbers = [float(i) for i in f.read().splitlines()]
    '''

def calc_limites_promedios(media, z, desviacion_estandar, total):
    calc = desviacion_estandar / sqrt(total)
    l_inf = media - z * calc
    l_sup = media + z * calc

    return (l_inf, l_sup)

def prueba_promedios(media, alfa, desviacion_estandar, numbers):
    l_inf, l_sup = calc_limites_promedios(media, alfa, desviacion_estandar, len(numbers))
    return l_inf <= sum(numbers)/len(numbers) <= l_sup


def execute_tests():
    print("Java")
    # java_media = 0.5
    # java_alfa = 1.96
    # java_desviacion_estandar = sqrt(1/12)

    print("\tPrueba de Promedio:", prueba_promedios(java_media, java_alfa, java_desviacion_estandar, java_numbers))
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Erlang")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Python")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("\tPython 2")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("C")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Scheme")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
    print("\tPrueba de Corridas:")
    print("\tPrueba de Huecos con digitos:")
    print("\tPrueba de Huecos con Numeros:")
    print("\tPrueba de Poker:")
    print("\tPrueba de Series:")

    print("Rust")
    print("\tPrueba de Promedio:")
    print("\tPrueba de Varianza:")
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