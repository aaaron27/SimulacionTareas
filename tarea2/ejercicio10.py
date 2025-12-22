import random

def esPrimo(n):
    if n<2:
        return False
    if n == 2:
        return True
    if n%2 == 0:
        return False
    
    i = 3
    while i*i < n:
        if n%i == 0:
            return False
        i += 2
    return True

def coins():
    escudos = 0
    casos = 100000

    for _ in range(casos):
        number = random.randint(0, 9)
        if number&1:
            escudos += 1
        
    print(f"Escudos {escudos/casos}, Coronas {(casos - escudos)/casos}\n")


def probabilities():
    a = 0
    b = 0
    c = 0
    casos = 1000000

    for _ in range(casos):
        number = random.randint(0, 9)

        if not number:
            a += 1
        elif esPrimo(number):
            b += 1
        else:
            c += 1
        
    print(f"a: {a/casos}, b = {b/casos}, c = {c/casos}\n")

def ignore():
    dados = [0 for _ in range(6)]
    casos = 1000000
    i = 0
    while i < casos:
        number = random.randint(0, 9)
        if number < 6:
            dados[number] += 1
        else:
            continue
    
        i += 1

    for i in dados:
        print(i/casos, end=" ")
    print("\n")

def ignore2():
    dados = [0 for _ in range(6)]
    casos = 1000000
    i = 0
    while i < casos:
        number = random.randint(0, 99)
        # como haciemos antes de ignorar del 6 al 9
        if number < 96:
            dados[number//16] += 1
        else:
            continue
        i += 1

    for i in dados:
        print(i/casos, end=" ")
    print("\n")

print("A:")
print("Si el valor pseudo aleatorio es par le asignamos corono y los impares el escudo, como tenemos misma cantidad de pares e impares se puede conseguir una distribucion uniformede probabilidad\n")

print("Prueba")
coins()

print("B:")
print("La probabilidad de salir 0, la probabilidad de salir un numero primo y la probabilidad de tener un numero compuesto con el 1\n")

print("Prueba")
probabilities()

print("C:")
print("Utilizar los primeros primeros numeros menores a 6 e ignorar los demas\n")

print("Prueba")
ignore()

print("D:")
print("Ignorar como lo haciamos anteriormente, solo que ahora tenemos que distribuir mejor los numeros. 100/6 = 16.66, 6x16 = 96. El dado con 0 le asignamos del 0 al 15, el 1, le asignamos del 16 al 31 y asi. Hasta el dado con 5 de 80 a 95\n")

print("Prueba")
ignore2()
