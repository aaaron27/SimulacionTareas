import math

def congMultiplicativo(n,a,m):
	return (n*a)%m

def congAditivo(n,a,m):
	return (n+a)%m

# None si es completo
# Sino da el numero que falta
def esCompleto(list):
    for i in range(1, len(list)):
        if not list[i]:
            return i
    return None

def todoMultiplicativo(x0, a, m):
    list = [0 for _ in range(m)]
    x = x0

    for _ in range(m):
        list[x] = 1
        x = congMultiplicativo(x,a,m)

    return esCompleto(list)

def todoAditivo(x0, a, m):
    list = [0 for _ in range(m)]
    x = x0

    for _ in range(m):
        list[x] = 1
        x = congAditivo(x,a,m)

    return esCompleto(list)

m = 10
a_mult = range(1, m)
x0_mult = range(1, m)

print("Metodo Congruencial Multiplicativo")
for a in a_mult:
    for x0 in x0_mult:
        if todoMultiplicativo(x0, a, m) == None:
            print(f"({x0}, {a})")

m = 10
a_aditivo = [i for i in range(1, m) if math.gcd(i, m) == 1]
x0_aditivo = range(m)

print("Metodo Congruencial Aditivo")
for a in a_aditivo:
    for x0 in x0_aditivo:
        if todoAditivo(x0, a, m) == None:
            print(f"({x0}, {a})")