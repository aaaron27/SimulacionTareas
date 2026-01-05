"""
Utilice el m todo congruencial binario pero en lugar de realizar un xor con las posiciones 0 y 1, de derecha a izquierda,
realice un xor con las posiciones 0 y 2. Utilice las siguientes semillas y muestre si realiza todo el recorrido o no.
a. x(0) = 110
b. x(0) = 1111
"""

def SET(n, i):
    return n | 1<<(i)

def congruencialBinario(n, numbits):
    ultimoBit = n&1
    antePenultimoBit = (n&4)>>2 
    xorBits = ultimoBit ^ antePenultimoBit

    if xorBits: return SET(n>>1, numbits-1)
    return n>>1

x0 = 6
print("A:", x0)
for _ in range(10):
    x0 = congruencialBinario(x0, 3)
    print(x0, end=" ")
print()

x02 = 15
print("B:", x02)
for _ in range(10):
    x02 = congruencialBinario(x02, 4)
    print(x02, end=" ")
print()

print("A(110): Recorrido completo")
print("B(1111): Recorrido incompleto")