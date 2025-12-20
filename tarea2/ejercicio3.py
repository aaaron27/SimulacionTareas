"""
Genere 50 n meros aleatorios de 4 d gitos utilizando el generador de cuadrados medios utilizando los siguientes par metros.

a. x(0) = 3567345

b. x(0) = 1234500012

c. x(0) = 4567234902
"""

DIGITOS = 4

def cantDigitos(n):
	if n: return 1+cantDigitos(n//10)
	return 0

def cuadradosMedios(n):
	n*=n
	cd = cantDigitos(n)
	if cd&1: cd+=1
	return (n//(10**((cd-DIGITOS)>>1)))%10**DIGITOS

def todo(x):
    freq = {}
    print(f"x(0) = {x}")
    freq[x] = freq.get(x,0) + 1
    for _ in range(50):
        x = cuadradosMedios(x)
        print(x,end=" ")
        freq[x] = freq.get(x,0) + 1
    print()
    print("\nFrecuencias:")
    for k, v in freq.items():
        print(f"\t{k}: {v}")
    print()

todo(3567345)
todo(1234500012)
todo(4567234902)