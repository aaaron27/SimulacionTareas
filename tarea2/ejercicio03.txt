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
    print(f"x(0) = {x}")
    for _ in range(50):
        x = cuadradosMedios(x)
        print(x,end=" ")
    print("\n")

todo(3567345)
todo(1234500012)
todo(4567234902)