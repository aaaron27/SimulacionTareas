import math

def esPotencia2(n):
    return n>0 and (n & (n-1)) == 0 

# el generador solo puede visitar numeros coprimos con m = a^b
def phi(n):
    res = n
    i = 2
    while i*i <= n:
        if n%i == 0:
            while n%i == 0:
                n //= i
            res -= res // i
        i+=1
    if n>1:
        res -= res // n
    return res

def expLog(a, b):
	if not b:
		return 1
	m = expLog(a, b//2)
	m *= m
	if b % 2:
		return m * a
	return m

def congMultiplicativo(n,a,m):
	return (n*a)%m

def calcMax(m):
    if esPotencia2(m) and m >= 8:
        return m // 4
    return phi(m)

def todo(x0, a, b1, b2, l):
    m = expLog(b1, b2)
    periodoMaximo = calcMax(m)

    print(f"{l}: x(0) = {x0}")

    # x0 tiene que ser coprimo con m para poder alcanzar todos los coprimos
    if math.gcd(x0, m) != 1 or math.gcd(a, m) != 1:
        return "No"
    
    vis = set()
    while x0 not in vis:
        vis.add(x0)
        x0 = congMultiplicativo(x0, a, m)
    
    if len(vis) == periodoMaximo: 
        return "Si"
    return "No"

    
print(todo(7, 5, 2, 6, "A"))
print(todo(9, 11, 2, 7, "B"))
print(todo(3, 221, 10, 3, "C"))
print(todo(17, 203, 10, 5, "D"))
print(todo(19, 211, 10, 8, "E"))