import math

def congMixto(n,a,c,m):
	return (n*a+c)%m

# None si es completo
# Sino da el numero que falta
def esCompleto(list):
    for i in range(1, len(list)):
        if not list[i]:
            return i
    return None

def todo(x0, a, c, m):
    list = [0 for _ in range(m)]
    x = x0

    for _ in range(m):
        list[x] = 1
        x = congMixto(x,a,c,m)

    return esCompleto(list)

m = 9
a = [i for i in range(1, 20) if (i-1) % 3 == 0]
c = [i for i in range(1, m) if math.gcd(i, m) == 1]
x0 = range(m)

print(f"(x0, b, c)")
for i in a:
    for j in c:
        for k in x0: 
            val = todo(k, i, j, m)
            if val == None:
                print(f"({k}, {i}, {j})")