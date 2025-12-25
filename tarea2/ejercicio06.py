import math

def congMixto(n,a,c,m):
	return (n*a+c)%m

def esCompleto(vis):
    for i in range(1, len(vis)):
        if not vis[i]:
            return False
    return True

def todo(x0, a, c, m):
    vis = [0 for _ in range(m)]
    x = x0

    for _ in range(m):
        vis[x] = 1
        x = congMixto(x,a,c,m)

    return esCompleto(vis)

m = 15
a = [i for i in range(1, 17) if (i-1) % 15 == 0]
c = [i for i in range(1, m) if math.gcd(i, m) == 1]
x0 = range(m)

print("x0 no decide si el recorrido es completo, solo en que punto empieza por tanto vamos a ignorar x0")

print("x0: [0, 15[")
print("a: {x: [1, 15[ | (x-1)%15 = 0}")
print("c: {x: [1, 15[ | gcd(x, 15) = 1}\n")

print(f"(x0, b, c)")
for i in a:
    for j in c:
        esCompletoVal = todo(10, i, j, m)
        if esCompletoVal:
            print(f"({i}, {j})")