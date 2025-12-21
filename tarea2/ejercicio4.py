def congMixto(n,a,c,m):
	return (n*a+c)%m

# None si es completo
# Sino da el numero que falta
def esCompleto(list):
    for i in range(1, len(list)):
        if not list[i]:
            return i
    return None

def todo(x0, a, c, m, l):
    list = [0 for _ in range(m)]
    list[x0] = 1

    print(f"{l}:")
    for _ in range(m+1):
        print(x0,end=" ")
        x0 = congMixto(x0, a, c, m)
        list[x0] = 1
    print()
    
    val = esCompleto(list)
    print("\nEs completo?", "Si" if val == None else f"Falta el numero {val} es aparecer\n")
    
todo(7, 5, 24, 32, 'a')
todo(8, 9, 13, 32, 'b')
todo(13, 50, 17, 64, 'c')
todo(15, 8, 16, 100, 'd')
todo(3, 5, 21, 100, 'e')