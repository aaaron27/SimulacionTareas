NUMBITS = 5

def SET(n,i):
	return n | 1<<(i)

def congruencialBinario(n):
    ultimoBit = n&1
    antePenultimoBit = (n&4)>>2 
    xorBits = ultimoBit ^ antePenultimoBit

    if xorBits: 
        num = SET(n>>1, NUMBITS-1)
        print(bin(n), "-", ultimoBit, "^", antePenultimoBit, "->", xorBits, "->", num)
        return num
    print(bin(n), "-", ultimoBit, "^", antePenultimoBit, "->", xorBits, "->", n>>1)
    return n>>1

print("La semilla: [1, 32[")

x0 = 16
print("Semilla:", x0)
for i in range(31):
    print(x0, "\t-> ", end="")
    x0 = congruencialBinario(x0)
    print()