M8_PATH = "./muestras/m8.txt"
M13_PATH = "./muestras/m13.txt"
M15_PATH = "./muestras/m15.txt"
M16_PATH = "./muestras/m16.txt"

m8 = []
m13 = []
m15 = []
m16 = []

def init_numbers():
    global m8, m13, m15, m16

    with open(M8_PATH, "r") as f:
        m8 = [float(i) for i in f.read().splitlines()]

    with open(M13_PATH, "r") as f:
        m13 = [int(i) for i in f.read().splitlines()]

    with open(M15_PATH, "r") as f:
        m15 = [float(i) for i in f.read().splitlines()]

    with open(M16_PATH, "r") as f:
        m16 = [float(i) for i in f.read().splitlines()]

def main():
    init_numbers()

if __name__ == '__main__':
    main()