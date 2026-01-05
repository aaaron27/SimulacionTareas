from random import random, randint
import os

n = 1000000

file_path1 = "../data/Python01.txt"
file_path2 = "../data/Python02.txt"

if os.path.exists(file_path1):
    os.remove(file_path1)
if os.path.exists(file_path2):
    os.remove(file_path2)

with open(file_path1, 'a') as f:
    f.writelines(f"{random()}\n" for _ in range(n))

with open(file_path2, 'a') as f:
    f.writelines(f"{randint(1, 6)}\n" for _ in range(n))
