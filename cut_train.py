import numpy as np
import pandas as pd

f = open('data-202205/train.txt', 'r')
b = open('data-202205/train2.txt', 'w')
while True:
    line = f.readline()
    user, num = line.split('|')
    if user == '500':
        break
    b.write(line)
    for i in range(int(num)):
        line = f.readline()
        b.write(line)
