import random
import os

# training set

f = open('water_train.txt', 'r')
files = f.readlines()
f.close()

f = open('river_train.txt', 'w')
for file in files:
    if 'ADE20K' not in file and 'lab' not in file:
        f.write(file)
f.close()

# validation set
f = open('water_val.txt', 'r')
files = f.readlines()
f.close()  

f = open('river_val.txt', 'w')
for file in files:
    if 'ADE20K' not in file and 'lab' not in file:
        f.write(file)
f.close()

f = open('river.yaml', 'w')
abs_path = os.path.abspath('.')
f.write(f'path: {abs_path}\n')
f.write('train: river_train.txt\n')
f.write('val: river_val.txt\n')
f.write('names:\n')
f.write('  0: water\n')
f.close()
