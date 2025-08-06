import random
import os

f = open('data.txt', 'r')
files = f.readlines()
f.close()

rate = 0.2

random.shuffle(files)
train_files = files[:int(len(files) * (1 - rate))]
val_files = files[int(len(files) * (1 - rate)):]

f = open('train.txt', 'w')
for file in train_files:
    f.write(file)
f.close()

f = open('val.txt', 'w')
for file in val_files:
    f.write(file)
f.close()

f = open('water.yaml', 'w')
abs_path = os.path.abspath('.')
f.write(f'path: {abs_path}\n')
f.write('train: train.txt\n')
f.write('val: val.txt\n')
f.write('names:\n')
f.write('  0: water\n')
f.close()

print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")