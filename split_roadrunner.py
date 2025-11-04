import random
import os

f = open('data.txt', 'r')
all_files = f.readlines()
f.close()
rate = 0.2

dataset = ['roadrunner_photos/images/' + file + '\n' for file in os.listdir('roadrunner_photos/images') if file.endswith('.png')]

random.shuffle(dataset)
split_index = int(len(dataset) * rate)
val_set = dataset[:split_index]
train_set = dataset[split_index:]

f = open('roadrunner_train.txt', 'w')
f.writelines(train_set)
f.close()

f = open('roadrunner_val.txt', 'w')
f.writelines(val_set)
f.close()

f = open('roadrunner.yaml', 'w')
f.write('train: roadrunner_train.txt\n')
f.write('val: roadrunner_val.txt\n')
f.write('names:\n')
f.write('  0: water\n')
f.close()

print(f"Train files: {len(train_set)}")
print(f"Validation files: {len(val_set)}")