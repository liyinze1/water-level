import random
import os

f = open('data.txt', 'r')
all_files = f.readlines()
f.close()
rate = 0.2

dataset = []

for file in all_files:
    if 'ADE20K' not in file and 'lab' not in file and 'water_v2' in file:
        dataset.append(file)


roadrunner_files = ['roadrunner_photos/images/' + file + '\n' for file in os.listdir('roadrunner_photos/images') if file.endswith('.png')]
random.shuffle(roadrunner_files)

# add first 200 roadrunner images to dataset
dataset += roadrunner_files[:200]

# add remaining roadrunner images to test set
test_set = roadrunner_files[200:]

random.shuffle(dataset)
split_index = int(len(dataset) * rate)
val_set = dataset[:split_index]
train_set = dataset[split_index:]

f = open('water_train.txt', 'w')
f.writelines(train_set)
f.close()

f = open('water_val.txt', 'w')
f.writelines(val_set)
f.close()

f = open('water_test.txt', 'w')
f.writelines(test_set)
f.close()

f = open('water.yaml', 'w')
abs_path = os.path.abspath('.')
f.write('train: water_train.txt\n')
f.write('val: water_val.txt\n')
f.write('test: water_test.txt\n')
f.write('names:\n')
f.write('  0: water\n')
f.close()

print(f"Train files: {len(train_set)}")
print(f"Validation files: {len(val_set)}")
print(f"Test files: {len(test_set)}")