import os

images_folder = "roadrunner_photos/images"
f = open('data.txt', 'a')
for filename in os.listdir(images_folder):
    f.write(f"{os.path.join(images_folder, filename)}\n")
f.close()