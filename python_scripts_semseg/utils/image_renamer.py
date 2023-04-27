import shutil
import os
from PIL import Image

index = 0
path = r'E:\Datasets\synthetic\train\rgb'
print(len(os.listdir(path)))
for file in os.listdir(path):
    if file.endswith('.png'):
        # index = int(file.split('.')[0].split('_')[-1])
        index = int(file.split('_')[0])
        print(index)

        new_file = f'synthetic_rgb_{index}.png'
        os.rename(f'{path}/{file}', f'{path}/{new_file}')
        
        # image = Image.open(path + file)
        # image_resized = image.resize((1434, 717))
        # image_resized.save(path + file)
