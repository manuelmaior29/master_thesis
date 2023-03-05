import shutil
import os

index = 0
path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\val\rgb'
for file in os.listdir(path):
    if file.endswith('.png'):
        os.rename(f'{path}/{file}', f'{path}/real_rgb_{index}.png')
        index += 1