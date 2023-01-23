import shutil
import os

index = 0
path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\train\semantic_segmentation'
for file in os.listdir(path):
    if file.endswith('.png'):
        os.rename(f'{path}/{file}', f'{path}/real_semantic_segmentation_{index}.png')
        index += 1