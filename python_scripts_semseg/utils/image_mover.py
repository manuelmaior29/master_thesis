import shutil
import os

index = 0
for root, dirs, files in os.walk(r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\test'):
    for name in files:
        if name.endswith("gtFine_color.png"):
            shutil.move(f'{root}/{name}', r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\test\rgb')