import shutil
import os

index = 0
for root, dirs, files in os.walk(r'D:\Research\CityScapes_SemSeg\leftImg8bit_trainvaltest\leftImg8bit\test'):
    for name in files:
        if name.endswith(".png"):
            shutil.move(f'{root}/{name}', r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\test\rgb')
# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file

#         if filepath.endswith(".html"):
#             print (filepath)