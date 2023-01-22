import os
import numpy as np
import cv2
import torch
import time
import shutil

CITYSCAPES_TO_COMMON = [0, 3, 3, 3, 19, 20, 14, 7, 8, 19, 16, 1, 11, 2, 17, 15, 3, 5, 5, 18, 12, 9, 22, 13, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10]
CARLA_TO_COMMON = [0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 22]
COMMON_COLOR_MAPPING = [
    [0,0,0],
    [70,70,70],
    [100,40,40],
    [55,90,80],
    [220,20,60],
    [153,153,153],
    [15,234,50],
    [128,64,128],
    [244,35,232],
    [107,142,35],
    [0,0,142],
    [102,102,156],
    [220,220,0],
    [70,130,180],
    [81,0,81],
    [150,100,150],
    [230,150,140],
    [180,165,180],
    [250,170,30],
    [110,190,160],
    [170,120,50],
    [45,60,150],
    [145,170,100],
]

def display_color_encoded_image(ids_image, color_mapping):
    color_encoded_image = np.zeros([ids_image.shape[0], ids_image.shape[1], 3], dtype=np.uint8)
    for i in range(ids_image.shape[0]):
        for j in range(ids_image.shape[1]):
            color_encoded_image[i][j] = color_mapping[ids_image[i][j]]
    cv2.imshow("Color encoded image", color_encoded_image)
    cv2.waitKey()

def map_value(map_list, value):
    return map_list[value]

def map_image(ids_image, ids_map):
    flattened_ids_image = np.ndarray.flatten(ids_image)
    mapped_image = np.array([map_value(ids_map, pixel) for pixel in flattened_ids_image])
    return mapped_image.reshape(ids_image.shape)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_root_folder = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\train'
    output_root_folder = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real_common_ids'

    index = -1
    for root, dirs, files in os.walk(f'{input_root_folder}\\semantic_segmentation'):
        for name in files:
            if name.endswith((".png")):
                start_time = time.time()
                path = f'{root}/{name}'
                index += 1
                image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
                id_mapped_image = map_image(image, CITYSCAPES_TO_COMMON)
                
                name_rgb = name.split('_')
                name_rgb = f'{name_rgb[0]}_{name_rgb[1]}_{name_rgb[2]}_leftImg8bit.png'
                shutil.copy(f'{input_root_folder}\\rgb\\{name_rgb}', f'{output_root_folder}\\rgb\\adcs_rgb_{index}.png')
                cv2.imwrite(f'{output_root_folder}\\semantic_segmentation\\adcs_semseg_{index}.png', id_mapped_image)
                print(f'Index: {index } | Name: {path} | Process time: {time.time() - start_time:.2f}')

if __name__ == "__main__":
    main()