from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf
import torchvision
import cv2

class SemanticLabelMapper():
    
    ID_TO_STRING = {
        'common': {
            0: 'road',
            1: 'sidewalk',
            2: 'building',
            3: 'wall',
            4: 'fence',
            5: 'trafficlight',
            6: 'trafficsign',
            7: 'vegetation',
            8: 'terrain',
            9: 'pedestrian',
            10: 'rider',
            11: 'car',
            12: 'truck',
            13: 'bus',
            14: 'motorcycle',
            15: 'bicycle',
            16: 'background'
        }
    }

    ID_TO_COLOR = {
        'common': {
            0: (70, 70, 70),
            1: (100, 40, 40),
            2: (55, 90, 80),
            3: (220, 20, 60),
            4: (153, 153, 153),
            5: (157, 234, 50),
            6: (128, 64, 128),
            7: (244, 35, 232),
            8: (107, 142, 35),
            9: (0, 0, 142),
            10: (102, 102, 156),
            11: (220, 220, 0),
            12: (70, 130, 180),
            13: (81, 0, 81),
            14: (150, 100, 100),
            15: (230, 150, 140),
            16: (0, 0, 0)
        }
    }

    MAPPING = {
        'carla_to_common': [
            16, 0, 1, 2, 3, 4, 16, 5, 6, 7, 8, 16, 9, 10, 11, 12, 13, 16, 14, 15, 16, 16, 16, 16, 0, 16, 16, 16, 16
        ],
        'cityscapes_to_common': [
            16, 16, 16, 16, 16, 16, 16, 0, 1, 16, 16, 2, 3, 4, 16, 16, 16, 16, 16, 5, 6, 7, 8, 16, 9, 10, 11, 12, 13, 16, 16, 16, 14, 15, 16   
        ]
    }

    def __init__(self, type=None) -> None:
        super().__init__()
        self.type = type

    def __map_value(self, pixel):
        return SemanticLabelMapper.MAPPING[self.type][pixel]

    def map_image(self, input):
        return np.vectorize(self.__map_value)(input)
    
    def map_from_dir(self, src_path, dst_path, extension):
        for file in tqdm(os.listdir(src_path)):
            if file.endswith(extension):
                src_image_path = f'{src_path}/{file}'
                dst_image_path = f'{dst_path}/{file}'
                src_image = np.array(Image.open(src_image_path, 'r'))
                dst_image = self.map_image(src_image)            
                dst_image = Image.fromarray(np.uint8(dst_image), 'L')
                dst_image.save(dst_image_path)

class HybridDataset(Dataset):

    def __init__(self, root_path, input_dir, target_dir, ipt_transform=None, tgt_transform=None, type='real', labels_mapping=None) -> None:
        super(HybridDataset, self).__init__()
        self.root_path = root_path
        self.input_data = input_dir
        self.target_data = target_dir
        self.ipt_transform = ipt_transform
        self.tgt_transform = tgt_transform
        self.type = type
        self.labels_mapping = labels_mapping
    
    def __len__(self):
        input_file_list = os.listdir(os.path.join(self.root_path, self.input_data))
        target_file_list = os.listdir(os.path.join(self.root_path, self.target_data))
        input_length = len(input_file_list)
        target_length = len(target_file_list)
        if target_length == input_length:
            return target_length

    def __getitem__(self, index):
        img_path_input_patch = os.path.join(self.root_path, self.input_data, f"{self.type}_rgb_{index}.png")
        img_path_tgt_patch = os.path.join(self.root_path, self.target_data, f"{self.type}_semantic_segmentation_{index}.png")
        
        ipt_patch = np.array(Image.open(img_path_input_patch, 'r'))
        tgt_patch = np.array(Image.open(img_path_tgt_patch, 'r',)).astype(np.int_)

        ipt_patch = torchvision.transforms.ToTensor()(ipt_patch)
        tgt_patch = tf.to_tensor(tgt_patch)

        if self.labels_mapping is not None:
            try:
                semantic_label_mapper = SemanticLabelMapper(self.labels_mapping)
                tgt_patch = semantic_label_mapper.map_image(tgt_patch)
            except Exception as e:
                raise Exception(f'Could not perform label mapping!\n {e}')
        # tgt_patch.astype(np.float32)
        np.expand_dims(tgt_patch, axis=0)
            
        if self.ipt_transform:
            ipt_patch = self.ipt_transform(ipt_patch)

        if self.tgt_transform:
            tgt_patch = self.tgt_transform(tgt_patch)
            
        return ipt_patch, tgt_patch
    
# Helper function to show a batch
def show_landmarks_batch(dataloader):
    for input_batch, target_batch in dataloader:
        input = input_batch[0].cpu().numpy().swapaxes(0,2).swapaxes(0,1) / 255.0
        target = target_batch[0].cpu().numpy().swapaxes(0,2).swapaxes(0,1)

        fig = plt.figure(figsize=(1,2))
        fig_cols = 2
        fig_rows = 1

        fig.add_subplot(fig_rows, fig_cols, 1)
        plt.imshow(input)
        plt.axis('off')
        plt.title('RGB')

        fig.add_subplot(fig_rows, fig_cols, 2)
        plt.imshow(target)
        plt.axis('off')
        plt.title('Semantic segmentation')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.plot()
        plt.show()

def test():
    custom_real_dataset = HybridDataset(
        root_path=r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\train',
        input_dir='rgb',
        target_dir='semantic_segmentation',
        transform=None,
        type='real',
        labels_mapping='cityscapes_to_common')

    dataloader = DataLoader(custom_real_dataset, batch_size=4, shuffle=False)
    show_landmarks_batch(dataloader)

def perform_image_mapping(src_path, dst_path, mapping_type):
    slm = SemanticLabelMapper(mapping_type)
    slm.map_from_dir(src_path=src_path, dst_path=dst_path, extension='.png')
    
def visualize_class_distribution():
    dataset_path = r'G:\My Drive\Master IVA\Master Thesis\Datasets\real\train'
    dataset = HybridDataset(
        root_path=dataset_path,
        input_dir='rgb',
        target_dir='semantic_segmentation_mapped',
        ipt_transform=None,
        tgt_transform=None,
        type='real',
        labels_mapping=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    classes_keys = list(SemanticLabelMapper.ID_TO_STRING['common'].keys())
    classes_labels = list(SemanticLabelMapper.ID_TO_STRING['common'].values())
    classes_distribution = {}
    for key in classes_keys:
        classes_distribution[key] = 0

    pixel_count = 0
    batch_idx = 0
 
    for _, target_batch in dataloader:
        print(batch_idx)
        for target_map in target_batch:
            flattened_target_map = (torch.flatten(target_map)).long()
            labels_count = torch.bincount(flattened_target_map)
            for i in range(len(labels_count)):
                if labels_count[i] != 0:
                    classes_distribution[i] += labels_count[i].item()

        pixel_count += 1
        batch_idx += 1

    keys = list(classes_labels)
    values = list(classes_distribution.values())
    values_count = sum(values)

    # Plot the data using a bar plot
    plt.bar(keys, values)
    plt.xticks(keys, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class distribution')

    for i, v in enumerate(values):
        plt.text(i, v, str(round(v/values_count*100, 3)) + '%', ha='center')

    plt.show()


src_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\synthetic\val\semantic_segmentation'
dst_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\synthetic\val\semantic_segmentation_mapped'
mapping_type = 'carla_to_common'

# test()
# perform_image_mapping(src_path, dst_path, mapping_type)
# visualize_class_distribution()