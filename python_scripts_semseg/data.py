import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf

class SemanticLabelMapper():
    
    MAPPING = {
        'carla_to_common': [
            0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 22
        ],
        'cityscapes_to_common': [
            0, 3, 3, 3, 19, 20, 14, 7, 8, 19, 16, 1, 11, 2, 17, 15, 3, 5, 5, 18, 12, 9, 22, 13, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10
        ]
    }

    def __init__(self, type=None) -> None:
        super().__init__()
        self.type = type

    def __map_value(self, pixel):
        return SemanticLabelMapper.MAPPING[self.type][pixel]

    def map_image(self, input):
        flattened_image = np.ndarray.flatten(input)
        mapped_image = np.array([self.__map_value(pixel) for pixel in flattened_image])
        return mapped_image.reshape(input.shape)

class HybridDataset(Dataset):

    def __init__(self, root_path, input_dir, target_dir, transform=None, type='real', labels_mapping=None) -> None:
        super(HybridDataset, self).__init__()
        self.root_path = root_path
        self.input_data = input_dir
        self.target_data = target_dir
        self.transform = transform
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
        
        ipt_patch = np.array(Image.open(img_path_input_patch, 'r')).astype(np.float32)
        tgt_patch = np.array(Image.open(img_path_tgt_patch, 'r'))

        if self.labels_mapping is not None:
            try:
                semantic_label_mapper = SemanticLabelMapper(self.labels_mapping)
                tgt_patch = semantic_label_mapper.map_image(tgt_patch)
            except Exception as e:
                raise Exception(f'Could not perform label mapping!\n {e}')
        tgt_patch.astype(np.float32)
        np.expand_dims(tgt_patch, axis=0)
            
        ipt_patch_tensor = tf.to_tensor(ipt_patch)
        tgt_patch_tensor = tf.to_tensor(tgt_patch)
        
        if self.transform:
            ipt_patch_tensor = self.transform(ipt_patch_tensor)
            tgt_patch_tensor = self.transform(tgt_patch_tensor)
            
        return ipt_patch_tensor, tgt_patch_tensor
    
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
