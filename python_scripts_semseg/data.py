import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf

class AdaptedCityscapesDataset(Dataset):

    def __init__(self, root_path, input_dir, target_dir, target_scale, transform=None):
        super(AdaptedCityscapesDataset, self).__init__()
        self.root_path = root_path
        self.input_data = input_dir
        self.target_data = target_dir
        self.target_scale = target_scale
        self.transform = transform
    
    def __len__(self):
        input_file_list = os.listdir(os.path.join(self.root_path, self.input_data))
        target_file_list = os.listdir(os.path.join(self.root_path, self.target_data))
        input_length = len(input_file_list)
        target_length = len(target_file_list)
        if target_length == input_length:
            return target_length

    def __getitem__(self, index):
        img_path_input_patch = os.path.join(self.root_path, self.input_data, f"adcs_rgb_{index}.png")
        img_path_tgt_patch = os.path.join(self.root_path, self.target_data, f"adcs_semseg_{index}.png")
        
        ipt_patch = np.array(Image.open(img_path_input_patch, 'r')).astype(np.float32)
        tgt_patch = np.array(Image.open(img_path_tgt_patch, 'r')).astype(np.float32)
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
        input = input_batch[0].cpu().numpy().swapaxes(0,2).swapaxes(0,1)
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
        plt.title('Sem. seg.')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.plot()
        plt.show()

def test():
    adcs_dataset = AdaptedCityscapesDataset(
        root_path=r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real_common_ids',
        input_dir='rgb',
        target_dir='semantic_segmentation',
        target_scale=None,
        train_transform=None)

    dataloader = DataLoader(adcs_dataset, batch_size=4, shuffle=True)
    show_landmarks_batch(dataloader)
