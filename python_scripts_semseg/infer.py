import torch
import torchvision
import torchvision.models.segmentation.deeplabv3 as dlv3
import torchvision.transforms.functional as tf
import numpy as np
import data
from PIL import Image
from matplotlib import pyplot as plt

def main():
    # File parameters
    model_parameters_load_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\python_scripts_semseg\params\deeplabv3_model.pt'
    sample_input_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\python_scripts_semseg\pred_sample_images\sample_1.png'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dlv3.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=len(data.SemanticLabelMapper.ID_TO_STRING['common'].keys()))
    model.load_state_dict(torch.load(model_parameters_load_path))
    model.eval()
    model.to(device)
    
    pil_sample_input_image = Image.open(f'{sample_input_path}', mode='r')
    sample_input_image = tf.to_tensor(np.array(pil_sample_input_image).astype(np.float32))
    transform = torchvision.transforms.Resize((256, 512))
    sample_input_image = transform(sample_input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(sample_input_image)['out']
        pred = pred.data.cpu().numpy()[0]
        pred_image = np.argmax(pred, axis=0)

        plt.figure()
        plt.imshow(pil_sample_input_image)
        
        plt.figure()
        plt.imshow(pred_image)
        plt.show()

if __name__ == "__main__":
    main()