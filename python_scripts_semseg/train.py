import data
import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.models.segmentation.deeplabv3 as dlv3
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchsummary import summary
from tqdm import tqdm

def train_epoch(model, train_dataloader, loss_function, optimizer):
    model.train()
    batch_losses = []
    
    for (inputs, targets) in tqdm(train_dataloader, desc=f'Batch progress'):
        ipts = torch.autograd.Variable(inputs).cuda()
        tgts = torch.autograd.Variable(targets).cuda()
        pred = model(ipts)['out']

        loss = loss_function(pred, tgts.squeeze(1).long())
        loss_val = loss.data.cpu().numpy()
        batch_losses += [loss_val]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_losses

def train(model, train_dataloader, epochs, loss_function, optimizer):
    model.train()
    epoch_losses = []
    for _ in tqdm(range(epochs), desc='Epoch progress'):
        batch_losses = train_epoch(
            model=model, 
            train_dataloader=train_dataloader,
            loss_function=loss_function, 
            optimizer=optimizer
        )
        epoch_average_loss = np.mean(batch_losses)
        epoch_losses += [epoch_average_loss]
        print(f'Epoch average loss: {epoch_average_loss:.2f}')
    
    plt.plot(epoch_losses) 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Loss over Epochs') 
    plt.show() 

def main():
    torch.cuda.empty_cache()

    # Misc parameters
    model_parameters_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\python_scripts_semseg\params/deeplabv3_model.pt'
    fine_tune = False

    # Model training hyper-parameters configuration
    ignored_label = 255
    epochs = 8
    learning_rate = 0.0001

    # Data
    batch_size = 4
    subset_size = 1000
    image_width = 512
    image_height = 256

    train_dataset = data.HybridDataset(
        root_path=r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real\train',
        input_dir='rgb',
        target_dir='semantic_segmentation_mapped',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_height, image_width))
        ]),
        type='real',
        labels_mapping=None
    )
    train_dataset = Subset(train_dataset, indices=range(subset_size))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    
    model = dlv3.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=len(data.SemanticLabelMapper.ID_TO_STRING['common'].keys()))
    if fine_tune:
        model.load_state_dict(torch.load(model_parameters_path))
        for name, param in model.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name or 'layer5' in name:
                print(f'---> Freezing layer: {name}.')
                param.requires_grad = False

        # Model training configuration (fine-tuning)
        learning_rate = 0.00001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training functions configuration
    params = utils.add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params=params, lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=ignored_label)

    # Start training
    train(
        model=model, 
        train_dataloader=train_dataloader, 
        epochs=epochs, 
        lr=learning_rate,
        loss_function=loss_function, 
        optimizer=optimizer
    )

    # Save parameters of the model
    torch.save(model.state_dict(), f'{model_parameters_path}',)

if __name__ == "__main__":
    main()