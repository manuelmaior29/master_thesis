import data
import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.models.segmentation.deeplabv3 as dlv3
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary

def train_epoch(model, train_loader, lr, loss_function, optimizer):
    model.train()
    batch_losses = []
    for batch_index, (inputs, targets) in enumerate(train_loader):
        print(f'----------[Batch {batch_index}]----------')

        ipts = torch.autograd.Variable(inputs).cuda()
        tgts = torch.autograd.Variable(targets).cuda()
        pred = model(ipts)['out']

        # TODO: Check target data and correctness of loss computation
        loss = loss_function(pred, tgts.squeeze(1).long())
        loss_val = loss.data.cpu().numpy()
        batch_losses += [loss_val]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_losses

def train(model, train_loader, epochs, lr, loss_function, optimizer):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        print(f'----------[Epoch {epoch}]----------')
        batch_losses = train_epoch(
            model=model, 
            train_loader=train_loader, 
            lr=lr, 
            loss_function=loss_function, 
            optimizer=optimizer
        )
        epoch_losses += [np.mean(batch_losses)]
    
    plt.plot(epoch_losses) 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Loss over Epochs') 
    plt.show() 

def main():
    torch.cuda.empty_cache()

    # Misc parameters
    model_parameters_save_path = r'./params/deeplabv3_model.pt'

    # Model
    model = dlv3.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=35)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model parameters
    batch_size = 4
    epochs = 16
    learning_rate = 0.0001
    params = utils.add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params=params, xlr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # Data
    train_dataset = data.AdaptedCityscapesDataset(
        root_path=r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\datasets\real_common_ids',
        input_dir='rgb',
        target_dir='semantic_segmentation',
        target_scale=None,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 512))
        ])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    # Start training
    train(
        model=model, 
        train_loader=train_dataloader, 
        epochs=epochs, 
        lr=learning_rate,
        loss_function=loss_function, 
        optimizer=optimizer
    )

    # Save parameters of the model
    torch.save(model.state_dict(), f'{model_parameters_save_path}',)

if __name__ == "__main__":
    main()