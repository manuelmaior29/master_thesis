import data
import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.models.segmentation.deeplabv3 as dlv3
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchsummary import summary
from tqdm import tqdm

def validate_epoch(model, val_dataloader, loss_function):
    model.eval()
    batch_losses = []
    metrics = {'miou': 0, 'loss': 0}
    
    with torch.no_grad():
        for (inputs, targets) in tqdm(val_dataloader, desc=f'Validation batch progress'):
            ipts = torch.autograd.Variable(inputs).cuda()
            tgts = torch.autograd.Variable(targets).cuda()
            
            preds = model(ipts)['out']
            loss = loss_function(preds, tgts.squeeze(1).long())
            loss = loss.data.cpu().numpy()
            batch_losses += [loss]

            preds = torch.argmax(preds.cpu(), dim=1)
            tgts = torch.squeeze(targets, dim=1)

            metrics['miou'] += sum(utils.iou(preds, tgts, num_classes=20)) / 20
            metrics['loss'] += loss

        metrics['miou'] /= float(len(val_dataloader))
        metrics['loss'] /= float(len(val_dataloader))
    return metrics

def train_epoch(model, train_dataloader, loss_function, optimizer):
    model.train()
    batch_losses = []
    
    for (inputs, targets) in tqdm(train_dataloader, desc=f'Training batch progress'):
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

def train(model, train_dataloader, val_dataloader, epochs, loss_function, optimizer):
    model.train()
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_mious = []
    for _ in tqdm(range(epochs), desc='Epoch progress'):
        batch_train_losses = train_epoch(
            model=model, 
            train_dataloader=train_dataloader,
            loss_function=loss_function, 
            optimizer=optimizer)
        
        batch_val_metrics = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function)

        epoch_average_train_loss = np.mean(batch_train_losses)
        epoch_train_losses += [epoch_average_train_loss]
        epoch_val_losses += [batch_val_metrics['loss']]
        epoch_val_mious += [batch_val_metrics['miou']]

        print(f'\n\n[TRAIN] Epoch average loss: {epoch_average_train_loss:.2f}')
        print(f'[VAL] Epoch average loss: {batch_val_metrics["loss"]:.2f}')
        print(f'[VAL] Epoch average miou: {batch_val_metrics["miou"]:.2f}\n')
    
    plt.plot(epoch_train_losses, label='Train loss', color='blue') 
    plt.plot(epoch_val_losses, label='Validation loss', color='yellow') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Loss over Epochs') 
    plt.legend()
    plt.show()
 
    plt.plot(epoch_val_mious, label='Validation mIoU', color='green') 
    plt.xlabel('Epoch') 
    plt.ylabel('mIoU') 
    plt.title('mIoU over Epochs') 
    plt.legend()
    plt.show()

def prepare_training(epochs, batch_size, subset_size, image_width, image_height):
    # TODO: Parametrize code so that it can be called from a driver
    pass

def main():

    argparser = argparse.ArgumentParser(description='DeepLabV3 - Train')
    argparser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning')
    argparser.add_argument('--fine-tune-model-path', default='./params/deeplabv3_model.pt', help='Path to the model to be fine-tuned')
    argparser.add_argument('--data-source', type=str, choices=['real', 'synthetic'], help='Data acquisition source')
    argparser.add_argument('--data-subset-size', type=int, help='Data subset size')
    argparser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    argparser.add_argument('--learning-rate', type=float, help='Training learning rate')
    argparser.add_argument('--batch-size', type=int, help='Size of a training data batch')
    argparser.add_argument('--image-width', type=int, help='Width of an image')
    argparser.add_argument('--image-height', type=int, help='Height of an image')
    args = argparser.parse_args()

    torch.cuda.empty_cache()

    # Misc parameters
    model_parameters_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\python_scripts_semseg\params/deeplabv3_model.pt'
    fine_tune = False

    # Model training hyper-parameters configuration
    ignored_label = 19
    epochs = 12
    learning_rate = 0.0001

    # Data
    data_source = 'real'
    batch_size = 4
    subset_size = 2500
    image_width = 512
    image_height = 256

    train_dataset = data.HybridDataset(
        root_path=f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\{data_source}\\train',
        input_dir='rgb',
        target_dir='semantic_segmentation_mapped',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_height, image_width))
        ]),
        type=data_source,
        labels_mapping=None
    )
    train_dataset = Subset(train_dataset, indices=range(subset_size))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    val_dataset = data.HybridDataset(
        root_path=f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\{data_source}\\val',
        input_dir='rgb',
        target_dir='semantic_segmentation_mapped',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_height, image_width))
        ]),
        type=data_source,
        labels_mapping=None
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
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
        val_dataloader=val_dataloader,
        epochs=epochs, 
        loss_function=loss_function, 
        optimizer=optimizer
    )

    # Save parameters of the model
    torch.save(model.state_dict(), f'{model_parameters_path}',)

if __name__ == "__main__":
    main()