import data
import torch
import torchvision
import sklearn.metrics as skm
import torchvision.models.segmentation.deeplabv3 as dlv3
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm


def eval(model, test_dataloader, num_classes):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for (inputs, targets) in tqdm(test_dataloader, desc='Batch progress'):
            ipts = torch.autograd.Variable(inputs).cuda()
            tgts = torch.autograd.Variable(targets).cuda()
            
            preds = model(ipts)['out']
            _, preds = torch.max(preds, dim=1)

            preds = preds.cpu().numpy().flatten()
            tgts = tgts.cpu().numpy().flatten()
            
            batch_confusion_matrix = skm.confusion_matrix(tgts, preds, labels=list(range(num_classes)))
            confusion_matrix += batch_confusion_matrix
        
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    iou = tp / (tp + fp + fn)
    miou = iou.mean()

    print(f'mIoU: {miou.item()}')

def main():
    # File parameters
    model_parameters_load_path = r'C:\Users\Manuel\Projects\GitHub_Repositories\master_thesis\python_scripts_semseg\params\deeplabv3_model.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dlv3.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=len(data.SemanticLabelMapper.ID_TO_STRING['common'].keys()))
    model.load_state_dict(torch.load(model_parameters_load_path))
    model.to(device)

    # Data
    data_source = 'real'
    batch_size = 4
    subset_size = 500
    image_width = 512
    image_height = 256

    test_dataset = data.HybridDataset(
        root_path=f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\{data_source}\\val',
        input_dir='rgb',
        target_dir='semantic_segmentation_mapped',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_height, image_width))
        ]),
        type=data_source,
        labels_mapping=None
    )
    test_dataset = Subset(test_dataset, indices=range(subset_size))
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    eval(model=model, test_dataloader=test_dataloader, num_classes=len(data.SemanticLabelMapper.ID_TO_STRING['common'].keys()))

if __name__ == "__main__":
    main()