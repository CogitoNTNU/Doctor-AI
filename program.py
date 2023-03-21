import torch
import os
import torchvision.transforms as transforms
from PIL import Image

dataPath = "../data/chest_xray/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using ImageNet mean and std
])

# Hyperparamters
batchSize = 16

# Import jpeg files
def import_image(path, transform=None):
    image = Image.open(path).convert('RGB')
    
    if transform:
        image = transform(image)

    return torch.flatten(image)

def get_batch(path, batchSize, transform):
    
    for label in ["NORMAL", "PNEUMONIA"]:
        files = os.listdir(os.path.join(path, label))
        idxs = torch.randint(0, len(files), (batchSize,))
        images, labels = [], []
        for idx in idxs:
            images.append(import_image(os.path.join(path, label, files[idx]), transform=transform))
            labels.append(torch.Tensor(1 if label == "PNEUMONIA" else 0)) # TODO
        imageTensors = torch.stack(images)
        labelTensors = torch.stack(labels)
            
get_batch(dataPath, batchSize, transform)