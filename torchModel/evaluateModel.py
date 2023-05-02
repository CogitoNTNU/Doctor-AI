import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Define tranformer for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using ImageNet mean and std
])

# Data set class
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = ["NORMAL", "PNEUMONIA"]
        self.data = []
        for label in self.labels:
            class_dir = os.path.join(root_dir, label)
            for img_file in os.listdir(class_dir):
                self.data.append((os.path.join(class_dir, img_file), self.labels.index(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = import_image(img_path, self.transform)
        return image, label
    
# Import jpeg files
def import_image(path, transform=None):

    return transform(Image.open(path).convert('RGB')) if transform else Image.open(path).convert('RGB')

# Define data path
dataPath = "../data/chest_xray/test"

# Load the dataset
dataset = ChestXRayDataset(dataPath, transform=transform)
dataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(num_classes=2)
model = model.to(device)

# Load the saved weights
model_load_path = "pneumoniaV3.pth"
model.load_state_dict(torch.load(model_load_path))

# Set the model to evaluation mode
model.eval()

criterion = nn.CrossEntropyLoss()

# Initialize a variable for storing individual loss values
individual_losses = []

running_loss = 0
running_corrects = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in dataLoader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        running_corrects += (predicted == labels).sum().item()
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate and store per-sample loss values
        per_sample_losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
        individual_losses.extend(per_sample_losses.cpu().numpy())

print(running_corrects, total_samples)
avg_loss = running_loss / total_samples
accuracy = 100 * running_corrects / total_samples
print(f'Loss on the evaluation dataset: {avg_loss:.4f}')
print(f'Accuracy on the evaluation dataset: {accuracy:.2f}%')

# Normalize the individual losses to the range [0, 1]
individual_losses = np.array(individual_losses)
normalized_losses = (individual_losses - individual_losses.min()) / (individual_losses.max() - individual_losses.min())

# Create a scatter plot of normalized individual losses
plt.scatter(normalized_losses, normalized_losses, s=5, alpha=0.5)
plt.xlabel("Normalized Loss")
plt.ylabel("Normalized Loss")
plt.title("Normalized Individual Losses for Each Prediction")
plt.show()