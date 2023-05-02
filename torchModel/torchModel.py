# Import requirements
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Define datapath
dataPath = os.path.join("..", "data", "chest_xray", "train")

batchSize = 16

# Define tranformer for images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using ImageNet mean and std
])


# Import jpeg files
def import_image(path, transform=None):

    return transform(Image.open(path).convert('RGB')) if transform else Image.open(path).convert('RGB')

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

# Create DataLoader 
dataset = ChestXRayDataset(dataPath, transform)
dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

# Define nerual network model
model = torchvision.models.resnet18(pretrained=True, progress=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming two classes: NORMAL and PNEUMONIA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize variables for storing loss and accuracy values
loss_values = []
accuracy_values = []

num_epochs = 5 # Number of training epochs
print_every_n_batches = 10  # Adjust this value to control the frequency of progress updates

for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    progress_bar = tqdm(enumerate(dataLoader), total=len(dataLoader), desc=f"Epoch {epoch + 1}/{num_epochs}")
    
    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update loss
        running_loss += loss.item() * inputs.size(0)

        # Update correct predictions count
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        # Update total samples count
        total_samples += inputs.size(0)

        # Update progress bar
        progress_bar.set_postfix(loss=(running_loss / total_samples), accuracy=(running_corrects.double() / total_samples))
        
        # Save loss and accuracy values for each batch
        loss_values.append(running_loss / total_samples)
        accuracy_values.append((running_corrects.double() / total_samples).item())

    # Calculate average loss and accuracy for the current epoch
    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_corrects.double() / total_samples

print("Training complete.")

# Save the trained model
savePath= "pneumoniaV4.pth"
torch.save(model.state_dict(), savePath)
print(f"Model saved to {os.path.abspath(savePath)}")

# Plot the loss and accuracy
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(loss_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(accuracy_values, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)

fig.tight_layout()
plt.show()

