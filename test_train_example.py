import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from train_example import train_model

# Define the teacher (weak model) and student (strong model)
teacher_model = resnet18(pretrained=False)  # Using pretrained=False for simplicity
student_model = resnet50(pretrained=False)

# Define a simple synthetic dataset resembling CIFAR-100
# CIFAR-100 has 32x32 images with 3 channels
# We'll create random data with similar dimensions
num_samples = 1000
input_channels = 3
input_height = 32
input_width = 32
num_classes = 100

# Generate random images and labels
images = torch.randn(num_samples, input_channels, input_height, input_width)
labels = torch.randint(0, num_classes, (num_samples,))

# Create DataLoader for the synthetic dataset
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define optimizer and device
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a unit test function
def test_training_loop():
    train_model(student_model, teacher_model, dataloader, optimizer, device)


# Run the test function
test_training_loop()
print("test passed!")
