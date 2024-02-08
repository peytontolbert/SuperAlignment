import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from PIL import Image
import os
import json
from generate_labels import generate_label_for_single_image
# Define the teacher (weak model) and student (strong model)
teacher_model = resnet18(pretrained=True)  # Using pretrained=False for simplicity
student_model = resnet50(pretrained=True)

# Define a simple synthetic dataset resembling CIFAR-100
# CIFAR-100 has 32x32 images with 3 channels
# We'll create random data with similar dimensions
num_samples = 1000
input_channels = 3
input_height = 32
input_width = 32
num_classes = 1000

# Step 1: Load Labels and Image Paths
def load_image_paths_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths = [os.path.join("./images", item['image_path']) for item in data]
    labels = [item['caption'] for item in data]
    return image_paths, labels

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, model_name='google/vit-base-patch16-224'):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.model_name = model_name
        self.label_to_index = {}  # Dynamically map labels to indices
        self.current_index = 0  # Keep track of the next index to assign

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Generate label for the image
        _ , predicted_label = generate_label_for_single_image(image_path)
        
        #if self.transform:
        #    image = self.transform(image)
        
        # Dynamically assign index to new labels
        if predicted_label not in self.label_to_index:
            self.label_to_index[predicted_label] = self.current_index
            self.current_index += 1
        label_index = self.label_to_index[predicted_label]
        

        if self.transform:
            image = self.transform(image)        
        return image, torch.tensor(label_index, dtype=torch.long)

# Create an instance of the CustomImageDataset
image_dir = './images'  # Specify the correct path to your images
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define optimizer and device
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Adaptive Confidence Distillation Loss
class AdaptiveConfidenceDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Soften probabilities
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Calculate the KL Divergence for the soft targets
        distillation_loss = F.kl_div(
            soft_student_probs, soft_teacher_probs, reduction="batchmean"
        )

        # Calculate the standard loss with hard targets
        hard_loss = self.ce_loss(student_logits, targets)

        # Adaptive weighting could be implemented here based on confidence
        # For simplicity, this example uses a fixed ratio
        return distillation_loss + hard_loss


# Example Training Loop Skeleton with Validation and Model Checkpointing
def train_model_with_validation(
    student_model,
    teacher_model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    num_epochs=10,
    checkpoint_path="checkpoint.pth",
):
    student_model.train()
    teacher_model.eval()  # Teacher model should be in eval mode

    loss_fn = AdaptiveConfidenceDistillationLoss(temperature=2.0)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training loop
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)

            loss = loss_fn(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                loss = loss_fn(student_outputs, teacher_outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"Saving model checkpoint at {checkpoint_path}")


# Example usage
# Assuming `train_dataloader`, `val_dataloader`, `optimizer`, and `device` are defined
train_model_with_validation(student_model, teacher_model, train_dataloader, val_dataloader, optimizer, device, num_epochs=10, checkpoint_path='checkpoint.pth')
