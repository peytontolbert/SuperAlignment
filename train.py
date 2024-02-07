import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

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

# Split data into train and validation sets
train_split = int(0.8 * num_samples)
train_images, val_images = images[:train_split], images[train_split:]
train_labels, val_labels = labels[:train_split], labels[train_split:]

# Create DataLoader for the synthetic dataset
train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define optimizer and device
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# train_model_with_validation(student_model, teacher_model, train_dataloader, val_dataloader, optimizer, device, num_epochs=10, checkpoint_path='checkpoint.pth')
