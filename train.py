import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from utils import plot_and_save_metrics
from dataset import CustomImageDataset
from AdaptiveConfidenceDistillation import AdaptiveConfidenceDistillationLoss

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
num_classes = 100

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

# Create an instance of the CustomImageDataset
image_dir = "./images"  # Specify the correct path to your images
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)


def save_memory_bank(memory_bank, filename="label_memory_bank.json"):
    with open(filename, "w") as file:
        json.dump(memory_bank, file)


# After training is complete or when you're done using the dataset
save_memory_bank(dataset.memory_bank)  # Save the updated memory bank to a file
# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define optimizer and device
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Step 1: Load Labels and Image Paths
def load_image_paths_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    image_paths = [os.path.join("./images", item["image_path"]) for item in data]
    labels = [item["caption"] for item in data]
    return image_paths, labels


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

    # Initialize metrics dictionary
    metrics = {
        "train_loss": [],
        "val_loss": [],
    }  # Add 'train_accuracy': [], 'val_accuracy': [] if tracking accuracy
    loss_fn = AdaptiveConfidenceDistillationLoss(temperature=2.0)
    best_val_loss = float("inf")
    step = 0
    for epoch in range(num_epochs):
        # Training loop
        for inputs, labels in train_dataloader:
            if step % 200 == 0:
                # Validation loop
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        teacher_outputs = teacher_model(inputs)
                        student_outputs = student_model(inputs)
                        loss = loss_fn(student_outputs, teacher_outputs, labels)
                        val_loss += loss.item() * inputs.size(
                            0
                        )  # Inside the training loop
                        metrics["train_loss"].append(
                            loss.item()
                        )  # Assuming loss is your loss variable

                        # Inside the validation loop
                        metrics["val_loss"].append(
                            val_loss
                        )  # Assuming val_loss is your accumulated validation loss
                        step += 1
                        print(f"val_loss: {val_loss}")
                # After training, plot and save metrics
                plot_and_save_metrics(metrics)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)

            loss = loss_fn(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss /= len(val_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), checkpoint_path)
            # After training, plot and save metrics
            plot_and_save_metrics(metrics)
            print(f"Saving model checkpoint at {checkpoint_path}")


# Example usage
# Assuming `train_dataloader`, `val_dataloader`, `optimizer`, and `device` are defined
train_model_with_validation(
    student_model,
    teacher_model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    num_epochs=10,
    checkpoint_path="checkpoint.pth",
)
