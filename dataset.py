import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from generate_labels import generate_label_for_single_image
import json


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(
        self, image_dir, transform=None, model_name="google/vit-base-patch16-224"
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.memory_bank = load_memory_bank()  # Load existing labels from memory bank
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.model_name = model_name
        self.label_to_index = {}  # Dynamically map labels to indices
        self.current_index = 0  # Keep track of the next index to assign

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if image_path not in self.memory_bank:  # Check if label already exists
            # Generate label for the image
            _, predicted_label = generate_label_for_single_image(image_path)
            self.memory_bank[image_path] = predicted_label  # Add to memory bank
        else:
            predicted_label = self.memory_bank[image_path]  # Use existing label
        # if self.transform:
        #    image = self.transform(image)

        # Dynamically assign index to new labels
        if predicted_label not in self.label_to_index:
            self.label_to_index[predicted_label] = self.current_index
            self.current_index += 1
        label_index = self.label_to_index[predicted_label]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_index, dtype=torch.long)


def load_memory_bank(filename="label_memory_bank.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return json.load(file)
    else:
        return {}
