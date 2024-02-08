import os
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import json
import torch

def generate_label_for_single_image(image_path):
    # Initialize the model and processor
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predict the class of the image
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Optionally, print the predicted class (if id2label is available)
    predicted_label = model.config.id2label[predicted_class_idx] if model.config.id2label else str(predicted_class_idx)

    print("Predicted class index:", predicted_class_idx)
    print("Predicted label:", predicted_label)

    return predicted_class_idx, predicted_label


def generate_labels(image_dir, model_name, output_file, max_images=100):
    # Load the CLIP model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Fetch image paths
    images = os.listdir(image_dir)
    images = images[:max_images]  # Limit to max_images

    labels = []

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt")

        # Generate image features (embeddings)
        with torch.no_grad():
            outputs = model(**inputs)

        # Here you would typically use the embeddings to find the closest text descriptions, but since
        # OpenCLIP doesn't provide direct label generation, we'll just save the embeddings for now.
        # This step is a placeholder for any specific label generation or nearest neighbor search you might implement.
        # For demonstration, let's assume a dummy label based on the max value index in the embeddings.
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()

        labels.append({
            "image_path": image_name,
            "label": predicted_class_idx
        })
        print("Predicted class: ", model.config.id2label[predicted_class_idx])
    # Save the labels to a JSON file
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=4)

image_dir = './images'
model_name = 'google/vit-base-patch16-224'
output_file = 'imagelabels.json'
