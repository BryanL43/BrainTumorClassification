from PIL import Image, ImageFilter
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import random

from Preprocessor import Preprocessor

def get_random_image_path(root_dir="./DataSet/Training"):
    # Get all subdirectories
    tumor_types = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))];
    if not tumor_types:
        raise ValueError("No tumor type folders found in the Training directory.");

    # Pick a random tumor type
    selected_type = random.choice(tumor_types);
    type_path = os.path.join(root_dir, selected_type);

    # Get all image files in that tumor type folder
    image_files = [f for f in os.listdir(type_path)];
    if not image_files:
        raise ValueError(f"No images found in {type_path}");

    # Pick a random image
    selected_image = random.choice(image_files);

    # Full image path
    return os.path.join(type_path, selected_image), selected_type;

def apply_randomness(img, scale=(0.7, 1.3), rotation_degrees=15):
    original_w, original_h = img.size;
    zoom_scale = random.uniform(scale[0], scale[1]);

    # Compute new padded size
    pad_w = int(original_w * zoom_scale);
    pad_h = int(original_h * zoom_scale);

    # Center pad
    pad_left = (pad_w - original_w) // 2;
    pad_top = (pad_h - original_h) // 2;
    img_padded = F.pad(img, padding=[pad_left, pad_top, pad_left, pad_top], fill=0);

    # Random crop with original dimensions (no enforced resize)
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        img_padded, scale=(1.0, 1.0), ratio=(1.0, 1.0)
    );
    cropped = F.resized_crop(img_padded, i, j, h, w, size=(original_h, original_w), interpolation=F.InterpolationMode.BILINEAR);

    # Apply rotation
    angle = random.uniform(-rotation_degrees, rotation_degrees);
    rotated = F.rotate(cropped, angle=angle, interpolation=F.InterpolationMode.BILINEAR);

    print(f"Zoomed out to {zoom_scale:.2f}x with padding");
    print(f"Rotated by {angle:.2f} degrees");
    return rotated;

# Denormalize the image for final output [TEMP]
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1);
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1);
    return tensor * std + mean;

def main():
    # Instantiate Preprocessor with specified parameters
    preprocessor = Preprocessor();

    # Image processing pipeline
    transform_pipeline = transforms.Compose([
        lambda img: apply_randomness(img, scale=(0.7, 1.3), rotation_degrees=15),
        transforms.Resize((224, 224)),
        preprocessor,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225]   # ImageNet std
        )
    ]);

    # img_path = "./DataSet/Training/meningioma_tumor/m (5).jpg";
    # img = Image.open(img_path).convert("RGB");
    # processed_tensor = transform_pipeline(img);

    img_path, tumor_type = get_random_image_path();
    img = Image.open(img_path).convert("RGB");
    processed_tensor = transform_pipeline(img);

    preprocessor.debug_steps(title="Selected tumor type: " + tumor_type + "\nImage path: " + img_path);

    img = denormalize(processed_tensor).permute(1, 2, 0).numpy().clip(0, 1);
    plt.figure(figsize=(4, 4));
    plt.imshow(img);
    plt.title("Final preprocessed image");
    plt.axis('off');
    plt.show();

if __name__ == "__main__":
    main();