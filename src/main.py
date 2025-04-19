from PIL import Image, ImageFilter
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from Preprocessor import Preprocessor

# Denormalize the image for final output [TEMP]
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1);
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1);
    return tensor * std + mean;

def main():
    # Instantiate Preprocessor with specified parameters
    laplacian_kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0,
                1, -4, 1,
                0, 1, 0],
        scale=None,
        offset=0
    );
    preprocessor = Preprocessor(
        clip_limit=1.0,
        gauss_std_radius=1,
        laplacian_kernel=laplacian_kernel
    );

    # Image processing pipeline
    transform_pipeline = transforms.Compose([
        preprocessor,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ]);

    img_path = "./DataSet/Training/meningioma_tumor/m (5).jpg";
    img = Image.open(img_path).convert("RGB");
    processed_tensor = transform_pipeline(img);

    preprocessor.debug_steps();

    img = denormalize(processed_tensor).permute(1, 2, 0).numpy().clip(0, 1);
    plt.figure(figsize=(4, 4));
    plt.imshow(img);
    plt.title("Final preprocessed image");
    plt.axis('off');
    plt.show();

if __name__ == "__main__":
    main();