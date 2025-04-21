# TO DO: TURN THIS SHAMBLE OF A MESS INTO INFERENCE PIPELINE

from PIL import Image
import torch
from torchvision import transforms
import os
import random

from src.Preprocessor import Preprocessor
from src.DenseCNN import DenseCNN

def get_random_image_path(root_dir=".\DataSet\Testing"):
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
    return os.path.join(type_path, selected_image);

def load_model(model_path, num_classes, device):
    model = DenseCNN(num_classes=num_classes);
    model.load_state_dict(torch.load(model_path, map_location=device));
    model.to(device);
    model.eval();
    return model;

def validate_single_image(model, image_path, transform_pipeline, class_names, device):
    image = Image.open(image_path).convert("RGB");
    input_tensor = transform_pipeline(image).unsqueeze(0).to(device);

    with torch.no_grad():
        output = model(input_tensor);
        predicted_class = torch.argmax(output, dim=1).item();
        probability = torch.max(output).item();

    print(f"\n--- Quick Validation ---");
    print(f"Image: {image_path}");
    print(f"Predicted: {class_names[predicted_class]} (Confidence: {probability*100:.2f}%)\n");

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

    # Same transform as training
    preprocessor = Preprocessor();
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        preprocessor,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]);

    # Set up classes manually
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'];

    # Load and test
    model = load_model("./model/DenseCNN_brain_tumor.pth", num_classes=4, device=device);

    for i in range(10):
        test_image_path = get_random_image_path();
        validate_single_image(model, test_image_path, transform_pipeline, class_names, device);

if __name__ == "__main__":
    main();
