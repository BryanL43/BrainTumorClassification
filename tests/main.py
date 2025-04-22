from PIL import Image
import torch
from torchvision import transforms, datasets
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Preprocessor import Preprocessor
from src.DenseCNN import DenseCNN

def load_model(model_path, num_classes, device):
    model = DenseCNN(num_classes=num_classes);
    checkpoint = torch.load(model_path, map_location=device);
    model.load_state_dict(checkpoint['model_state_dict']);
    model.to(device);
    model.eval();
    return model;

def evaluate_on_test_set(model, test_loader, device):
    correct = 0;
    total = 0;

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device);
            outputs = model(images);
            probs = F.softmax(outputs, dim=1);
            preds = torch.argmax(probs, dim=1);
            correct += (preds == labels).sum().item();
            total += labels.size(0);

    accuracy = 100 * correct / total;
    print(f"\nFinal Test Accuracy: {accuracy:.2f}% ({correct}/{total})\n");

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    );

    # Same transform as training
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        Preprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]);

    # Load test set using ImageFolder
    test_root = "./DataSet/Testing";
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform_pipeline);
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False);

    class_names = test_dataset.classes;
    model = load_model("./model/partial_model_test.pth", num_classes=len(class_names), device=device);

    evaluate_on_test_set(model, test_loader, device);


if __name__ == "__main__":
    main();
