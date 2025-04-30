import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.train.Preprocessor import Preprocessor
from src.train.Model import DenseCNN
from src.inference.Inference import Inference

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    );

    # Additional for local setup
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, 0);

    print("Using device:", device);

    # Parameters for Inference
    model_path = "./model/DenseCNN_Brain_Tumor.pth";
    test_root = "./DataSet/Test";
    batch_size = 64;
    num_workers = 12;

    # Same transform as training transform pipeline
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
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform_pipeline);

    # Load dataset into DataLoader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    );

    # Inference pipeline on test dataset
    inference = Inference(
        DenseCNN(num_classes=len(test_dataset.classes)), 
        model_path,
        test_loader,
        device
    );

    criterion = torch.nn.CrossEntropyLoss();
    y_true, y_pred = inference.evaluate(test_loader, criterion);

    inference.generate_confusion_matrix(test_dataset.classes, y_true, y_pred);
    inference.create_loss_curves();

    if torch.cuda.is_available():
        torch.cuda.empty_cache();


if __name__ == "__main__": 
    main();

