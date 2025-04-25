from PIL import Image
import torch
from torchvision import transforms, datasets
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.Preprocessor import Preprocessor
from src.DenseCNN import DenseCNN

def load_model(model_path, num_classes, device):
    model = DenseCNN(num_classes=num_classes);
    checkpoint = torch.load(model_path, map_location=device);
    model.load_state_dict(checkpoint['model_state_dict']);
    model.to(device);
    model.eval();
    history = checkpoint.get('history', None)  
    return model, history;
def evaluate_model(model, test_loader, loss_fn, history, class_names=None):
    device = next(model.parameters()).device  # Get device from model
    
    # Set model to evaluation mode
    model.eval()

    # Initialize variables
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    # No gradients are needed during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            # Predictions and accuracy calculation
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Average test loss and accuracy
    avg_test_loss = test_loss / total
    test_accuracy = correct / total

    # Print test results
    print("\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    if class_names:
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        # Classification report
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)
    else:
        cm = None
        report = None

    if history:
        # Plot train and validation loss curves
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    return avg_test_loss, test_accuracy, cm, report

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
    test_root = "./DataSet/Test";
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform_pipeline);
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False);

    class_names = test_dataset.classes;
    model, history = load_model("./model/DenseCNN_Brain_Tumor.pth", num_classes=len(class_names), device=device);

    evaluate_on_test_set(model, test_loader, device);
    loss_fn = torch.nn.CrossEntropyLoss()
    evaluate_model(model, test_loader, loss_fn, history=history, class_names=class_names)


if __name__ == "__main__":
    main();