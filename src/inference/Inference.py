import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.train.Model import DenseCNN

class Inference:
    def __init__(
        self, 
        model: DenseCNN, 
        model_path: str,
        history_path: str,
        test_loader: DataLoader,
        device: torch.device
    ):
        # Load the best model checkpoint in evaluate mode
        self.model = model.to(device);
        model_checkpoint = torch.load(model_path, map_location=device);
        self.model.load_state_dict(model_checkpoint["model_state_dict"]);
        self.model.eval();

        # Load the training history
        history_checkpoint = torch.load(history_path, map_location=device);
        self.history = history_checkpoint["history"];

        self.test_loader = test_loader;
        self.device = device;

    def evaluate(self, test_loader: DataLoader, criterion: any) -> tuple[list[float], list[float]]:
        """
            Evaluate the model on the test (unseen) set.
        
            Parameters
            ----------
            test_loader : DataLoader
                The test data loader object
            criterion : any
                Loss function

            Returns
            -------
            tuple[list[float], list[float]]
                y_true, y_pred
        """
        test_loss, correct, total = 0, 0, 0;
        y_true, y_pred = [], [];

        # No gradients are needed during evaluation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device);

                # Forward pass
                outputs = self.model(images);
                loss = criterion(outputs, labels);
                test_loss += loss.item() * labels.size(0);

                # Predictions and accuracy calculation
                _, preds = torch.max(outputs, 1);
                y_pred.extend(preds.cpu().numpy());
                y_true.extend(labels.cpu().numpy());

                correct += (preds == labels).sum().item();
                total += labels.size(0);
        
        avg_test_loss = test_loss / total;
        avg_test_acc = correct / total;

        print("Evaluation results on test set:");
        print(f"Loss: {avg_test_loss:.4f}");
        print(f"Accuracy: {avg_test_acc:.4f}");

        return y_true, y_pred;

    def generate_confusion_matrix(self, class_names: list[str], y_true: list[float], y_pred: list[float]) -> None:
        """
            Generate confusion matrix and classification report.
        
            Parameters
            ----------
            class_names : list[str]
                List of class names
            y_true : list[float]
                True labels
            y_pred : list[float]
                Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred);

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names, 
            yticklabels=class_names
        );
        plt.title('Confusion Matrix');
        plt.xlabel('Predicted');
        plt.ylabel('True');
        plt.tight_layout();
        plt.show();

        # Classification report
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)

    def create_loss_curves(self):
        """ Plot train and validation loss curves. """
        plt.figure();
        plt.plot(self.history['train_loss'], label='Train Loss');
        plt.plot(self.history['val_loss'], label='Val Loss');
        plt.legend();
        plt.title('Loss Curves');
        plt.xlabel('Epochs');
        plt.ylabel('Loss');
        plt.show();

