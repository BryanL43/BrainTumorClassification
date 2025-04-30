import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from src.DenseCNN import DenseCNN

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    );

    # Parameters for Plotting
    model_path = "./model/DenseCNN_Brain_Tumor.pth";
    test_root = "./DataSet/Test";

    # Load the model & its stored history
    model = DenseCNN(num_classes=len(datasets.ImageFolder(root=test_root).classes)).to(device);
    checkpoint = torch.load(model_path, map_location=device);
    model.load_state_dict(checkpoint["model_state_dict"]);
    model.eval();
    history = checkpoint.get("history", None);

    # Denormalize history
    epochs = np.arange(len(history['train_loss']));
    training_losses = history['train_loss'];
    validation_losses = history['val_loss'];
    accuracies = history['val_acc'];
    learning_rates = history['lr_history'];

    # Create graph
    fig, ax1 = plt.subplots(figsize=(12, 6));

    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss', color='tab:blue');
    ax1.plot(epochs, training_losses, color='tab:blue', label='Training Loss');
    ax1.plot(epochs, validation_losses, color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue');

    ax2 = ax1.twinx();
    ax2.set_ylabel('Accuracy', color='tab:green');
    ax2.plot(epochs, accuracies, color='tab:green', label='Accuracy');
    ax2.tick_params(axis='y', labelcolor='tab:green');

    for i, lr in enumerate(learning_rates):
        if i % 5 == 0:
            ax1.annotate(
                f'lr:{lr:.6f}', 
                (epochs[i], training_losses[i]),
                textcoords="offset points", 
                xytext=(0, 25), 
                ha='center', 
                fontsize=8, 
                color='tab:orange'
            );
    
    threshold = 95;
    for i, accuracy in enumerate(accuracies):
        if accuracy > threshold:
            threshold = accuracy
            ax2.annotate(
                f'Acc:{accuracy:.2f}', 
                (epochs[i], accuracies[i]),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8, 
                color='tab:green'
            );


    ax1.legend(loc='upper left');
    ax2.legend(loc='upper right');

    plt.title("Training/Validation Loss, Accuracy, & Learning Rate over Epochs");

    fig.tight_layout();

    plt.show();


if __name__ == "__main__":
    main();