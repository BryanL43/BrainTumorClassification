import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys

from src.train.Preprocessor import Preprocessor
from src.train.Model import DenseCNN

import torch
import os
from torch.utils.data import DataLoader

def train_model(
    model: any, 
    data_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: any, 
    scheduler: any, 
    device: torch.device, 
    model_path: str, 
    history_path: str,
    epochs: int = 10, 
):
    """
        Train the model for a given number of epochs. 
        Save the best model as checkpoints and training history. 

        Parameters
        ----------
        model : any
            The model to be trained
        data_loader : DataLoader
            The training data loader object
        val_loader : DataLoader
            The validation data loader object
        optimizer : any
            The optimizer object, i.e. AdamW
        scheduler : any
            The scheduler object, i.e. CosineAnnealingWarmRestarts
        device : torch.device
            The device to be used for training
        model_path : str
            The path to save the best model checkpoint
        history_path : str
            The path to save the training history
        epochs : int, optional
            The number of epochs to train the model, by default 10
    """
    criterion = torch.nn.CrossEntropyLoss();
    best_val_acc = 0.0;
    best_epoch = -1;
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr_history': []
    };

    for epoch in range(epochs):
        model.train();
        total_loss, correct, total = 0, 0, 0;

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device);

            # Forward pass
            outputs = model(images);
            loss = criterion(outputs, labels);

            # Backpropagation
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            # Collect statistics
            total_loss += loss.item();
            _, predicted = torch.max(outputs, 1);
            correct += (predicted == labels).sum().item();
            total += labels.size(0);

            # Free memory per batch
            del images, labels, outputs, loss, predicted;
            torch.cuda.empty_cache();

        # Validation
        model.eval();
        val_loss, val_correct, val_total = 0, 0, 0;

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device);

                outputs = model(images);
                loss = criterion(outputs, labels);
                
                # Track metrics
                val_loss += loss.item();
                _, predicted = torch.max(outputs, 1);
                val_correct += (predicted == labels).sum().item();
                val_total += labels.size(0);

                # Free memory per batch
                del images, labels, outputs, loss, predicted;
                torch.cuda.empty_cache();

        # Calculate Epoch Stats
        total_loss /= len(data_loader);
        train_acc = 100 * correct / total;
        val_loss /= len(val_loader);
        val_acc = 100 * val_correct / val_total;
        
        # Update history
        history['train_loss'].append(total_loss);
        history['train_acc'].append(train_acc);
        history['val_loss'].append(val_loss);
        history['val_acc'].append(val_acc);
        history['lr_history'].append(optimizer.param_groups[0]['lr']);

        # Scheduler step after each epoch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss);
        else:
            scheduler.step();
        
        # Save best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc;
            best_epoch = epoch;
            
            # Ensure save directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True);

            print(f"Saving new best model at epoch {epoch}...");
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc
            }, model_path);

        print(
            f'Epoch: {epoch} | '
            f'Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
            f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
        );
    
    # Final save of full history
    torch.save({
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'history': history
    }, history_path);

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

    # Hyperparameters for training
    training_root = "./DataSet/Training";
    validation_root = "./DataSet/Validation";
    model_path = "./model/DenseCNN_Brain_Tumor.pth";
    history_path = "./model/DenseCNN_Brain_Tumor_history.pth";
    batch_size = 64;
    num_workers = 12;
    scheduler_T_0 = 3;
    scheduler_T_mult = 2;
    learning_rate = 0.0001;
    num_epochs = 20; # 20 for full training

    # Training image processing pipeline
    train_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        Preprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225] # ImageNet std
        )
    ]);

    # Load and augment the training dataset 21 times to minimize overfitting
    train_dataset = datasets.ImageFolder(root=training_root, transform=train_transform_pipeline);
    print(train_dataset.class_to_idx); # ImageFolder internally maps labels already

    print("Base dataset size:", len(train_dataset));

    # Validation image processing pipeline
    val_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        Preprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225] # ImageNet std
        )
    ]);

    # Load the validation dataset
    val_dataset = datasets.ImageFolder(root=validation_root, transform=val_transform_pipeline);
    print("Validation dataset size:", len(val_dataset));
    
    # Load datasets into DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    );

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    );
    
    model = DenseCNN(num_classes=4).to(device);
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01);

    # CosineAnnealingWarmRestarts for model to explore more
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_T_0, # First restart happens after 3 epochs
        T_mult=scheduler_T_mult, # Restart occurs after each T_0 * T_mult epochs
        eta_min=1e-6 # Min learning rate
    );

    train_model(model, train_loader, val_loader, optimizer, scheduler, device, model_path, history_path, epochs=num_epochs);
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache();


if __name__ == "__main__":
    main();