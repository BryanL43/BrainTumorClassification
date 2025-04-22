from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

from Preprocessor import Preprocessor
from RepeatDataSet import RepeatDataSet
from DenseCNN import DenseCNN

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
    epochs: int = 10, 
    patience: int = 3
):
    criterion = torch.nn.CrossEntropyLoss();
    best_val_acc = 0;
    patience_counter = 0;

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

                # Foward pass
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

        # Scheduler step after each epoch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss);
        else:
            scheduler.step();

        # Early stopping and checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc;
            patience_counter = 0;

            # Ensure save directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True);

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc
            }, model_path);
        else:
            patience_counter += 1;
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}");
                break;

        print(f'Epoch: {epoch+1} | '
              f'Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}');

    # Load best model
    checkpoint = torch.load(model_path);
    model.load_state_dict(checkpoint['model_state_dict']);
    return model;

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
    validation_root = "./DataSet/Testing";
    # model_path = "./model/DenseCNN_Brain_Tumor.pth";
    model_path = "./model/partial_model_test.pth";
    batch_size = 64;
    num_workers = 12;
    scheduler_T_0 = 3;
    scheduler_T_mult = 1;
    learning_rate = 0.0001;
    num_epochs = 10; # 20 for full training
    num_patience = 3;

    # Training image processing pipeline
    train_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15), # random rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1), # width & height shift (10%)
            scale=(0.7, 1.3) # zoom range (70% to 130%)
        ),
        Preprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225] # ImageNet std
        )
    ]);

    # Load and augment the training dataset 21 times to minimize overfitting
    base_train_dataset = datasets.ImageFolder(root=training_root, transform=train_transform_pipeline);
    print(base_train_dataset.class_to_idx); # ImageFolder internally maps labels already
    augmented_train_dataset = RepeatDataSet(base_train_dataset, 21);

    print("Base dataset size:", len(base_train_dataset));
    print("Augmented dataset size:", len(augmented_train_dataset));

    # ================ DEBUG: Visualize Augmented Images ================
    # Pick a random base index from the original dataset
    base_idx = random.randint(0, len(base_train_dataset) - 1);

    plt.figure(figsize=(20, 4));
    for i in range(21):
        # Calculate the index within the repeated dataset
        repeated_idx = base_idx + i * len(base_train_dataset);
        
        # Fetch the transformed image
        image_tensor, label = augmented_train_dataset[repeated_idx];

        # Denormalize the tensor image
        image = image_tensor.clone().detach();
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1);
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1);
        image = image.permute(1, 2, 0).clamp(0, 1).numpy();

        plt.subplot(2, 11, i + 1);
        plt.imshow(image);
        plt.axis("off");
        plt.title(f"Aug {i+1}");

    idx_to_class = {v: k for k, v in base_train_dataset.class_to_idx.items()};
    label = idx_to_class[label];
    plt.suptitle(f"21 Augmented Views of Image #{base_idx} (Class: {label})");
    plt.tight_layout();
    plt.show();
    # ================ DEBUG: Visualize Augmented Images ================

    # Validation (Testing) image processing pipeline
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
        dataset=augmented_train_dataset,
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

    train_model(model, train_loader, val_loader, optimizer, scheduler, device, model_path, epochs=num_epochs, patience=num_patience);
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache();


if __name__ == "__main__":
    main();
