from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

import sys

from Preprocessor import Preprocessor
from RepeatDataSet import RepeatDataSet
from DenseCNN import DenseCNN


def train_model(model, dataloader, val_loader, optimizer, scheduler, device, epochs=10, patience=3):
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()  # For mixed precision
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0;

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device);
            
            # Mixed Precision Forward Pass
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation with scaling
            optimizer.zero_grad();
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Collect statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate Epoch Stats
        total_loss /= len(dataloader)
        train_acc = 100 * correct / total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning Rate Scheduling 
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # For ReduceLROnPlateau
        else:
            scheduler.step()  # For other schedulers
        
       # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        print(f'Epoch: {epoch+1} | '
              f'Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    );
    torch.cuda.empty_cache() # Added
    
    # Additional for local setup
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, 0);

    print("Using device:", device);

    # Hyperparameters for training
    root = "./DataSet/Training";
    model_path = "./model/DenseCNN_Brain_Tumor.pth";
    batch_size = 16; # Lowered from 64 
    num_workers = 2; # Also lowered(12)
    scheduler_T_0 = 3;
    scheduler_T_mult = 1;
    learning_rate = 0.0001;
    num_epochs = 10; # 20 for full training
    validation_split = 0.2; # 20% for validation

    transform_pipeline = transforms.Compose([
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

    # Load and augment the dataset 21 times to minimize overfitting
    base_train_dataset = datasets.ImageFolder(root=root, transform=transform_pipeline);
    print(base_train_dataset.class_to_idx); # ImageFolder internally maps labels already
    
    train_size = int((1 - validation_split) * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_dataset, val_dataset = random_split(base_train_dataset, [train_size, val_size])
    
    augmented_train_dataset = RepeatDataSet(train_dataset, 1);

    print("Base dataset size:", len(base_train_dataset));
    print(f"Training set size: {len(augmented_train_dataset)}");
    print("Augmented dataset size:", len(augmented_train_dataset));
    
    # Display Samples
    plt.figure(figsize=(15, 3))
    for i in range(5):
        # Get a random sample
        img, label = base_train_dataset[np.random.randint(0, len(base_train_dataset))]
        
        # Convert tensor back to PIL for display
        img = transforms.ToPILImage()(img)
        
        # Show image
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f'Class: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Show some augmented samples
    print("\nShowing 5 random augmented samples:")
    
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True 
    );
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    
    model = DenseCNN(num_classes=4).to(device);
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01);

    # CosineAnnealingWarmRestarts for model to explore more
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_T_0, # First restart happens after 3 epochs
        T_mult=scheduler_T_mult, # Restart occurs after each T_0 * T_mult epochs
        eta_min=1e-6 # Min learning rate
    );

    train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=num_epochs, patience=3);

    # Save trained model
    best_checkpoint = torch.load('best_model.pth')
    torch.save(best_checkpoint['model_state_dict'], model_path)
    print(f"Best model saved to {model_path} (Val Acc: {best_checkpoint['acc']:.2f}%)")
    
    os.remove('best_model.pth')


if __name__ == "__main__":
    main();
