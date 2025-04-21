from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

from Preprocessor import Preprocessor
from RepeatDataSet import RepeatDataSet
from DenseCNN import DenseCNN

def train_model(model, dataloader, optimizer, scheduler, device, epochs=10):
    criterion = torch.nn.CrossEntropyLoss();
    model.train();

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0;

        for images, labels in dataloader:
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

        # Prompt scheduler at the end of each epoch
        scheduler.step(epoch);

        acc = 100.0 * correct / total;
        current_lr = optimizer.param_groups[0]['lr'];
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%, LR: {current_lr:.6f}");

    # TO DO: VALIDATION LOOP & EARLY STOPPING

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
    root = "./DataSet/Training";
    model_path = "./model/DenseCNN_Brain_Tumor.pth";
    batch_size = 64;
    num_workers = 12;
    scheduler_T_0 = 3;
    scheduler_T_mult = 1;
    learning_rate = 0.0001;
    num_epochs = 10; # 20 for full training

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
    augmented_train_dataset = RepeatDataSet(base_train_dataset, 21);

    print("Base dataset size:", len(base_train_dataset));
    print("Augmented dataset size:", len(augmented_train_dataset));

    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    train_model(model, train_loader, optimizer, scheduler, device, epochs=num_epochs);

    # Save trained model
    torch.save(model.state_dict(), model_path);
    print(f"Model saved to {model_path}");


if __name__ == "__main__":
    main();
