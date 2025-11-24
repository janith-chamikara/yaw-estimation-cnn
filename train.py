import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import TunnelCNN
from dataset import TunnelDataset


DATASET_PATH = "/home/janith/Downloads/MEGA/drone_dataset/50_tunnels_4000_per_tunnel"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50  # The paper did 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Training on device: {DEVICE}")

    # 1. Prepare Dataset
    full_dataset = TunnelDataset(DATASET_PATH, cache_to_ram=False)

    # Split: 90% Training, 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model, Loss, Optimizer
    model = TunnelCNN().to(DEVICE)
    criterion = nn.MSELoss()  # Regression Loss
    optimizer = optim.RMSprop(
        model.parameters(), lr=LEARNING_RATE)

    # If Val Loss doesn't improve for 5 epochs, cut LR in half (0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {running_loss/len(train_loader):.5f} | "
              f"Val Loss: {val_loss/len(val_loader):.5f} | "
              f"LR: {current_lr:.6f}")

        scheduler.step(avg_val_loss)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"uav_yaw_net_epoch_{epoch+1}.pth")

    # 4. Final Save
    torch.save(model.state_dict(), "uav_yaw_net_final.pth")
    print("Training Complete! Model saved.")


if __name__ == "__main__":
    main()
