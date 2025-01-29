import os
import torch
import matplotlib.pyplot as plt
from data import DataSetWrapper
from model import SimCLR
from train import Trainer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs, batch_size, T, proj_dim = 175, 128, 0.07, 512
    num_workers, valid_size, strength = 3, 0.2, 1.0

    dataset = DataSetWrapper(batch_size, num_workers, valid_size, (448, 448, 3), strength)
    train_loader, valid_loader = dataset.get_data_loaders()

    model = SimCLR(out_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))

    trainer = Trainer(model, optimizer, scheduler, device, T)
    train_losses, val_losses = [], []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = sum(trainer.train_step(xi, xj) for (xi, xj), _ in train_loader) / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = sum(trainer.valid_step(xi, xj) for (xi, xj), _ in valid_loader) / len(valid_loader)
            val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/simclr_best.pth")

        scheduler.step()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
