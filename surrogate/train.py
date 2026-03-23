import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess
from model import LandauSurrogate
import os


def train():
    # 1. Load data
    inputs, targets = load_and_preprocess()
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = LandauSurrogate()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 150  # Slightly longer run helps stability with the physics penalty
    lambda_p = 0.001  # Strength of physics penalty (reasonable starting value)
    
    print(f"Starting Physics-Informed training (lambda_p={lambda_p})...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in loader:
            # --- Enable gradients w.r.t. inputs for physics penalty ---
            batch_x.requires_grad = True

            optimizer.zero_grad()
            outputs = model(batch_x)

            # 2a. Standard MSE loss
            loss_mse = criterion(outputs, batch_y)

            # 2b. Physics loss: penalize dE/dt > 0 (energy should not grow with time)
            # Derivative of each output element w.r.t. batch_x
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=batch_x,
                grad_outputs=grad_outputs,
                create_graph=True,  # Needed for second derivatives through the loss
                retain_graph=True,
            )[0]

            # batch_x[:, 2] is normalized time t
            # Penalize positive slope (unphysical energy increase)
            d_energy_dt = gradients[:, 2]
            loss_physics = torch.mean(torch.relu(d_energy_dt))  # relu keeps only dE/dt > 0

            # 2c. Total loss
            loss = loss_mse + lambda_p * loss_physics
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f} (MSE: {loss_mse.item():.6f}, Phys: {loss_physics.item():.6f})")

    # Save checkpoint
    os.makedirs("surrogate/models", exist_ok=True)
    torch.save(model.state_dict(), "surrogate/models/landau_model.pth")
    print("Training complete. Physics-Informed Model saved.")

if __name__ == "__main__":
    train()