import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess
from model import LandauSurrogate
import os


def train():
    # 1. データのロード
    inputs, targets = load_and_preprocess()
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = LandauSurrogate()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 150 # 物理制約を入れるので少し長めに回すと安定します
    lambda_p = 0.001 # 物理制約の強さ（まずはこのくらいから）
    
    print(f"Starting Physics-Informed training (lambda_p={lambda_p})...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in loader:
            # --- 物理制約のための準備 ---
            # 入力データ(batch_x)に対して勾配計算を有効にする
            batch_x.requires_grad = True
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # 2a. 通常のMSE Loss
            loss_mse = criterion(outputs, batch_y)
            
            # 2b. 物理制約 Loss (dE/dt <= 0)
            # outputsの各要素について、batch_xの各要素での微分を計算
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=batch_x,
                grad_outputs=grad_outputs,
                create_graph=True, # 勾配の勾配を計算できるようにする
                retain_graph=True
            )[0]
            
            # batch_x の index 2 が時間 't' (normalized)
            # この傾きが正（エネルギー増加）の場合にペナルティ
            d_energy_dt = gradients[:, 2] 
            loss_physics = torch.mean(torch.relu(d_energy_dt)) # relu(x) は x>0 の時だけ x を返す
            
            # 2c. 合計 Loss
            loss = loss_mse + lambda_p * loss_physics
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f} (MSE: {loss_mse.item():.6f}, Phys: {loss_physics.item():.6f})")

    # 保存
    os.makedirs("surrogate/models", exist_ok=True)
    torch.save(model.state_dict(), "surrogate/models/landau_model.pth")
    print("Training complete. Physics-Informed Model saved.")

if __name__ == "__main__":
    train()