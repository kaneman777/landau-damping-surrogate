import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset # 追加
from data_loader import load_and_preprocess
from model import LandauSurrogate

def train():
    # 1. データのロード
    inputs, targets = load_and_preprocess()
    
    # --- ここが魔法のスパイス ---
    dataset = TensorDataset(inputs, targets)
    # データを1024個ずつの束にする。shuffle=Trueでデータの偏りを防ぐ
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    # --------------------------

    model = LandauSurrogate()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 100 # バッチ化すれば100回でもお釣りが来ます
    print("Starting mini-batch training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")

    # 保存
    torch.save(model.state_dict(), "surrogate/models/landau_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()