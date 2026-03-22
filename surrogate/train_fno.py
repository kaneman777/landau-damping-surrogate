import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from data_loader import load_and_preprocess_fno
from fno_model import FNO1d


def train_fno(
    val_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 200,
    plot_val_samples: int = 5,
):
    # 1. Load data: X [N, 2], Y [N, T]
    inputs, targets = load_and_preprocess_fno()
    if inputs is None or targets is None:
        raise RuntimeError("load_and_preprocess_fno() returned no data.")

    n = inputs.shape[0]
    if n < 2:
        raise RuntimeError(f"Need at least 2 simulations for train/val split, got {n}.")

    dataset = TensorDataset(inputs, targets)

    # Train / val split (~80/20 by default; e.g. 32 -> 25 train, 7 val)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    if n_train < 1:
        n_val = n - 1
        n_train = 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    batch_train = max(1, min(32, n_train))
    batch_val = max(1, min(32, n_val))

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_val, shuffle=False)

    print(
        f"Train/val split: {n_train} train, {n_val} val "
        f"(fraction={val_fraction}, seed={seed})"
    )

    # 2. Model
    model = FNO1d(modes=8, width=64, out_steps=1000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    os.makedirs("surrogate/models", exist_ok=True)
    model_path = "surrogate/models/fno_landau_model.pth"

    print("Starting FNO training with validation...")

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                total_val += criterion(outputs, batch_y).item()
        avg_val = total_val / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"train MSE: {avg_train:.6f} | val MSE: {avg_val:.6f} "
                f"(best val: {best_val:.6f})"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Best val MSE: {best_val:.6f}. Model saved to {model_path}")

    # 3. Plot several validation curves (model never trained on these)
    model.eval()
    n_plot = min(plot_val_samples, len(val_ds))
    if n_plot == 0:
        return

    cols = min(3, n_plot)
    rows = (n_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    with torch.no_grad():
        for i in range(n_plot):
            x, y_true = val_ds[i]
            pred = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
            y_np = y_true.cpu().numpy()
            ax = axes[i]
            ax.plot(y_np, label="Target (preprocessed)", alpha=0.7)
            ax.plot(pred, label="FNO pred", linewidth=2)
            ax.set_title(f"Val sample {i + 1}/{n_plot}")
            ax.legend(fontsize=8)
            ax.set_xlabel("index")
            ax.set_ylabel("log10(E)")

    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("FNO: validation set (held-out simulations)", fontsize=12)
    fig.tight_layout()
    out_png = "fno_val_samples.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {n_plot} validation curves to {out_png}")


if __name__ == "__main__":
    train_fno()
