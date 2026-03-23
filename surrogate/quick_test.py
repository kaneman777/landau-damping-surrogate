# Run from repo root: python surrogate/quick_test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LandauSurrogate
import os
import time

def quick_check(te, lx):
    # 1. Load model
    model = LandauSurrogate()
    model.load_state_dict(torch.load("surrogate/models/landau_model.pth"))
    model.eval()
    
    # 2. Load normalization parameters
    norm = torch.load("surrogate/models/norm_params.pth")
    in_min, in_max = norm['min'], norm['max']
    
    # 3. Predict
    t_eval = np.linspace(0, 1e-7, 500)
    inputs = torch.tensor([[te, lx, t] for t in t_eval], dtype=torch.float32)
    inputs_norm = (inputs - in_min) / (in_max - in_min)
    
    # --- Time batch inference (500 points) ---
    start_time = time.perf_counter()  # High-resolution timer
    with torch.no_grad():
        log_e_pred = model(inputs_norm).flatten().numpy()  # 1D array of predictions

    # 4. (Optional) Overlay original PIC data if present
    data_path = f"sweep_2d_results/data/data_Lx{lx}_Te{te}.npy"
    plt.figure(figsize=(10, 6))
    
    end_time = time.perf_counter()
    runtime_ms = (end_time - start_time) * 1000  # Convert to ms


    if os.path.exists(data_path):
        pic_data = np.load(data_path, allow_pickle=True).item()
        plt.semilogy(pic_data['t'], pic_data['energy'], 'b--', alpha=0.5, label='Original PIC')
    
    plt.semilogy(t_eval, 10**log_e_pred, 'r-', linewidth=2, label=f'AI Surrogate ({te}eV)')
   
    plt.title(f"AI Prediction (Te={te}eV, Lx={lx}cm)\nRuntime: {runtime_ms:.4f} ms")
    # ----------------------------------
    plt.xlabel("Time [s]")
    plt.ylabel("Field Energy")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    return runtime_ms

if __name__ == "__main__":
    # Example sweep over electron temperatures (eV)
    temperatures = [100, 500, 1000, 1500, 2000]
    runtimes = []
    for t in temperatures:
        rt = quick_check(t, 0.01)
        runtimes.append(rt)

    # Print mean runtime across temperatures
    print("\n" + "="*30)
    print(f"Average Runtime: {np.mean(runtimes):.4f} ms")
    print("="*30)