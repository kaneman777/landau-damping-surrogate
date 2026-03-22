import numpy as np
import torch
import glob
import os
from scipy.signal import savgol_filter

def load_and_preprocess_fno(data_dir="sweep_2d_results/data"):
    all_params = []   # Inputs: [Te, Lx]
    all_curves = []   # Targets: [Log_E_0, Log_E_1, ..., Log_E_999]
    
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        files = glob.glob(os.path.join("../", data_dir, "*.npy"))
        
    if not files:
        print("Error: data files not found.")
        return

    print(f"Loading {len(files)} simulations for FNO...")

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        te, lx, t, energy = data['te'], data['lx'], data['t'], data['energy']
        
        # 1. Cut initial transient (same as before)
        start_idx = int(len(t) * 0.05)
        e_s = energy[start_idx:]

        # 2. Log-scale transform
        log_e = np.log10(np.maximum(e_s, 1e-25))

        # 3. Smoothing (Savitzky-Golay) 
        # FNO is a filter itself, but keeping this helps remove extreme PIC spikes
        if len(log_e) > 51:
            log_e = savgol_filter(log_e, window_length=51, polyorder=3)

        # --- IMPORTANT CHANGE FOR FNO ---
        # Instead of looping through time steps, we store the WHOLE curve
        # We also truncate/pad to a fixed length (e.g., 1000 steps) for batching
        target_length = 1000 
        if len(log_e) >= target_length:
            log_e = log_e[:target_length] # Truncate if too long
            all_params.append([te, lx])
            all_curves.append(log_e)
        # --------------------------------

    # Convert to Tensors
    # X shape: [Num_Simulations, 2]
    # Y shape: [Num_Simulations, 1000]
    X = torch.tensor(all_params, dtype=torch.float32)
    Y = torch.tensor(all_curves, dtype=torch.float32)

    # Normalization (Te and Lx only)
    in_min = X.min(dim=0)[0]
    in_max = X.max(dim=0)[0]
    X_norm = (X - in_min) / (in_max - in_min)

    # Save normalization params for the meeting/future use
    os.makedirs("surrogate/models", exist_ok=True)
    torch.save({'min': in_min, 'max': in_max}, "surrogate/models/fno_norm_params.pth")

    print(f"FNO Pre-processing complete.")
    print(f"Input shape (Params): {X_norm.shape}") # e.g., [3000, 2]
    print(f"Output shape (Curves): {Y.shape}")     # e.g., [3000, 1000]
    
    return X_norm, Y

if __name__ == "__main__":
    load_and_preprocess_fno()