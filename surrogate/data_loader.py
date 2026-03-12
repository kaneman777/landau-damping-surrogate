import numpy as np
import torch
import glob
import os
from scipy.signal import savgol_filter

def load_and_preprocess(data_dir="sweep_2d_results/data"):
    all_inputs = []
    all_targets = []
    
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        # 階層が違う場合を考慮して一つ上も探す
        files = glob.glob(os.path.join("../", data_dir, "*.npy"))
        
    if not files:
        print("Error: data files not found. Check your directory.")
        return

    print(f"Loading {len(files)} files...")

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        te, lx, t, energy = data['te'], data['lx'], data['t'], data['energy']
        
        # 初期5%をカット
        start_idx = int(len(t) * 0.05)
        t_s, e_s = t[start_idx:], energy[start_idx:]

        #1 対数スケール (Energy -> Log10)
        log_e = np.log10(np.maximum(e_s, 1e-25))

        # 2. 【ここに追加！】クレンジング（平滑化）
        # window_length: 51（500点に対して約1割の窓幅）
        # polyorder: 3（3次多項式で近似）
        if len(log_e) > 51: # データ点数が窓幅より多い場合のみ実行
            log_e = savgol_filter(log_e, window_length=51, polyorder=3)

        for i in range(len(t_s)):
            all_inputs.append([te, lx, t_s[i]])
            all_targets.append([log_e[i]])

    inputs = torch.tensor(all_inputs, dtype=torch.float32)
    targets = torch.tensor(all_targets, dtype=torch.float32)

    # 正規化用のパラメータを計算・保存
    in_min = inputs.min(dim=0)[0]
    in_max = inputs.max(dim=0)[0]
    inputs_norm = (inputs - in_min) / (in_max - in_min)

    # 後で使うために保存
    os.makedirs("surrogate/models", exist_ok=True)
    torch.save({'min': in_min, 'max': in_max}, "surrogate/models/norm_params.pth")

    print(f"Pre-processing complete. Samples: {len(inputs_norm)}")
    return inputs_norm, targets

if __name__ == "__main__":
    load_and_preprocess()