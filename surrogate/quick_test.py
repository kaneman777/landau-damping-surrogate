# surrogate/quick_test.py として保存して実行
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LandauSurrogate
import os
import time

def quick_check(te, lx):
    # 1. モデルの準備
    model = LandauSurrogate()
    model.load_state_dict(torch.load("surrogate/models/landau_model.pth"))
    model.eval()
    
    # 2. 正規化パラメータのロード
    norm = torch.load("surrogate/models/norm_params.pth")
    in_min, in_max = norm['min'], norm['max']
    
    # 3. 予測
    t_eval = np.linspace(0, 1e-7, 500)
    inputs = torch.tensor([[te, lx, t] for t in t_eval], dtype=torch.float32)
    inputs_norm = (inputs - in_min) / (in_max - in_min)
    
    # --- 追加：計測開始 ---
    # 500点一気に予測するのにかかる時間を計測
    start_time = time.perf_counter() # より高精度なタイマー
    with torch.no_grad():
        log_e_pred = model(inputs_norm).flatten().numpy() # flatten()を追加して1次元に
    
    # 4. (オプション) もしPICの元データがあれば重ねる
    data_path = f"sweep_2d_results/data/data_Lx{lx}_Te{te}.npy"
    plt.figure(figsize=(10, 6))
    
    end_time = time.perf_counter()
    runtime_ms = (end_time - start_time) * 1000  # ms単位に変換


    if os.path.exists(data_path):
        pic_data = np.load(data_path, allow_pickle=True).item()
        plt.semilogy(pic_data['t'], pic_data['energy'], 'b--', alpha=0.5, label='Original PIC')
    
    plt.semilogy(t_eval, 10**log_e_pred, 'r-', linewidth=2, label=f'AI Surrogate ({te}eV)')
   
   # --- 変更：タイトルにRuntimeを表示 ---
    plt.title(f"AI Prediction (Te={te}eV, Lx={lx}cm)\nRuntime: {runtime_ms:.4f} ms")
    # ----------------------------------
    plt.xlabel("Time [s]")
    plt.ylabel("Field Energy")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    return runtime_ms

if __name__ == "__main__":
   # 100から2000まで、200刻みで一気にプロットしてみる
    temperatures = [100, 500, 1000, 1500, 2000]
    runtimes = []
    for t in temperatures:
        rt = quick_check(t, 0.01)
        runtimes.append(rt)

    # 全体の平均を出力
    print("\n" + "="*30)
    print(f"Average Runtime: {np.mean(runtimes):.4f} ms")
    print("="*30)