# もし evaluate.py をまだ作っていなければ、これを使ってください
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LandauSurrogate

def quick_check(te, lx):
    model = LandauSurrogate()
    model.load_state_dict(torch.load("surrogate/models/landau_model.pth"))
    model.eval()
    
    # 正規化パラメータのロード
    norm = torch.load("surrogate/models/norm_params.pth")
    in_min, in_max = norm['min'], norm['max']
    
    # 予測用の時間軸
    t_eval = np.linspace(0, 1e-7, 500)
    inputs = torch.tensor([[te, lx, t] for t in t_eval], dtype=torch.float32)
    inputs_norm = (inputs - in_min) / (in_max - in_min)
    
    with torch.no_grad():
        log_e_pred = model(inputs_norm).numpy()
    
    plt.semilogy(t_eval, 10**log_e_pred, 'r-', label=f'AI: {te}eV')
    plt.title(f"AI Prediction at Te={te}eV, Lx={lx}cm")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

# 学習が終わったら、これをつぶやいてください
# quick_check(te=500, lx=0.01)