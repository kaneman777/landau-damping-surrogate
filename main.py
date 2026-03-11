import numpy as np
import matplotlib.pyplot as plt
import os
from pic.constantes import (me, q, kb, eps_0)
from simulation import run_simulation

def save_comparison_plot(result, te_val, lx_val, gamma_theory, fname):
    """2次元スイープ用の比較プロット保存"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(result['t_energy'], result['energy_history'], label='Field Energy (PIC)', color='blue', alpha=0.7)
    
    t_theory = result['t_energy']
    E_theory = result['energy_history'][0] * np.exp(-2 * gamma_theory * t_theory)
    plt.semilogy(t_theory, E_theory, 'r--', label=f'Theory (gamma={gamma_theory:.2e})')
    
    # 物理指標の計算
    n_m3 = result['params']['n'] * 1e6
    Lx_m = lx_val * 1e-2
    w_pe = np.sqrt(n_m3 * q**2 / (me * eps_0))
    v_th = np.sqrt(kb * (te_val * 11604.5) / me)
    k_val = 2 * np.pi / Lx_m
    kld = k_val * (v_th / w_pe)
    
    plt.title(f"Sweep: Te={te_val}eV, Lx={lx_val}cm (k*Ld={kld:.3f})")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy (log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='lower left')
    
    # パラメータ情報を刻印
    info = f"Te: {te_val}eV\nLx: {lx_val}cm\nk*Ld: {kld:.3f}\ngamma: {gamma_theory:.2e}"
    plt.gca().text(0.95, 0.95, info, transform=plt.gca().transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    # フォルダ自動作成
    os.makedirs("sweep_2d_results/data", exist_ok=True)
    os.makedirs("sweep_2d_results/plots", exist_ok=True)

    # 2次元パラメータリスト
    te_list = [100, 300, 500, 700, 900, 1100, 1300, 1500]
    lx_list = [0.005, 0.01, 0.015, 0.02] # システムサイズ（波長）を振る

    total = len(te_list) * len(lx_list)
    count = 0

    print(f"Starting 2D sweep: {total} cases. Good night!")

    for lx in lx_list:
        for te in te_list:
            count += 1
            tag = f"Lx{lx}_Te{te}"
            print(f"[{count}/{total}] Running: {tag} ...")
            
            params = {
                "Lx": lx,
                "dX": 1e-4,
                "n": 1e16,
                "dT": 1e-12,
                "Te_0": te,
                "Ti_0": 10,
                "Npart_factor": 100,
                "n_average": 5000,
                "sim_time": 1e-7,
                "verbose": False,
            }

            # 実行
            result = run_simulation(params, use_restart=False)

            # 理論値ガンマ計算
            n_m3 = params['n'] * 1e6
            Lx_m = params['Lx'] * 1e-2
            w_pe = np.sqrt(n_m3 * q**2 / (me * eps_0))
            v_th = np.sqrt(kb * (te * 11604.5) / me)
            lambda_d = v_th / w_pe
            k_val = 2 * np.pi / Lx_m
            kld = k_val * lambda_d
            # 線形ランダウ減衰の公式
            gamma_theory = np.sqrt(np.pi/8) * (w_pe / kld**3) * np.exp(-1/(2*kld**2) - 1.5)

            # データ保存 (AI学習用メタデータ込み)
            np.save(f"sweep_2d_results/data/data_{tag}.npy", {
                't': result['t_energy'],
                'energy': result['energy_history'],
                'te': te,
                'lx': lx,
                'kld': kld,
                'gamma': gamma_theory
            })
            
            # プロット保存
            save_comparison_plot(result, te, lx, gamma_theory, f"sweep_2d_results/plots/plot_{tag}.png")

    print("\n[SUCCESS] 2D Sweep completed. Check 'sweep_2d_results/'")