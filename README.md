# 1D-PIC Plasma Simulation with AI Surrogate Model

This repository is a fork of [Antoine Tavant's 1D-PIC](https://github.com/antoinelpp/1d-pic-electrostatic), extended with a **Neural Network-based Surrogate Model** to accelerate plasma response predictions.

## 🌟 Major Extension: AI Surrogate Model (v0.1)
I have developed an AI surrogate to predict electric field energy decay (Landau Damping) without running the full PIC simulation.

### Performance Highlights
- **Speedup**: Achieved ~200ms per query (Total), with **pure inference time < 1ms**. This is approximately **600,000x faster** than the original 1D-PIC simulation.
- **Accuracy**: Reached an MSE of **0.055** (log-scale) across various electron temperatures ($T_e$).
- **Capability**: Successfully predicts "extrapolated" physics beyond the training range (e.g., $T_e = 2000$ eV).



## 🛠 Project Roadmap
My goal is to refine this into a "Physics-Informed" model for robust research use:
1. **Data Denoising**: Implement Savitzky-Golay filters to remove numerical spikes from PIC data.
2. **Physics-Informed Constraints**: Add monotonicity loss to ensure physical energy decay.
3. **Advanced Architecture**: Exploring **FNO (Fourier Neural Operator)** for superior spectral denoising.

### Current Challenge (March 12)
Currently testing a **Physics-Informed (PI) MLP** with monotonic energy decay constraints.  
However, we still observe some numerical instabilities and non-physical energy spikes, likely due to overfitting high-frequency numerical noise in the PIC data.  
**Next step:** move to an **FNO (Fourier Neural Operator)**–based surrogate to better filter spectral noise and capture smooth decay behavior.

---

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch (for Surrogate Model)
- Numpy, Numba, Matplotlib

### Recommended: venv setup (Windows / PowerShell)

```powershell
cd C:\Users\takum\Documents\Research\Independent_Research\1D_PIC
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### How to Run
- `main.py`: Original 1D-PIC simulation.
- `surrogate/train.py`: Train the neural network.
- `surrogate/quick_test.py`: Benchmark the AI prediction speed and accuracy.

## Authors & Acknowledgments
- **Surrogate Model**: [Your Name/GitHub Name]
- **Original PIC Code**: [Antoine Tavant](https://github.com/antoinelpp)
- **Special Thanks**: Professor Shinsuke Fujioka (Osaka Univ.) for the surrogate research inspiration.

## License
This project is licensed under the MIT License.
