# 1D-PIC Plasma Simulation + WIP Surrogate Model

This repository is a fork of [Antoine Tavant's 1D-PIC](https://github.com/antoinelpp/1d-pic-electrostatic), plus my own experiments with a **neural-network surrogate** for Landau damping.  
Everything is still **work in progress** – numbers and models will likely change.

## Current Status (2026-03-12)

- **PIC side**
  - 1D electrostatic PIC is running and validated against a standard Landau damping test.
  - Electric field energy \( \sum E^2 \) is stored every time step for use as training data.

- **Surrogate side (MLP, WIP)**
  - Input: \([T_e, L_x, t]\), Output: \(\log_{10}(E(t))\).
  - Data is preprocessed with:
    - initial-transient cut (first ~5% in time),
    - log-scale transform,
    - Savitzky–Golay smoothing to reduce numerical spikes.
  - Training uses a **physics-informed loss** that penalizes \( \partial E / \partial t > 0 \) to encourage monotonic decay.

### Known Issues / Limitations
- The current MLP surrogate is **not production‑ready**:
  - still shows non-physical bumps for some parameters,
  - is sensitive to high-frequency numerical noise in PIC outputs,
  - has not been systematically benchmarked (no solid error numbers yet).

### Next Steps
1. Tune the physics-informed loss (weight, schedule) and smoothing parameters.
2. Try **FNO (Fourier Neural Operator)** or other spectral architectures to better filter noise.
3. Add proper evaluation scripts and plots to quantify generalization and failure modes.

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
