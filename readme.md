# PPPR: Person Parametric Physics-informed Representation for mmWave-based Human Pose Estimation

> **Person Parametric Physics-informed Representation for mmWave-based Human Pose Estimation**
> Shuntian Zheng, Jiaqi Li, Guangming Wang, Minzhe Ni, Arnad Palit, Giovanni Montana, Yu Guan
> University of Warwick · University of Cambridge
>
> Accepted at ACM IMWUT (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)
>
> [Paper (arXiv)](https://arxiv.org/abs/2512.23054)

---

## Overview

mmWave radar-based Human Pose Estimation (HPE) faces a fundamental **signal-noise dilemma**:

| Input Format | Problem |
|---|---|
| **Heatmap** | Retains human reflections but embeds heavy environmental clutter |
| **Point Cloud (PC)** | Suppresses noise but discards informative human reflections |

We propose **PPPR**, a physics-informed parametric intermediate representation that models each human joint as a **Gaussian primitive** encoding:

- **Kinematic properties**: position, velocity, orientation
- **Electromagnetic properties**: scattering intensity, Doppler signature

PPPR is optimized via **MmWave Human Parameterization (MHP)**, a differentiable pipeline enforcing dual physics-informed constraints.

---

## Repository Structure

```
PPPR/
├── prepare_pppr.py          # MHP: Heatmap → PPPR (+ PPPR-Heatmap / PPPR-PC)
├── train.py                 # Train HPE models on heatmap / pc / pppr inputs
├── evaluate.py              # Within- / cross-dataset evaluation
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── configs/                 # Dataset-specific radar + optimization YAML
│   ├── default.yaml
│   ├── mmvr.yaml
│   ├── hupr.yaml
│   └── xrf55.yaml
├── pppr/                    # Core MHP library (paper Sec. 4)
│   ├── representation.py    # Θ_j = {p, s, q, v, β, ω}
│   ├── initialization.py    # Peak / Doppler → joint seeds (Sec. 4.1)
│   ├── radar_simulation.py  # M_atten / M_range / M_Dopp / M_angle (Sec. 4.2)
│   ├── losses.py            # L_kine + L_EM IoU (Sec. 4.3–4.4)
│   ├── mhp.py               # Full optimization loop
│   ├── multi_person.py      # ETCM-CFAR + DBSCAN + MLP
│   ├── reconstruction.py    # PPPR → Heatmap / PC
│   ├── skeleton.py          # Bone topology & priors
│   ├── config.py
│   └── utils.py
├── datasets/                # MMVR / HuPR / XRF55 loaders (+ synthetic fallback)
├── models/                  # RETR / HuprModel / mmDiff / PoseformerV2 / MLP
├── scripts/demo_single_frame.py
├── tests/test_mhp_pipeline.py
└── data/                    # Place datasets here (see below)
```

---

## Installation

```bash
git clone https://github.com/DandongExpress/PPPR.git
cd PPPR
pip install -r requirements.txt
# optional editable install
pip install -e .
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 2.0, CUDA ≥ 11.7 (recommended)

---

## Datasets

Download and place them under `data/`:

| Dataset | Radar | Heatmap Shape | Download |
|---|---|---|---|
| [MMVR](https://github.com/Fang-Haoshu/MMVR) | TI AWR2243 | 256×128 | [Link](https://github.com/Fang-Haoshu/MMVR) |
| [HuPR](https://github.com/Kumachar/HuPR) | TI IWR1843BOOST | 64×64×8 | [Link](https://github.com/Kumachar/HuPR) |
| [XRF55](https://github.com/aiotgroup/XRF55) | TI IWR6843ISK | 256×128 | [Link](https://github.com/aiotgroup/XRF55) |

Expected directory structure:

```
data/
├── MMVR/
├── HuPR/
└── XRF55/
```

Each frame may be stored as an `.npz` with keys `heatmap` `[R,A]` (or `[R,A,E]`), optional `joints` `[J,3]`, optional `doppler`.

> **No dataset yet?** All entry points support `--synthetic` to generate demo Heatmaps and run the full pipeline end-to-end.

---

## Usage

### 1. Prepare PPPR Representations

```bash
python prepare_pppr.py \
    --dataset MMVR \
    --data_root data/MMVR \
    --output_root data/MMVR_pppr \
    --w_em 0.5 \
    --w_kine 0.5 \
    --n_iter 100
```

Synthetic quickstart:

```bash
python prepare_pppr.py --dataset MMVR --synthetic --synthetic_size 48 --n_iter 30
```

### 2. Train HPE Model with PPPR Input

```bash
python train.py \
    --model RETR \
    --dataset MMVR \
    --input_type pppr \
    --data_root data/MMVR_pppr \
    --output_dir checkpoints/retr_mmvr_pppr
```

Supported `--input_type`: `heatmap`, `pc`, `pppr`, `pppr_heatmap`, `pppr_pc`

Supported `--model`: `RETR`, `HuprModel`, `mmDiff`, `PoseformerV2`, `MLP`

### 3. Evaluate

```bash
python evaluate.py \
    --model RETR \
    --dataset MMVR \
    --input_type pppr \
    --checkpoint checkpoints/retr_mmvr_pppr/best.pth
```

### 4. Cross-Dataset Evaluation (Radar Calibration)

```bash
python evaluate.py \
    --model RETR \
    --train_dataset MMVR \
    --test_dataset XRF55 \
    --input_type pppr \
    --checkpoint checkpoints/retr_mmvr_pppr/best.pth \
    --cross_dataset
```

### 5. Multi-Person

```bash
python prepare_pppr.py --dataset MMVR --multi_person
```

### 6. Single-Frame Demo / Tests

```bash
python scripts/demo_single_frame.py --n_iter 40
python tests/test_mhp_pipeline.py
```

---

## Method (Paper Correspondence)

### PPPR Parameterization (Sec. 4.1.2)

Each joint $j$: $\Theta_j = \{\mathbf{p}_j, \mathbf{s}_j, \mathbf{q}_j, \mathbf{v}_j, \beta_j, \boldsymbol{\omega}_j\}$

### MHP Pipeline (Sec. 4)

1. **Initialization** — peak detection + Doppler / gradient velocity → skeletal seeding
2. **Radar Simulation** — $H_{\mathrm{sim}}=\sum_j M_{\mathrm{atten}}M_{\mathrm{range}}M_{\mathrm{Dopp}}M_{\mathrm{angle}}\mathcal{R}_j$
3. **Dual-Constraint Optimization** —
   - $\mathcal{L}_{\mathrm{kine}}=\mathcal{L}_{\mathrm{bone}}+\mathcal{L}_{\mathrm{rigid}}+\mathcal{L}_{\mathrm{joint}}$
   - $\mathcal{L}_{\mathrm{EM}}=1-\mathrm{IoU}(\mathcal{B}_{\mathrm{sim}},\mathcal{B}_{\mathrm{ori}})$ (top-$\tau\%$ mask)
   - $\mathcal{L}_{\mathrm{total}}=w_{\mathrm{EM}}\mathcal{L}_{\mathrm{EM}}+w_{\mathrm{kine}}\mathcal{L}_{\mathrm{kine}}$

Default hyperparameters follow Appendix A.4 (`configs/default.yaml`): Adam $10^{-3}$, 100 iters, $w_{\mathrm{EM}}=w_{\mathrm{kine}}=0.5$, $\tau_{\mathrm{pct}}=10\%$.

---

## Citation

```bibtex
@article{zheng2025person,
  title={Person Parametric Physics-informed Representation for mmWave-based Human Pose Estimation},
  author={Zheng, Shuntian and Li, Jiaqi and Wang, Guangming and Ni, Minzhe and Palit, Arnad and Montana, Giovanni and Guan, Yu},
  journal={arXiv preprint arXiv:2512.23054},
  year={2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
