# OPD Papers — Multimodal On-Policy Distillation

A curated collection of recent papers on **On-Policy Distillation (OPD)** for large language and multimodal models, with a focus on multimodal / video / vision-language settings.

PDFs are committed locally under [`pdfs/`](pdfs/) for offline reading; original sources are linked below.

---

## 📚 Papers

### 1. Lightning OPD — *NVIDIA*

> **Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation**
> Yecheng Wu, Song Han, Han Cai · NVIDIA · 2026

Collects rollouts and teacher log-probabilities **offline**, freeing all GPUs for student training. Identifies *teacher consistency* (same teacher for SFT and OPD) as the key condition for offline OPD to match standard OPD. Achieves **4.0× higher training efficiency** and reaches 69.9% on AIME 2024 in 30 GPU-hours starting from Qwen3-8B.

- arXiv: <https://arxiv.org/abs/2604.13010>
- PDF: <https://arxiv.org/pdf/2604.13010>
- HF paper page: <https://huggingface.co/papers/2604.13010>
- Local: [`pdfs/01_Lightning-OPD_2604.13010.pdf`](pdfs/01_Lightning-OPD_2604.13010.pdf)

---

### 2. VOLD — *Reasoning Transfer LLM → VLM*

> **VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation**
> Walid Bousselham, Hilde Kuehne, Cordelia Schmid · 2025–2026

Transfers reasoning capability from **text-only LLM teachers** to **VLM students** by combining GRPO with on-policy distillation. Shows that a *cold-start alignment* is essential — without sufficient teacher/student distributional overlap, on-policy distillation fails to provide useful guidance. Evaluated on MMMU-Pro, MathVision, MathVista, LogicVista.

- arXiv: <https://arxiv.org/abs/2510.23497>
- PDF: <https://arxiv.org/pdf/2510.23497>
- OpenReview: <https://openreview.net/forum?id=lkv7sOGtfk>
- Local: [`pdfs/02_VOLD_2510.23497.pdf`](pdfs/02_VOLD_2510.23497.pdf)

---

### 3. Uni-OPD — *Unified LLM + MLLM Recipe*

> **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe**
> 2026

A unified OPD framework that generalizes across **LLMs and MLLMs** via a dual-perspective optimization strategy. From the *student side*: data-balancing strategies promote exploration of informative student-generated states. From the *teacher side*: an outcome-guided margin calibration restores order-consistency between correct and incorrect trajectories. Tested across 5 domains and 16 benchmarks covering single/multi-teacher, strong-to-weak, and cross-modal distillation.

- arXiv: <https://arxiv.org/abs/2605.03677>
- PDF: <https://arxiv.org/pdf/2605.03677>
- HTML: <https://arxiv.org/html/2605.03677v1>
- Local: [`pdfs/03_Uni-OPD_2605.03677.pdf`](pdfs/03_Uni-OPD_2605.03677.pdf)

---

### 4. Video-OPD — *Temporal Video Grounding*

> **Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation**
> 2026

Applies OPD to **Temporal Video Grounding (TVG)** in MLLMs. Optimizes trajectories sampled from the current policy (preserving train/inference alignment) while a frontier teacher supplies token-level supervision via reverse-KL. Introduces *Teacher-Validated Disagreement Focusing (TVDF)*, a lightweight curriculum prioritizing informative trajectories. Outperforms GRPO with substantially faster convergence and lower cost.

- arXiv: <https://arxiv.org/abs/2602.02994>
- PDF: <https://arxiv.org/pdf/2602.02994>
- HTML: <https://arxiv.org/html/2602.02994>
- Local: [`pdfs/04_Video-OPD_2602.02994.pdf`](pdfs/04_Video-OPD_2602.02994.pdf)

---

### 5. AnyFlow — *NVIDIA · Any-Step Video Diffusion*

> **AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation**
> Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, Mike Zheng Shou
> NVIDIA · Show Lab, NUS · MIT · 2026

The first **any-step video-diffusion distillation framework based on flow maps**. Optimizes the full ODE sampling trajectory rather than only a fixed number of steps. Introduces *Flow Map Backward Simulation* — decomposes a full Euler rollout into shortcut flow-map transitions for efficient on-policy distillation, reducing discretization error in few-step sampling and exposure bias in causal generation. Scales 1.3B → 14B across bidirectional and causal architectures.

- arXiv: <https://arxiv.org/abs/2605.13724>
- PDF: <https://arxiv.org/pdf/2605.13724>
- Project page: <https://nvlabs.github.io/AnyFlow/>
- Code: <https://github.com/NVlabs/AnyFlow>
- Local: [`pdfs/05_AnyFlow_2605.13724.pdf`](pdfs/05_AnyFlow_2605.13724.pdf)

---

## 🧭 Reading Order

| Order | Paper | Why |
|------|------|-----|
| 1 | Lightning OPD | Foundational — offline OPD mechanics, teacher consistency |
| 2 | Uni-OPD | Generalization — dual-perspective recipe across LLM/MLLM |
| 3 | VOLD | Cross-modal transfer — LLM → VLM reasoning |
| 4 | Video-OPD | Domain application — temporal video grounding |
| 5 | AnyFlow | Diffusion-flavored — flow-map distillation for video diffusion |

## 🗂 Repo layout

```
opd-papers/
├── README.md
└── pdfs/
    ├── 01_Lightning-OPD_2604.13010.pdf
    ├── 02_VOLD_2510.23497.pdf
    ├── 03_Uni-OPD_2605.03677.pdf
    ├── 04_Video-OPD_2602.02994.pdf
    └── 05_AnyFlow_2605.13724.pdf
```

## 📝 Notes

- All PDFs are downloaded from [arXiv](https://arxiv.org/) under the authors' original licenses; this repo only re-hosts them for personal study. Refer to each paper's arXiv page for citation and license details.
- Last updated: 2026-05-15.
