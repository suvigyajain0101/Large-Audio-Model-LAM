# Large-Audio-Model (LAM) Experimental Research Log

This document tracks hypotheses, configurations, and empirical findings across our audio tokenization and multimodal alignment experiments.

---

## Core Research Question
When mapping `16 kHz` audio through a `25 Hz` (`40 ms` frame stride) Conformer encoder into a language model:
1. How do different discrete quantization bottlenecks (**VQ-STE**, **BEST-RQ**, **FSQ**, **RVQ**) compare in **codebook utilization (perplexity)** and **reconstruction/probing fidelity**?
2. What paralinguistic (speaker emotion, pitch contour) and musical (instrument timbre, genre) information is lost in **Discrete Hard Tokens** compared to **Continuous Soft Embeddings (`bfloat16`)**, and at what bitrate (codebooks $M$) does discrete quantization close the gap?

---

## Planned Ablation Matrix

| Representation Type | Frame Rate | Bitrate (bits/sec) | Codebook Perplexity (%) | Speech Probe (WER / Phoneme Acc) | Emotion Probe (RAVDESS Acc %) | Music Probe (GTZAN / MusicCaps Acc %) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Continuous Soft Tokens (`bfloat16`, $D=512$)** | `25 Hz` (`40 ms`) | $204.8\text{ kbps}$ | N/A (Continuous) | *TBD* | *TBD* | *TBD* |
| **Standard VQ-STE ($K=1024$, 1 codebook)** | `25 Hz` (`40 ms`) | $250\text{ bps}$ | *TBD* | *TBD* | *TBD* | *TBD* |
| **BEST-RQ Random Projection ($K=1024$)** | `25 Hz` (`40 ms`) | $250\text{ bps}$ | *TBD* | *TBD* | *TBD* | *TBD* |
| **Finite Scalar Quantization (FSQ $[8, 5, 5, 5]$)** | `25 Hz` (`40 ms`) | $250\text{ bps}$ | *TBD* | *TBD* | *TBD* | *TBD* |
| **Residual VQ (RVQ, $M=2, 4, 8$ codebooks)** | `25 Hz` (`40 ms`) | $0.5\text{–}2.0\text{ kbps}$ | *TBD* | *TBD* | *TBD* | *TBD* |

---

## Experiment Log

### Exp 01: Pure-JAX Differentiable STFT & 128-Bin Log-Mel Spectrogram
* **Status**: In Progress (`notebooks/02_audio_data_and_jax_dsp.ipynb`)
* **Goal**: Verify numerical equivalence against reference SciPy/Librosa STFT and benchmark `@jax.jit` + `jax.vmap` batch throughput on GPU.
