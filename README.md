# Large-Audio-Model (LAM): Audio Tokenization & Multimodal Alignment in JAX

[![Open Notebook 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suvigyajain0101/Large-Audio-Model-LAM/blob/main/notebooks/01_jax_foundations.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX / Flax NNX](https://img.shields.io/badge/Framework-JAX%20%7C%20Flax%20NNX-orange.svg)](https://github.com/google/jax)

An open-source research repository and educational journey building **Large Audio Models (LAMs)** from first principles in **pure JAX and Flax NNX**.

Unlike cascaded voice pipelines ($\text{Audio} \to \text{ASR Transcript} \to \text{Text LLM}$) that discard paralinguistic cues, speaker emotion, prosody, and musical structure, **natively multimodal Large Audio Models** tokenize raw audio waveforms directly into discrete codebook IDs or continuous soft embeddings that share a unified representation space with text tokens.

---

## Research Thesis & Core Ablation Study

When building a native Large Audio Model, a central architectural question arises at the boundary between the **Audio Encoder** (e.g., a Conformer operating at `25 Hz` / `40 ms` frames) and the **Backbone Language Model**:

> **How much linguistic, paralinguistic (emotion, prosody, speaker state), and musical information is lost when quantizing continuous audio representations ($z_t \in \mathbb{R}^D$) into discrete tokens (`VQ`, `BEST-RQ`, `FSQ`, `RVQ`) versus feeding continuous soft embeddings (`bfloat16`) directly into the LLM?**

```
16 kHz Mono Audio (16,000 samples/sec)
  │
  ▼
Differentiable Log-Mel Spectrogram (25ms window, 10ms hop, 128 Mel bins -> 100 frames/sec)
  │
  ▼
Conformer Audio Encoder (4x temporal subsampling -> 25 frames/sec, 1 vector every 40ms)
  │
  ├────► [Path A: Continuous Soft Tokens] ──► bfloat16[T, D] ──► Linear/MLP Adapter ──► LLM Embedding Space
  │
  └────► [Path B: Discrete Hard Tokens]
            ├── 1. Vector Quantization (VQ-STE / EMA)
            ├── 2. Random-Projection Quantizer (BEST-RQ)
            ├── 3. Finite Scalar Quantization (FSQ)
            └── 4. Residual Vector Quantization (RVQ)
                   └──► int32[T] (+ audio_vocab_offset) ──► Unified LLM Embedding Lookup Table
```

This repository systematically implements, benchmarks, and ablates each component from scratch in JAX.

---

## Repository Structure & Learning Roadmap

```text
Large-Audio-Model-LAM/
├── README.md                                  # Project overview, architecture & research roadmap
├── requirements.txt                           # Pinned dependencies for Colab & local execution
├── notebooks/
│   ├── 01_jax_foundations.ipynb               # Stage 1: Functional JAX, @jit, vmap, grad, lax.scan & Flax NNX
│   ├── 02_audio_data_and_jax_dsp.ipynb        # Stage 2: 16kHz Nyquist, pure-JAX STFT, 128-bin Mel Filterbank & SpecAugment
│   ├── 03_audio_tokenizers_vq_fsq_rvq.ipynb   # Stage 3: Conformer Encoder & Quantizers (VQ-STE, BEST-RQ, FSQ, RVQ)
│   └── 04_lam_soft_vs_hard_ablation.ipynb     # Stage 4: Soft vs. Hard Token LLM Probing (ASR, Emotion, Music QA)
├── lam/                                       # Reusable pure-JAX library modules graduated from notebooks
│   ├── __init__.py
│   ├── dsp.py                                 # Differentiable STFT, Mel-Spectrogram & SpecAugment
│   ├── conformer.py                           # Flax NNX Conformer blocks & 4x Conv subsampling
│   └── quantizers.py                          # VQ-STE, BEST-RQ, FSQ & RVQ implementations
├── assets/                                    # Spectrogram visualizations & ablation plots
└── docs/
    └── research_log.md                        # Running experimental log & ablation tables
```

### Interactive Notebooks

| Notebook | Topic | Key Concepts Covered | Colab Link |
| :--- | :--- | :--- | :--- |
| **`01_jax_foundations.ipynb`** | **JAX Mastery for Audio & LLMs** | Immutability (`.at[].set()`), Pure Functions, PRNG Splitting, `@jax.jit`, `jax.vmap`, `jax.value_and_grad`, `jax.lax.scan` for sequential audio frames, Flax NNX | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suvigyajain0101/Large-Audio-Model-LAM/blob/main/notebooks/01_jax_foundations.ipynb) |
| **`02_audio_data_and_jax_dsp.ipynb`** | **Audio Signal Processing in Pure JAX** | Digital audio basics, $16\text{ kHz}$ Nyquist bandwidth ($0\text{–}8\text{ kHz}$), framing (`25 ms` window / `10 ms` stride), Hann windowing, FFT, 128-bin Mel filterbank, vectorized SpecAugment | *Coming Next* |
| **`03_audio_tokenizers_vq_fsq_rvq.ipynb`** | **Neural Audio Tokenizers (`25 Hz`)** | Conformer $4\times$ temporal downsampling (`40 ms` frames), Straight-Through Estimator (STE), Codebook Collapse analysis, `BEST-RQ`, `FSQ`, and `RVQ` | *Planned* |
| **`04_lam_soft_vs_hard_ablation.ipynb`** | **Soft vs. Hard Audio Tokens in LLMs** | Vocabulary expansion (`audio_vocab_offset`) vs. Soft-token MLP projection; evaluation across Speech (LibriSpeech), Paralinguistics (RAVDESS/IEMOCAP), and Music (MusicCaps/GTZAN) | *Planned* |

---

## Key Academic References

All implementations in this repository are clean-room JAX reproductions grounded in public peer-reviewed literature:

1. **Conformer**: Gulati et al., *"Conformer: Convolution-augmented Transformer for Speech Recognition"*, Interspeech 2020. ([arXiv:2005.08100](https://arxiv.org/abs/2005.08100))
2. **BEST-RQ**: Chiu et al., *"Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition"*, ICML 2022. ([arXiv:2202.01855](https://arxiv.org/abs/2202.01855))
3. **USM (Universal Speech Model)**: Zhang et al., *"Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages"*, 2023. ([arXiv:2303.01037](https://arxiv.org/abs/2303.01037))
4. **SoundStream & Residual VQ**: Zeghidour et al., *"SoundStream: An End-to-End Neural Audio Codec"*, IEEE/ACM TASLP 2021. ([arXiv:2107.03312](https://arxiv.org/abs/2107.03312))
5. **AudioLM & AudioPaLM**: Borsos et al., *"AudioLM: a Language Modeling Approach to Audio Generation"*, 2022 ([arXiv:2209.03143](https://arxiv.org/abs/2209.03143)); Rubenstein et al., *"AudioPaLM: A Large Language Model That Can Speak and Listen"*, 2023 ([arXiv:2306.12925](https://arxiv.org/abs/2306.12925)).
6. **Finite Scalar Quantization (FSQ)**: Mentzer et al., *"Finite Scalar Quantization: VQ-VAE Made Simple"*, ICLR 2024. ([arXiv:2309.15505](https://arxiv.org/abs/2309.15505))
7. **SpeechTokenizer & Mimi (Moshi)**: Zhang et al., *"SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models"*, ICLR 2024 ([arXiv:2308.16692](https://arxiv.org/abs/2308.16692)); Défossez et al., *"Moshi: a speech-text foundation model for real-time dialogue"*, Kyutai 2024 ([arXiv:2410.00037](https://arxiv.org/abs/2410.00037)).

---

## Quickstart

```bash
git clone https://github.com/suvigyajain0101/Large-Audio-Model-LAM.git
cd Large-Audio-Model-LAM
pip install -r requirements.txt
```
