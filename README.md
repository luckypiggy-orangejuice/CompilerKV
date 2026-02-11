# 🚀 CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation

---

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](TODO) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

**CompilerKV** is a **prefill-only**, **risk-adaptive**, and **head-aware** KV cache compression framework for long-context LLM inference. It **compiles offline experience into reusable decision tables**, enabling **stable one-shot pruning** under tight budgets and substantially reducing long-context **tail failures**.

> 💡 **Key Insight**: Under strict memory budgets, compression risk varies **prompt-by-prompt** (e.g., different entropy/perplexity regimes), and attention heads exhibit strong **functional heterogeneity** (e.g., retrieval-focused vs. noisy heads). Ignoring either factor destabilizes token selection and increases tail failure rates.

---

## 🔍 Method Overview

### Why CompilerKV?
Most prefill-only KV compression methods rely on:
- **Prompt-agnostic policies** (fixed thresholds or fixed patterns), which can over-prune **high-risk** prompts.
- **Noisy attention signals**, where small estimation errors lead to irreversible eviction under one-shot pruning.

CompilerKV adopts a **compile-then-execute** paradigm: we **compile robust priors offline** into compact tables, then perform **O(1) table lookups** during prefill for fast, reliable compression.

---

## 🧠 How It Works (3-Stage Prefill Pipeline)

CompilerKV applies a three-stage pipeline **once at the end of prefill** (no online learning during decode). The compressed cache stays **static** throughout decoding.

### Stage 1 — Stabilized Token Utility Estimation
Compute a **noise-resilient base utility** to obtain stable rankings under tight budgets:
- Window-cumulative attention (durable relevance) + sample-wise normalized value norm (scale-invariant density).

### Stage 2 — Head-Aware Importance Injection (Static Table)
Model head heterogeneity using an **offline-compiled Head Heterogeneity Table**:
- Compile `W_head[l, h]` offline, then apply **weighted max-pooling** so reliable heads can “veto” noisy ones during selection.

### Stage 3 — Risk-Adaptive Threshold Gating (Static LUT)
Adapt pruning aggressiveness using **prompt risk signals**:
- Risk signals: **attention entropy** (structural risk) + **local perplexity** (semantic risk).
- Query a **Risk-Gating LUT** to select threshold `τ(l)`, then enforce the per-layer budget using **Top-`B_l` correction**.

✅ **Plug-and-play**: prefill-only one-shot pruning; compatible with standard attention implementations and long-context inference stacks.

---

## 🧱 Offline Compilation (Decision Tables)

CompilerKV compiles two artifacts on a held-out **calibration corpus** `D_cal` (strictly disjoint from LongBench to avoid leakage):

- **Head Heterogeneity Table**: `W_head[l, h]` (compiled via offline decision-making / bandit-style optimization).
- **Risk-Adaptive Threshold LUT**: `T_gate` / `M_lex(l, b_h, b_p)` for risk-aware threshold gating.

**Calibration corpus**: ~50K long-context prompts from diverse sources (e.g., PG19, arXiv, PubMed, GovReport, QMSum, BookSum, ShareGPT/UltraChat, subsets of The Pile, etc.).

---

## 📊 Experimental Results

### LongBench (KV Cache Budget = 512 tokens / layer)
Under a strict 512-token per-layer budget, CompilerKV achieves **97.7% of FullKV performance** on LLaMA-3-8B and improves by **up to +5.2 points** over the strongest baseline in our setting.

> Some results are included in this repo; you can reproduce all numbers by running `eval.py`.

### 📊 Model Comparison (LongBench, KV Cache = 512)

| Model | FullKV | StreamingLLM | H2O | SnapKV | PyramidKV | DynamicKV | **CompilerKV (Ours)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3-8B-Instruct | 41.95 | 34.70 | 37.20 | 40.30 | 40.18 | 40.73 | **40.98** |
| Mistral-7B-Instruct-v0.2 | 42.71 | 30.06 | 37.37 | 40.71 | 40.47 | 40.90 | **41.21** |
| Qwen2-7B-Instruct | 40.71 | 29.65 | 35.63 | 38.47 | 38.19 | 39.16 | **39.50** |
| InternLM-2.5-7B-Chat-1M | 43.21 | 32.25 | 34.65 | 37.84 | 37.86 | 38.39 | **39.18** |

(Example: on LLaMA-3-8B, CompilerKV recovers **97.7%** of FullKV: 40.98 vs. 41.95.)

### Needle-in-a-Haystack (32K context, 64 cache; Mistral-7B)
CompilerKV preserves retrieval behavior close to FullKV and achieves **0.86** average accuracy, closing much of the gap to the oracle (FullKV **0.92**) under extreme compression.

| Method | Accuracy |
|---|---:|
| FullKV | 92% |
| StreamingLLM | 26% |
| PyramidKV | 72% |
| DynamicKV | 83% |
| **CompilerKV** | **86%** |

---

## ⚡ Quick Start

### Install
```bash
git clone https://github.com/luckypiggy-orangejuice/CompilerKV.git
cd CompilerKV
pip install -r requirements.txt
# or minimally:
# pip install "transformers>=4.44.1"
```
### Run and Evaluate (example)
``` bash
bash run_compilerkv_full.sh
python eval.py --results_dir ../../results/Mistral-7B-Instruct-v0.2_compilerkv_512_64_7_avgpool --method compilerkv
```

## 📚 Citation

If you find CompilerKV useful, please cite our paper:

```bibtex
@misc{compilerkv2026,
  title         = {CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation},
  author        = {Yang, Ning and Wang, Chengzhi and Liu, Yibo and Tian, Baoliang and Zhang, Haijun},
  year          = {2026},
  eprint        = {2602.08686},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.08686}
}

```

> 🔗 **Code**: [https://github.com/luckypiggy-orangejuice/CompilerKV](https://github.com/luckypiggy-orangejuice/CompilerKV)
> 📄 **Paper**: [https://arxiv.org/abs/2602.08686](https://arxiv.org/abs/2602.08686)
