# 🚀 CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation

---

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](TODO) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()

**CompilerKV** is a **prefill-only**, **risk-adaptive** and **head-aware** KV cache compression framework for long-context LLM inference. It **compiles offline experience into reusable decision tables**, producing **stable one-shot pruning** under tight budgets and mitigating long-context **tail failures**. :contentReference[oaicite:0]{index=0}

> 💡 **Key Insight**: Under strict budgets, compression risk varies **prompt-by-prompt** (e.g., entropy/perplexity regimes), and attention heads exhibit strong **functional heterogeneity** (retrieval vs. noisy heads). Ignoring either factor destabilizes token selection and triggers tail failures. :contentReference[oaicite:1]{index=1}

---

## 🔍 Method Overview

### Why CompilerKV?
Most KV compression methods in **prefill-only** settings rely on:
- **Prompt-agnostic** policies (one-size-fits-all thresholds / patterns), which over-prune high-risk prompts.
- **Weakly-conditioned**, noisy attention signals, where small estimation errors cause irreversible eviction under one-shot pruning. 

CompilerKV instead shifts to a **compile-then-execute** paradigm: compile robust priors offline, then execute O(1) lookups online during prefill. 

---

## 🧠 How It Works (3-Stage Prefill Pipeline)

CompilerKV consists of three integrated stages (executed **once** at the end of prefill; **no online learning during decode**). 

### Stage 1 — Stabilized Token Utility Estimation
Compute a **noise-resilient base utility** to get stable rankings under tight budgets:
- Window-cumulative attention (durable relevance) + sample-wise normalized value norm (scale-invariant density). 

### Stage 2 — Head-Aware Importance Injection (Static Table)
Handle head heterogeneity via an **offline-compiled Head Heterogeneity Table**:
- Compile `W_head[l, h]` offline, then apply **weighted max-pooling** so reliable heads can “veto” noisy ones during selection. 

### Stage 3 — Risk-Adaptive Threshold Gating (Static LUT)
Adapt pruning aggressiveness using **prompt risk signals**:
- Risk signals: **attention entropy** (structural risk) + **local perplexity** (semantic risk). 
- Query a **Risk-Gating LUT** to select a threshold `τ(l)`; then enforce the per-layer budget with **Top-B_l correction**. 

✅ **Plug-and-play**: prefill-only one-shot; compressed KV stays static during decoding. 

---

## 🧱 Offline Compilation (Decision Tables)

CompilerKV compiles two artifacts on a held-out **calibration corpus** `D_cal` (strictly disjoint from LongBench to avoid leakage): 

- **Head Heterogeneity Table**: `W_head[l, h]` (offline RL / contextual bandit style compilation). 
- **Risk-Adaptive Threshold LUT**: `T_gate` / `M_lex(l, b_h, b_p)` for threshold gating. 

**Calibration corpus**: ~50K long-context prompts from diverse sources (PG19, arXiv, PubMed, GovReport, QMSum, BookSum, ShareGPT/UltraChat, The Pile subsets, etc.). 

---

## 📊 Results

### LongBench (KV Cache Budget = 512 tokens / layer)
CompilerKV dominates SOTA under a strict 512-token budget, **recovering 97.7% of FullKV** and achieving **up to +5.2 points** gain over the strongest competitor. 
 

### 📊 Model Comparison（LongBench, KV Cache = 512）

| Model | FullKV | StreamingLLM | H2O | SnapKV | PyramidKV | DynamicKV | **CompilerKV (Ours)** |
|-------|--------|--------------|-----|--------|-----------|----------------------|
| Llama-3-8B-Instruct | 41.95 | 34.70 | 37.20 | 40.30 | 40.18 | 40.73 | **40.98** |
| Mistral-7B-Instruct-v0.2 | 42.71 | 30.06 | 37.37 | 40.71 | 40.47 | 40.90 | **41.21** |
| Qwen2-7B-Instruct | 40.71 | 29.65 | 35.63 | 38.47 | 38.19 | 39.16 | **39.50** |
| InternLM-2.5-7B-Chat-1M | 43.21 | 32.25 | 34.65 | 37.84 | 37.86 | 38.39 | **39.18** |

(Also noted: on LLaMA-3-8B, CompilerKV recovers **97.7%** of FullKV: 40.98 vs. 41.95.) 

### Needle-in-a-Haystack Pressure Test (Mistral-7B)
CompilerKV maintains a retrieval pattern comparable to FullKV and achieves **Avg. Score = 0.86**, closing the gap to the oracle (FullKV **0.92**) and outperforming baselines under extreme lengths.
| Method | Accuracy |
|--------|----------|
| FullKV | 92% |
| StreamingLLM | 26% |
| PyramidKV | 72% |
| DynamicKV | 83% |
| **CompilerKV** | **86%** |
---

## ⚡ Quick Start

### Install
```bash
git clone TODO_YOUR_REPO_URL
cd CompilerKV
pip install -r requirements.txt
```
---

## 📚 Citation

If you find DynamicKV useful, please cite our paper:

```bibtex

```

> 🔗 **Code**: []()  
> 📄 **Paper**: []()
