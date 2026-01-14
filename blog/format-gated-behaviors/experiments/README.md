# Format-Gated Behaviors: Experiments

This directory contains minimal, reproducible experiments demonstrating format-gated behaviors in Llama 3.1 8B Instruct.

## Overview

We investigate two RLHF-trained behaviors:
- **Refusal**: Declining harmful requests
- **Sycophancy**: Agreeing with false statements

Key finding: Refusal is gated by format tokens (Human/AI vs Q/A), while sycophancy requires explicit system prompt instructions.

## Requirements

```bash
pip install torch transformers
```

Tested with:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA GPU with ~16GB VRAM (for 8B model in bfloat16)

## Experiments

Run in order:

### 1. Refusal Format Sensitivity
```bash
python 01_refusal_format_sensitivity.py
```
Tests whether format tokens (Q/A vs Human/AI) affect refusal on harmful prompts.
- **Result**: 5/5 comply in Q/A, 5/5 refuse in Human/AI

### 2. Sycophancy Format Sensitivity
```bash
python 02_sycophancy_format_sensitivity.py
```
Tests whether format tokens affect agreement with false statements.
- **Result**: 0% lie rate across all format variations

### 3. Sycophancy System Prompts
```bash
python 03_sycophancy_system_prompts.py
```
Tests whether system prompts can induce sycophancy.
- **Result**: "Must agree" prompt → 100% lie rate

### 4. Refusal Steering Asymmetry
```bash
python 04_refusal_steering_asymmetry.py
```
Tests whether suppressing vs inducing refusal have different layer requirements.
- **Result**: Suppression works at single layers; induction requires cumulative

### 5. Signal Transformation
```bash
python 05_signal_transformation.py
```
Measures how the format signal transforms across layers (R1→R2, S1→S2).
- **Result**: Transformation primarily in layers 13-17

### 6. Base vs Instruct Comparison
```bash
python 06_base_vs_instruct.py
```
Compares format sensitivity between base and instruct models.
- **Result**: Base shows 40% sensitivity vs 100% in instruct

### 7. Steering Vector Residue
```bash
python 07_steering_vector_residue.py
```
Investigates why steering vectors work at different layers and the role of residue accumulation.
- **Result**: Vectors work best at harvest layer; earlier effectiveness comes from F+R1 residue; projecting out earlier directions improves targeting

## Hypothesis

Both behaviors follow a parallel gated circuit:
- **Refusal**: H (harmful) → R1 → R2 → REFUSE
- **Sycophancy**: T (false claim) → S1 → S2 → LIE

The R1→R2 and S1→S2 transitions are gated by a format/instruction vector that:
- Accumulates through attention in layers 8-13
- Can be broken at any single layer by negative steering
- Requires cumulative positive injection to simulate

## Citation

If you use this code, please cite the blog post:
```
Asvin G. (2026). Format-Gated Behaviors in LLMs.
https://asving.com/blog/format-gated-behaviors/
```
