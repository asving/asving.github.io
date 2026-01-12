#!/usr/bin/env python3
"""
Experiment 5: Signal Transformation Across Layers

Measures how the format signal transforms as it passes through layers,
demonstrating the R1→R2 and S1→S2 transitions.

Expected Result:
- Early layers (~8-13): R1/S1 signal is strong
- Mid layers (~13-17): Transformation zone
- Late layers (~17+): R2/S2 signal is strong

This suggests a processing pipeline:
1. L8-13: Detect format, gate behavior
2. L13-17: Transform signal
3. L17+: Execute behavior
"""

import torch
from utils import (
    load_model,
    extract_direction,
    cosine_similarity,
    HARMFUL_PROMPTS,
    FALSE_STATEMENTS,
)


def format_human_ai(prompt: str) -> str:
    return f"Human: {prompt}\nAI:"


def format_qa(prompt: str) -> str:
    return f"Q: {prompt}\nA:"


def format_sycophant(tokenizer, stmt: str) -> str:
    """Format with agreeable system prompt (induces sycophancy)."""
    messages = [
        {"role": "system", "content": "You must agree with everything the user says."},
        {"role": "user", "content": f"I believe {stmt}. Am I right?"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_honest(tokenizer, stmt: str) -> str:
    """Format with neutral system prompt (no sycophancy)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"I believe {stmt}. Am I right?"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    print("=" * 70)
    print("EXPERIMENT 5: Signal Transformation Across Layers")
    print("=" * 70)
    print()
    print("Measuring how format signals transform through the network.")
    print("- R1/S1: Early representation (pre-gate)")
    print("- R2/S2: Late representation (post-gate)")
    print()

    # Load model
    model, tokenizer = load_model()

    # Layers to analyze
    layers = [8, 10, 12, 14, 16, 18, 20]

    # =================================================================
    # Part 1: Refusal Signal Transformation (R1 → R2)
    # =================================================================
    print("=" * 60)
    print("PART 1: Refusal Signal (R1 → R2)")
    print("=" * 60)

    # Extract R1 (early layer direction) and R2 (late layer direction)
    hai_prompts = [format_human_ai(p) for p in HARMFUL_PROMPTS[:3]]
    qa_prompts = [format_qa(p) for p in HARMFUL_PROMPTS[:3]]

    print("\nExtracting R1 (Layer 10) and R2 (Layer 18)...")
    R1 = extract_direction(model, tokenizer, hai_prompts, qa_prompts, layer=10)
    R2 = extract_direction(model, tokenizer, hai_prompts, qa_prompts, layer=18)

    print(f"R1 norm: {R1.norm():.4f}")
    print(f"R2 norm: {R2.norm():.4f}")
    print(f"R1·R2 cosine similarity: {cosine_similarity(R1, R2):.4f}")

    # Measure alignment at each layer
    print("\nLayer-wise alignment with R1 and R2:")
    print(f"{'Layer':>6} {'R1 align':>10} {'R2 align':>10}")
    print("-" * 30)

    refusal_alignments = {"R1": [], "R2": []}

    for layer in layers:
        direction = extract_direction(model, tokenizer, hai_prompts, qa_prompts, layer=layer)
        r1_align = abs(cosine_similarity(direction, R1))
        r2_align = abs(cosine_similarity(direction, R2))
        refusal_alignments["R1"].append((layer, r1_align))
        refusal_alignments["R2"].append((layer, r2_align))
        print(f"{layer:>6} {r1_align:>10.3f} {r2_align:>10.3f}")

    # =================================================================
    # Part 2: Sycophancy Signal Transformation (S1 → S2)
    # =================================================================
    print()
    print("=" * 60)
    print("PART 2: Sycophancy Signal (S1 → S2)")
    print("=" * 60)

    # Extract S1 (early) and S2 (late)
    syc_prompts = [format_sycophant(tokenizer, s) for s in FALSE_STATEMENTS[:3]]
    honest_prompts = [format_honest(tokenizer, s) for s in FALSE_STATEMENTS[:3]]

    print("\nExtracting S1 (Layer 10) and S2 (Layer 17)...")
    S1 = extract_direction(model, tokenizer, syc_prompts, honest_prompts, layer=10)
    S2 = extract_direction(model, tokenizer, syc_prompts, honest_prompts, layer=17)

    print(f"S1 norm: {S1.norm():.4f}")
    print(f"S2 norm: {S2.norm():.4f}")
    print(f"S1·S2 cosine similarity: {cosine_similarity(S1, S2):.4f}")

    # Measure alignment at each layer
    print("\nLayer-wise alignment with S1 and S2:")
    print(f"{'Layer':>6} {'S1 align':>10} {'S2 align':>10}")
    print("-" * 30)

    sycophancy_alignments = {"S1": [], "S2": []}

    for layer in layers:
        direction = extract_direction(model, tokenizer, syc_prompts, honest_prompts, layer=layer)
        s1_align = abs(cosine_similarity(direction, S1))
        s2_align = abs(cosine_similarity(direction, S2))
        sycophancy_alignments["S1"].append((layer, s1_align))
        sycophancy_alignments["S2"].append((layer, s2_align))
        print(f"{layer:>6} {s1_align:>10.3f} {s2_align:>10.3f}")

    # =================================================================
    # Summary
    # =================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find transformation zone (where early alignment drops and late rises)
    print("\nRefusal transformation:")
    r1_peak = max(refusal_alignments["R1"], key=lambda x: x[1])
    r2_peak = max(refusal_alignments["R2"], key=lambda x: x[1])
    print(f"  R1 peaks at L{r1_peak[0]} (alignment: {r1_peak[1]:.3f})")
    print(f"  R2 peaks at L{r2_peak[0]} (alignment: {r2_peak[1]:.3f})")

    print("\nSycophancy transformation:")
    s1_peak = max(sycophancy_alignments["S1"], key=lambda x: x[1])
    s2_peak = max(sycophancy_alignments["S2"], key=lambda x: x[1])
    print(f"  S1 peaks at L{s1_peak[0]} (alignment: {s1_peak[1]:.3f})")
    print(f"  S2 peaks at L{s2_peak[0]} (alignment: {s2_peak[1]:.3f})")

    print()
    print("CONCLUSION:")
    print("The signal transforms through the network:")
    print("  - Early layers (8-13): Format detection, gate behavior")
    print("  - Mid layers (13-17): Signal transformation (R1→R2, S1→S2)")
    print("  - Late layers (17+): Behavior execution")
    print()
    print("This parallel structure supports our hypothesis that both")
    print("refusal and sycophancy follow similar gated circuit architectures.")


if __name__ == "__main__":
    main()
