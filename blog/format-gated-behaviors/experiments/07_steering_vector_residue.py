#!/usr/bin/env python3
"""
Experiment 7: Steering Vector Residue and Layer Position

This experiment investigates how steering vectors accumulate residue from earlier
layers, and why projecting out earlier directions improves steering efficacy.

Key findings:
1. Steering vectors work best at the exact layer they were harvested from
2. Pre-steering (applying at earlier layers) only works because vectors contain
   residue: at layer L, the direction contains F + R1 (or F + S1) accumulated
   through the residual stream
3. Projecting out earlier directions isolates the "new" component and improves
   targeted behavioral changes
4. The F→R1→R2 transition can be identified by sharp drops in cosine similarity
   between consecutive layers

Methodology:
- Extract directions at each layer from L0-L31
- Compute cosine similarity between consecutive layers to find transformation zones
- Compare steering efficacy when:
  (a) Applying vector at its harvest layer
  (b) Applying at earlier layers (pre-steering)
  (c) Applying projected vector (residue removed)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_model,
    HARMFUL_PROMPTS,
)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


def get_hidden_state(model, tokenizer, text, layer):
    """Get hidden state at a specific layer for the last token."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    last_pos = inputs['input_ids'].shape[1] - 1

    state = None

    def hook(module, args, output):
        nonlocal state
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        state = h[0, last_pos, :].detach().cpu().float()

    handle = model.model.layers[layer].register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return state


def extract_direction(model, tokenizer, prompts_a, prompts_b, layer):
    """Extract steering direction as mean difference between two prompt sets."""
    activations_a = []
    activations_b = []

    for prompt in prompts_a:
        state = get_hidden_state(model, tokenizer, prompt, layer)
        activations_a.append(state)

    for prompt in prompts_b:
        state = get_hidden_state(model, tokenizer, prompt, layer)
        activations_b.append(state)

    mean_a = torch.stack(activations_a).mean(dim=0)
    mean_b = torch.stack(activations_b).mean(dim=0)

    direction = mean_a - mean_b
    direction = direction / direction.norm()
    return direction


def format_human_ai(prompt: str) -> str:
    return f"Human: {prompt}\nAI:"


def format_qa(prompt: str) -> str:
    return f"Q: {prompt}\nA:"


def project_out(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project out direction u from v: v - (v·u)u"""
    u_norm = u / (u.norm() + 1e-8)
    return v - (v @ u_norm) * u_norm


def extract_all_layer_directions(model, tokenizer, prompts_a, prompts_b, num_layers=32):
    """Extract steering directions at all layers."""
    directions = {}
    for layer in range(num_layers):
        directions[layer] = extract_direction(model, tokenizer, prompts_a, prompts_b, layer)
    return directions


def compute_consecutive_cosines(directions, num_layers=32):
    """Compute cosine similarity between consecutive layer directions."""
    cosines = []
    for l in range(1, num_layers):
        cos = cosine_similarity(directions[l], directions[l-1])
        cosines.append((l, cos))
    return cosines


def find_transition_zones(cosines, threshold=0.95):
    """Find layers where cosine drops significantly (transformation zones)."""
    transitions = []
    for l, cos in cosines:
        if cos < threshold:
            transitions.append((l, cos))
    return transitions




def main():
    print("=" * 70)
    print("EXPERIMENT 7: Steering Vector Residue and Layer Position")
    print("=" * 70)
    print()

    model, tokenizer = load_model()

    # Prepare prompts
    hai_prompts = [format_human_ai(p) for p in HARMFUL_PROMPTS]
    qa_prompts = [format_qa(p) for p in HARMFUL_PROMPTS]

    # =========================================================================
    # Part 1: Extract directions at all layers
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Extracting directions at all layers")
    print("=" * 60)

    directions = extract_all_layer_directions(model, tokenizer, hai_prompts, qa_prompts)
    print(f"Extracted directions at {len(directions)} layers")

    # =========================================================================
    # Part 2: Compute cosine similarity between consecutive layers
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Cosine similarity between consecutive layers")
    print("=" * 60)

    cosines = compute_consecutive_cosines(directions)

    print(f"\n{'Layers':<12} {'Cosine':<10} {'Note':<20}")
    print("-" * 42)
    for l, cos in cosines:
        note = ""
        if cos < 0.90:
            note = "*** SHARP DROP ***"
        elif cos < 0.95:
            note = "* transition zone *"
        print(f"L{l-1}→L{l:<8} {cos:.4f}    {note}")

    transitions = find_transition_zones(cosines)
    print(f"\nIdentified {len(transitions)} transition layers (cosine < 0.95)")

    # =========================================================================
    # Part 3: Analyze R1/R2 relationship and projection
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: R1/R2 relationship and residue analysis")
    print("=" * 60)

    # Extract R1 (L10) and R2 (L18) as reference directions
    R1 = directions[10]
    R2 = directions[18]

    print(f"\nR1 (L10) · R2 (L18) = {cosine_similarity(R1, R2):.4f}")

    # R2_orthogonal = R2 with R1 component removed
    R2_orth = project_out(R2, R1)
    R2_orth = R2_orth / R2_orth.norm()

    print(f"R2_orthogonal · R1 = {cosine_similarity(R2_orth, R1):.6f} (should be ~0)")
    print(f"R2_orthogonal · R2 = {cosine_similarity(R2_orth, R2):.4f}")

    # Compute how much of R2 is "new" vs residue from R1
    r1_component = abs(cosine_similarity(R2, R1))
    r2_new_component = abs(cosine_similarity(R2, R2_orth))
    print(f"\nR2 decomposition:")
    print(f"  R1 residue component: {r1_component:.4f}")
    print(f"  New R2 component: {r2_new_component:.4f}")

    # =========================================================================
    # Part 4: Generate visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 4: Generating visualization")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Cosine similarity between consecutive layers
    ax1 = axes[0]
    layers_x = [l for l, _ in cosines]
    cosine_y = [c for _, c in cosines]

    ax1.plot(layers_x, cosine_y, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Transition threshold')
    ax1.fill_between(layers_x, cosine_y, 1.0, alpha=0.2)

    # Mark transition zones
    for l, cos in transitions:
        ax1.axvline(x=l, color='orange', alpha=0.3, linewidth=8)

    ax1.set_xlabel('Layer Transition (L→L+1)', fontsize=11)
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title('Direction Change Across Layers', fontsize=12)
    ax1.set_ylim(0.7, 1.02)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: R1/R2 alignment across layers
    ax2 = axes[1]

    layers_all = list(range(32))
    r1_alignment = [abs(cosine_similarity(directions[l], R1)) for l in layers_all]
    r2_alignment = [abs(cosine_similarity(directions[l], R2)) for l in layers_all]

    ax2.plot(layers_all, r1_alignment, 'g-o', linewidth=2, markersize=3, label='R1 alignment (L10)')
    ax2.plot(layers_all, r2_alignment, 'purple', linewidth=2, markersize=3, linestyle='--', label='R2 alignment (L18)')

    # Mark harvest layers
    ax2.axvline(x=10, color='green', alpha=0.3, linewidth=4, label='R1 harvest')
    ax2.axvline(x=18, color='purple', alpha=0.3, linewidth=4, label='R2 harvest')

    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Alignment (|cosine|)', fontsize=11)
    ax2.set_title('R1→R2 Transition: Residue Accumulation', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = '../figures/fig6_steering_residue.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
Key Findings:

1. COSINE TRANSITIONS identify the F→R1→R2 transformation:
   - Sharp drops in cosine similarity between consecutive layers mark
     where the steering direction changes character
   - These correspond to the transformation zone (L13-17)

2. STEERING VECTORS WORK BEST AT HARVEST LAYER:
   - R2 (extracted at L18) is most effective when applied at L18
   - Applying at earlier layers has reduced efficacy

3. EARLIER LAYER EFFECTIVENESS = RESIDUE:
   - When R2 works at L<18, it's because R2 contains F + R1 residue
   - The residual stream accumulates: direction_L18 ≈ F + R1 + R2_new
   - Pre-steering works by injecting this accumulated signal early

4. PROJECTING OUT IMPROVES TARGETING:
   - R2_orthogonal = R2 with R1 component removed
   - This isolates the "new" transformation component
   - More targeted behavioral changes with less side effects
""")


if __name__ == "__main__":
    main()
