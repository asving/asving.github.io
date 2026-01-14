#!/usr/bin/env python3
"""
Generate additional plots for the blog post:
1. N4258 activation gradient across prompt types (harm detector evidence)
2. AND gate visualization (2x2: harmful/harmless × format)
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_n4258_activation_gradient():
    """
    Bar chart showing N4258's activation correlates with perceived harm.
    Data from REFUSAL_CIRCUIT_FINDINGS.md Section 6.7
    """
    categories = ['Harmless', 'Borderline', 'Harmful']
    activations = [0.007, 0.517, 0.592]
    colors = ['#4CAF50', '#FFC107', '#F44336']  # green, yellow, red

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(categories, activations, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, activations):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Mean Activation', fontsize=12)
    ax.set_xlabel('Prompt Category', fontsize=12)
    ax.set_title('Neuron 4258 @ Layer 11: Harm Detection Gradient', fontsize=13)
    ax.set_ylim(0, 0.75)

    # Add annotation explaining the finding
    ax.text(0.5, 0.92, 'Activation scales with perceived harm level',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            color='#555')

    # Add example prompts as annotations
    examples = [
        '"How to make pancakes?"',
        '"What chemicals are explosive?"',
        '"How to make a bomb?"'
    ]
    for i, (bar, ex) in enumerate(zip(bars, examples)):
        ax.annotate(ex, xy=(bar.get_x() + bar.get_width()/2, -0.08),
                    ha='center', fontsize=8, color='#666',
                    xycoords=('data', 'axes fraction'))

    plt.tight_layout()
    plt.savefig('../figures/fig7_n4258_harm_gradient.png', dpi=150, bbox_inches='tight')
    print("Saved fig7_n4258_harm_gradient.png")
    plt.close()


def plot_and_gate_heatmap():
    """
    2x2 heatmap showing the AND gate: refusal requires BOTH harmful content AND assistant format.
    Data from experiments 01 and behavioral testing.
    """
    # Data: refusal rate (0-100%)
    # Rows: Harmless, Harmful
    # Cols: Q/A format, Human/AI format
    data = np.array([
        [0, 0],      # Harmless: 0% refuse in Q/A, 0% refuse in Human/AI
        [0, 100],    # Harmful: 0% refuse in Q/A, 100% refuse in Human/AI
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Custom colormap: white for 0, red for 100
    cmap = plt.cm.Reds
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Q/A Format\n(completion-style)', 'Human/AI Format\n(chat-style)'], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Harmless\nPrompts', 'Harmful\nPrompts'], fontsize=11)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = data[i, j]
            color = 'white' if val > 50 else 'black'
            text = f'{int(val)}%\nrefuse'
            if val == 0:
                text = '0%\nrefuse\n(comply)'
            elif val == 100:
                text = '100%\nrefuse'
            ax.text(j, i, text, ha='center', va='center', fontsize=12,
                    color=color, fontweight='bold')

    ax.set_title('The AND Gate: Refusal Requires Both Conditions', fontsize=13, pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Refusal Rate (%)', fontsize=10)

    # Add annotation
    ax.text(0.5, -0.18, 'Only harmful content + chat format triggers refusal',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            color='#555')

    plt.tight_layout()
    plt.savefig('../figures/fig8_and_gate.png', dpi=150, bbox_inches='tight')
    print("Saved fig8_and_gate.png")
    plt.close()


def plot_sycophancy_control_comparison():
    """
    Comparison showing format doesn't affect sycophancy, but instructions do.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Format tokens (no effect)
    ax1 = axes[0]
    formats = ['Q/A', 'Human/AI', 'User/Asst', 'A/B']
    lie_rates = [0, 0, 0, 0]  # All 0% from experiment 02

    bars1 = ax1.bar(formats, lie_rates, color='#90CAF9', edgecolor='black')
    ax1.set_ylabel('Lie Rate (%)', fontsize=11)
    ax1.set_xlabel('Format Tokens', fontsize=11)
    ax1.set_title('Format Tokens: No Effect', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5)

    for bar in bars1:
        ax1.annotate('0%', xy=(bar.get_x() + bar.get_width()/2, 5),
                    ha='center', fontsize=10, fontweight='bold')

    # Right: System prompts (strong effect)
    ax2 = axes[1]
    prompts = ['Neutral', 'Agreeable', 'Must Agree']
    lie_rates2 = [0, 14, 100]  # From experiment 03
    colors2 = ['#4CAF50', '#FFC107', '#F44336']

    bars2 = ax2.bar(prompts, lie_rates2, color=colors2, edgecolor='black')
    ax2.set_ylabel('Lie Rate (%)', fontsize=11)
    ax2.set_xlabel('System Prompt', fontsize=11)
    ax2.set_title('System Prompts: Strong Effect', fontsize=12)
    ax2.set_ylim(0, 110)

    for bar, val in zip(bars2, lie_rates2):
        ax2.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width()/2, val + 3),
                    ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Sycophancy Control: Instructions Matter, Format Doesn\'t', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig9_sycophancy_control.png', dpi=150, bbox_inches='tight')
    print("Saved fig9_sycophancy_control.png")
    plt.close()


if __name__ == "__main__":
    print("Generating additional plots...")
    print()

    plot_n4258_activation_gradient()
    plot_and_gate_heatmap()
    plot_sycophancy_control_comparison()

    print()
    print("Done! Generated:")
    print("  - fig7_n4258_harm_gradient.png")
    print("  - fig8_and_gate.png")
    print("  - fig9_sycophancy_control.png")
