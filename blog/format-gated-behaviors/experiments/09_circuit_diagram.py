#!/usr/bin/env python3
"""
Generate a circuit diagram showing the refusal mechanism.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_circuit_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    input_color = '#E3F2FD'  # light blue
    gate_color = '#FFF3E0'   # light orange
    neuron_color = '#E8F5E9' # light green
    suppress_color = '#FFEBEE' # light red
    transform_color = '#F3E5F5' # light purple
    output_color = '#E0F7FA'  # light cyan

    # Title
    ax.text(7, 7.6, 'Refusal Circuit: Llama 3.1 8B Instruct',
            fontsize=14, fontweight='bold', ha='center')

    # ===== INPUT SECTION =====
    # Harmful content box
    harm_box = FancyBboxPatch((0.3, 5.5), 2.4, 1.2,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=input_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(harm_box)
    ax.text(1.5, 6.3, 'Harmful Content', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.5, 5.85, '"How to pick a lock?"', fontsize=8, ha='center', style='italic')

    # Format tokens box
    format_box = FancyBboxPatch((0.3, 3.8), 2.4, 1.2,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=input_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(format_box)
    ax.text(1.5, 4.6, 'Format Tokens', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.5, 4.15, '"Human: ... AI:"', fontsize=8, ha='center', style='italic')

    # ===== AND GATE =====
    # AND gate circle
    and_circle = Circle((4, 5.1), 0.6, facecolor=gate_color, edgecolor='#E65100', linewidth=2)
    ax.add_patch(and_circle)
    ax.text(4, 5.1, 'AND', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(4, 4.3, 'L8-13\nAttention', fontsize=8, ha='center', color='#666')

    # Arrows to AND gate
    ax.annotate('', xy=(3.4, 5.4), xytext=(2.7, 6.1),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
    ax.annotate('', xy=(3.4, 4.8), xytext=(2.7, 4.4),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))

    # ===== N4258 TRANSLATOR =====
    n4258_box = FancyBboxPatch((5.2, 4.4), 2.2, 1.4,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=neuron_color, edgecolor='#388E3C', linewidth=2)
    ax.add_patch(n4258_box)
    ax.text(6.3, 5.45, 'N4258 @ L11', fontsize=10, fontweight='bold', ha='center')
    ax.text(6.3, 5.0, 'Harm Detector', fontsize=9, ha='center')
    ax.text(6.3, 4.6, 'read: H → write: R1', fontsize=8, ha='center', color='#666')

    # Arrow from AND to N4258
    ax.annotate('', xy=(5.2, 5.1), xytext=(4.6, 5.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Activation gradient annotation
    ax.text(6.3, 3.9, '0.01 → 0.52 → 0.59', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#388E3C', alpha=0.8))
    ax.text(6.3, 3.5, 'harmless → borderline → harmful', fontsize=7, ha='center', color='#666')

    # ===== SUPPRESSOR NEURONS =====
    suppress_box = FancyBboxPatch((5.2, 1.2), 2.2, 1.4,
                                   boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor=suppress_color, edgecolor='#C62828', linewidth=2)
    ax.add_patch(suppress_box)
    ax.text(6.3, 2.25, 'Suppressors', fontsize=10, fontweight='bold', ha='center')
    ax.text(6.3, 1.8, 'L12-14 MLPs', fontsize=9, ha='center')
    ax.text(6.3, 1.4, '−1.61 contribution', fontsize=8, ha='center', color='#C62828')

    # ===== COMPETITION ZONE =====
    compete_box = FancyBboxPatch((8, 2.8), 2.4, 2.6,
                                  boxstyle="round,pad=0.05,rounding_size=0.2",
                                  facecolor=transform_color, edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(compete_box)
    ax.text(9.2, 5.0, 'Competition', fontsize=10, fontweight='bold', ha='center')
    ax.text(9.2, 4.55, 'L13-17', fontsize=9, ha='center')

    # Builder vs suppressor
    ax.text(9.2, 3.9, 'Builders: +2.92', fontsize=9, ha='center', color='#388E3C')
    ax.text(9.2, 3.5, 'Suppressors: −1.61', fontsize=9, ha='center', color='#C62828')
    ax.plot([8.3, 10.1], [3.2, 3.2], 'k-', linewidth=1)
    ax.text(9.2, 3.0, 'Net: +1.31', fontsize=10, ha='center', fontweight='bold')

    # Arrows to competition
    ax.annotate('', xy=(8, 4.5), xytext=(7.4, 5.1),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2))
    ax.annotate('', xy=(8, 3.5), xytext=(7.4, 2.0),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2))

    # N4258 contribution annotation
    ax.annotate('+0.94\n(N4258)', xy=(7.7, 4.8), fontsize=8, ha='center', color='#388E3C')

    # ===== OUTPUT =====
    output_box = FancyBboxPatch((11, 3.4), 2.6, 1.4,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=output_color, edgecolor='#00838F', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.3, 4.45, 'Output (L17+)', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.3, 3.95, 'R1 → R2', fontsize=9, ha='center')
    ax.text(12.3, 3.6, 'REFUSE', fontsize=11, ha='center', fontweight='bold', color='#C62828')

    # Arrow to output
    ax.annotate('', xy=(11, 4.1), xytext=(10.4, 4.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # ===== KEY INSIGHT BOX =====
    insight_box = FancyBboxPatch((0.3, 0.3), 13.4, 0.9,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor='#FFFDE7', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(7, 0.75, 'Key Insight: N4258\'s +0.94 contribution is what tips the balance past the suppressors. '
                     'Without it, net signal ≈ 0 → no refusal.',
            fontsize=9, ha='center', va='center')

    # ===== BYPASS ANNOTATION =====
    ax.annotate('', xy=(4, 3.4), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5, linestyle='dashed'))
    ax.text(4.6, 3.8, '−F at L12\n= bypass', fontsize=8, ha='left', color='#666', style='italic')

    plt.tight_layout()
    plt.savefig('../figures/fig_circuit_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved fig_circuit_diagram.png")
    plt.close()


if __name__ == "__main__":
    create_circuit_diagram()
