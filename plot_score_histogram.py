"""
Plot 3: MRMC score distribution overlaid with per-method selection histograms.

Shows *where* each method's selections fall on the MRMC score axis. Makes it
visually obvious that:
  - Pure MRMC concentrates on the right (hardest) tail
  - Random spreads uniformly by rank
  - Score-stratified spans the whole distribution
  - K-Means sits between random and top-k

One reference MRMC score array is used (from the first MRMC method that
exposed `last_scores`). Each method gets its own subplot.

Usage:
    python plot_score_histogram.py
"""
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, 'run_*.pkl')))
    if not paths:
        raise FileNotFoundError(f"No run_*.pkl found in {RESULTS_DIR}. Run experiment_pipeline.py first.")

    # Use the first run as the reference
    with open(paths[0], 'rb') as f:
        run = pickle.load(f)

    # Pick a reference score array (first method with mrmc_scores stored)
    ref_scores = None
    ref_name = None
    for name, rec in run.items():
        if name.startswith('_'):
            continue
        if rec.get('mrmc_scores') is not None:
            ref_scores = np.asarray(rec['mrmc_scores'])
            ref_name = name
            break
    if ref_scores is None:
        raise RuntimeError("No MRMC scores stored in any method's result.")

    methods = [n for n in run.keys() if not n.startswith('_')]
    n = len(methods)

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    bins = np.linspace(ref_scores.min(), ref_scores.max(), 80)

    for ax, name in zip(axes, methods):
        rec = run[name]
        idx = np.asarray(rec['coreset_indices'])
        sel_scores = ref_scores[idx]

        ax.hist(ref_scores, bins=bins, color='lightgray', alpha=0.6,
                label=f'All train samples (n={len(ref_scores)})')
        ax.hist(sel_scores, bins=bins, color='tomato', alpha=0.75,
                label=f'{name} selections (n={len(idx)})')
        ax.set_ylabel('count')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)
        ax.set_yscale('log')

    axes[-1].set_xlabel(f'MRMC score (reference: {ref_name})')
    fig.suptitle('Where each method samples on the MRMC score axis', y=1.0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'plot_score_histogram.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
