"""
Plot 4: Per-class coreset composition (grouped bar chart).

For each method, count how many samples per digit class landed in the coreset.
Makes class-balance pathologies visible: e.g. pure top-k MRMC starves easy
classes (1, 0) and over-picks hard pairs (4/9, 3/8).

Uses the first run's coreset indices and training labels (class composition
is not seed-dependent in any meaningful way for these methods).

Usage:
    python plot_class_composition.py
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

    with open(paths[0], 'rb') as f:
        run = pickle.load(f)

    labels = run['_meta']['train_labels']
    num_classes = run['_meta'].get('num_classes', int(labels.max()) + 1)
    classes = np.arange(num_classes)

    methods = [n for n in run.keys() if not n.startswith('_')]
    counts = np.zeros((len(methods), num_classes), dtype=int)
    for i, name in enumerate(methods):
        idx = np.asarray(run[name]['coreset_indices'])
        sel_labels = labels[idx]
        for c in classes:
            counts[i, c] = int((sel_labels == c).sum())

    fig, ax = plt.subplots(figsize=(14, 6))
    w = 0.8 / len(methods)
    x = np.arange(num_classes)
    cmap = plt.cm.tab10
    for i, name in enumerate(methods):
        ax.bar(x + i * w, counts[i], w, label=name, color=cmap(i % 10))

    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel('Digit class')
    ax.set_ylabel('Coreset samples (count)')
    ax.set_title(f'Per-Class Coreset Composition (MNIST, {int(run["_meta"]["coreset_fraction"]*100)}% fraction)')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Reference line: perfectly balanced coreset
    total = sum(run[methods[0]]['coreset_indices'].shape[0] for _ in [0])
    ideal = run[methods[0]]['coreset_indices'].shape[0] / num_classes
    ax.axhline(ideal, color='black', linestyle='--', alpha=0.4, label=f'balanced reference ({ideal:.0f})')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'plot_class_composition.png')
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
