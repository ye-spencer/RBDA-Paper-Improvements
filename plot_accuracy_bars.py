"""
Plot 1: Test accuracy bar chart per method.

Loads results/run_*.pkl, averages test accuracy over seeds, renders a
horizontal bar chart sorted low-to-high with mean±std labels.

Usage:
    python plot_accuracy_bars.py
"""
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def load_runs():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, 'run_*.pkl')))
    if not paths:
        raise FileNotFoundError(f"No run_*.pkl found in {RESULTS_DIR}. Run experiment_pipeline.py first.")
    runs = []
    for p in paths:
        with open(p, 'rb') as f:
            runs.append(pickle.load(f))
    return runs


def main():
    runs = load_runs()
    n_seeds = len(runs)

    accs = {}
    for run in runs:
        for name, rec in run.items():
            if name.startswith('_'):
                continue
            accs.setdefault(name, []).append(rec['test_acc'])

    methods = list(accs.keys())
    means = np.array([np.mean(accs[m]) for m in methods])
    stds = np.array([np.std(accs[m]) for m in methods])

    order = np.argsort(means)
    methods = [methods[i] for i in order]
    means = means[order]
    stds = stds[order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(methods, means, xerr=stds, capsize=4,
                   color='steelblue', edgecolor='black')

    # highlight best and random for storytelling
    for bar, name in zip(bars, methods):
        if name == 'RandomCoresetSelection':
            bar.set_color('gray')
        elif name == 'FullDatasetSelection':
            bar.set_color('seagreen')
        elif name == methods[-1]:
            bar.set_color('tomato')

    ax.set_xlabel('Test Accuracy')
    ax.set_xlim(min(0.70, means.min() - 0.03), 1.0)
    ax.set_title(f'Coreset Selection Methods on MNIST (10% fraction, n={n_seeds} seed{"s" if n_seeds > 1 else ""})')
    ax.grid(axis='x', alpha=0.3)

    for i, (m, s) in enumerate(zip(means, stds)):
        label = f'{m*100:.2f}%' if n_seeds == 1 else f'{m*100:.2f} ± {s*100:.2f}%'
        ax.text(m + max(s, 0) + 0.003, i, label, va='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'plot_accuracy_bars.png')
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
