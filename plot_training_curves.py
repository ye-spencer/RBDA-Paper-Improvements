"""
Plot 2: Training curves (loss and accuracy vs epoch) overlaid across methods.

Averages across seeds; fills ±1 std if multiple seeds are available.
Two side-by-side subplots: train loss and train accuracy.

Usage:
    python plot_training_curves.py
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
    return [pickle.load(open(p, 'rb')) for p in paths]


def main():
    runs = load_runs()
    n_seeds = len(runs)

    curves = {}  # name -> {'losses': [n_seeds, n_epochs], 'accs': ...}
    for run in runs:
        for name, rec in run.items():
            if name.startswith('_'):
                continue
            curves.setdefault(name, {'losses': [], 'accs': []})
            curves[name]['losses'].append(rec['train_losses'])
            curves[name]['accs'].append(rec['train_accs'])

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.tab10
    for i, (name, data) in enumerate(curves.items()):
        L = np.asarray(data['losses'])   # (seeds, epochs)
        A = np.asarray(data['accs'])
        epochs = np.arange(1, L.shape[1] + 1)
        color = cmap(i % 10)

        ax_loss.plot(epochs, L.mean(0), label=name, color=color, linewidth=2)
        ax_acc.plot(epochs, A.mean(0), label=name, color=color, linewidth=2)
        if n_seeds > 1:
            ax_loss.fill_between(epochs, L.mean(0) - L.std(0), L.mean(0) + L.std(0), alpha=0.15, color=color)
            ax_acc.fill_between(epochs, A.mean(0) - A.std(0), A.mean(0) + A.std(0), alpha=0.15, color=color)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Train Loss')
    ax_loss.set_title('Training Loss per Coreset Method')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8, loc='upper right')

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Train Accuracy')
    ax_acc.set_title('Training Accuracy per Coreset Method')
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8, loc='lower right')

    fig.suptitle(f'Training Dynamics (MNIST, 10% coreset, n={n_seeds} seed{"s" if n_seeds > 1 else ""})', y=1.02)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'plot_training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
