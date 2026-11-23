from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def draw(ax, x, shape, title=None):
    ax.imshow(np.asarray(x).reshape(shape), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)


def show_grid(patterns, shape, path=None, ncols=8, titles=None):
    p = len(patterns)
    ncols = min(ncols, p)
    nrows = -(-p // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, x in enumerate(patterns):
        axs.ravel()[i].axis("on")
        draw(axs.ravel()[i], x, shape, None if titles is None else titles[i])
    save(fig, path)


def plot_history(history, shape, path=None, max_frames=12, ncols=6, label="krok"):
    idx = np.unique(np.linspace(0, len(history) - 1, min(len(history), max_frames)).astype(int))
    show_grid([history[i] for i in idx], shape, path, ncols, [f"{label} {i}" for i in idx])


def plot_energy(net, history, path=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([net.energy(x) for x in history], marker="o", ms=3)
    ax.set_xlabel("klatka")
    ax.set_ylabel("energia E")
    save(fig, path)


def animate(history, shape, path=None, interval=300, net=None):
    fig, ax = plt.subplots(figsize=(5, 5 * shape[0] / shape[1] + 0.4))
    im = ax.imshow(history[0].reshape(shape), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    def upd(i):
        im.set_data(history[i].reshape(shape))
        e = f", E={net.energy(history[i]):.2f}" if net is not None else ""
        ax.set_title(f"klatka {i}/{len(history) - 1}{e}")
        return [im]

    anim = FuncAnimation(fig, upd, frames=len(history), interval=interval, repeat=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=max(1, 1000 // interval)))
        plt.close(fig)
    else:
        plt.show()
    return anim
