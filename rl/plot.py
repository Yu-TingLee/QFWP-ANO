"""The paper's RL figure: return over episodes, Acrobot above SimpleCrossingS9N1.

Each arm is drawn with (opaque) and without (faded) input scaling w; the line is
the seed mean of the moving-average return and the band is +/- one std.

    python -m rl.plot [--out figures/rl_returns.png]
"""

import argparse
import os
import pickle

import numpy as np

from rl.agents import ARMS, PROGRAMMED_ARMS
from rl.envs import EPISODE_BUDGET
from rl.train import result_dir

COLOR = {
    "ANO-VQC": "#1f77b4",
    "QFWP-ANO-R": "#2ca02c",
    "QFWP-ANO-O": "#d62728",
    "QFWP-ANO-RO": "#9467bd",
}
# Per-episode return range from each env's reward function.
REWARD_RANGE = {"Acrobot-v1": (-500, 0), "MiniGrid-SimpleCrossingS9N1-v0": (0, 1)}
# (env_id, title, moving-average window)
PANELS = (
    ("Acrobot-v1", "Acrobot", 50),
    ("MiniGrid-SimpleCrossingS9N1-v0", "SimpleCrossingS9N1", 150),
)
LINE_ALPHA = {1: 1.0, 0: 0.5}  # w/ w opaque, w/o w faded


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_returns(env_id, arm, depth, seeds, learnable_scaling):
    """(seeds, episodes) returns, or None if any seed is missing."""
    rets = []
    for s in seeds:
        path = os.path.join(
            result_dir(env_id, arm, depth, s, learnable_scaling), "episode_returns.pkl"
        )
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            rets.append(np.asarray(pickle.load(f), dtype=float))
    n = min(len(r) for r in rets)
    return np.stack([r[:n] for r in rets])


def _plot_panel(ax, env_id, window, arms, depth, seeds, show_xlabel):
    curves = []
    for arm in arms:
        for learnable_scaling in (1, 0):
            returns = load_returns(env_id, arm, depth, seeds, learnable_scaling)
            if returns is None:
                print(f"skip {env_id} {arm} scaling={learnable_scaling}: missing seeds")
                continue
            ma = np.stack([moving_average(r, window) for r in returns])
            mu, sd = ma.mean(0), ma.std(0)
            # mode="valid": ma[i] covers episodes i..i+window-1, so plot it at the
            # window's right edge.
            x = np.arange(len(mu)) + (returns.shape[1] - len(mu))
            curves.append((arm, learnable_scaling, x, mu, sd))

    # All bands first, then all lines, so no band mutes an earlier line.
    for arm, _, x, mu, sd in curves:
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[arm], alpha=0.10, lw=0)
    for arm, learnable_scaling, x, mu, _ in curves:
        ax.plot(x, mu, color=COLOR[arm], lw=2.0, alpha=LINE_ALPHA[learnable_scaling])
    if show_xlabel:
        ax.set_xlabel("Episode")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.set_ylim(*REWARD_RANGE[env_id])
    ax.set_xlim(0, EPISODE_BUDGET[env_id])


def make_figure(arms, depth, seeds, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 1, figsize=(6, 8.8))
    for i, (ax, (env_id, title, window)) in enumerate(zip(axes, PANELS)):
        _plot_panel(
            ax, env_id, window, arms, depth, seeds, show_xlabel=i == len(axes) - 1
        )
        ax.set_title(title)

    def handles(scaling, row_label):
        # The row label is an invisible first handle, so it reads as leading text.
        return [Line2D([0], [0], color="none", label=row_label)] + [
            Line2D([0], [0], color=COLOR[a], lw=2.0, alpha=LINE_ALPHA[scaling], label=a)
            for a in ordered
        ]

    ordered = [a for a in arms if a not in PROGRAMMED_ARMS] + [
        a for a in arms if a in PROGRAMMED_ARMS
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    xc = 0.5
    leg_kw = {
        "loc": "upper center",
        "ncol": len(ordered) + 1,
        "frameon": False,
        "columnspacing": 1.0,
        "handletextpad": 0.5,
        "handlelength": 1.5,
    }
    leg_top = fig.legend(
        handles=handles(1, r"w/ $\mathbf{w}$"), bbox_to_anchor=(xc, 0.995), **leg_kw
    )
    fig.add_artist(leg_top)
    leg_bot = fig.legend(
        handles=handles(0, r"w/o $\mathbf{w}$"), bbox_to_anchor=(xc, 0.955), **leg_kw
    )

    # The row labels differ in width; shift the bottom row so the swatches of
    # each arm line up vertically.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_x0 = leg_top.legend_handles[1].get_window_extent(renderer).x0
    bot_x0 = leg_bot.legend_handles[1].get_window_extent(renderer).x0
    shift = (top_x0 - bot_x0) / fig.bbox.width
    leg_bot.remove()
    leg_bot = fig.legend(
        handles=handles(0, r"w/o $\mathbf{w}$"),
        bbox_to_anchor=(xc + shift, 0.955),
        **leg_kw,
    )

    # A rule between the two legend rows.
    fig.canvas.draw()
    b1 = leg_top.get_window_extent(renderer)
    b2 = leg_bot.get_window_extent(renderer)
    x0, x1 = min(b1.x0, b2.x0), max(b1.x1, b2.x1)
    y_mid = (b1.y0 + b2.y1) / 2
    inv = fig.transFigure.inverted()
    (fx0, fy), (fx1, _) = inv.transform((x0, y_mid)), inv.transform((x1, y_mid))
    fig.add_artist(
        Line2D(
            [fx0, fx1], [fy, fy], transform=fig.transFigure, color="0.4", linewidth=0.8
        )
    )

    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out", default="figures/rl_returns.png")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    make_figure(a.arms, a.depth, a.seeds, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
