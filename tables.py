"""Print the paper's MTSF tables from results/mtsf/<run>/<DATASET>_L16_H<H>_s<seed>/metrics.npz.

    python tables.py [--results results/mtsf]

Table 1: per variant, the mean MSE / MAE over seeds of its best (k, D) arm.
Table 2: per (D, k, H), the MSE averaged over the four datasets of the best variant.
"""

import argparse
import os

import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
PRED_LENS = [1, 5, 16, 32, 48]
SEEDS = range(1, 11)
SEQ_LEN = 16
KS = range(1, 8)
DEPTHS = [1, 3]
VARIANTS = ["QFWP-ANO-RO", "QFWP-ANO-R", "QFWP-ANO-O"]


def run_name(variant, k, depth):
    """Run directory written by scripts/train.sh (D = 3 is untagged)."""
    return f"{variant}-comb{k}local" + ("" if depth == 3 else f"-L{depth}")


def mean_metrics(results, run, dataset, pred_len):
    """(MSE, MAE) averaged over SEEDS, or None unless every seed has finished."""
    mse, mae = [], []
    for seed in SEEDS:
        path = os.path.join(
            results, run, f"{dataset}_L{SEQ_LEN}_H{pred_len}_s{seed}", "metrics.npz"
        )
        if not os.path.isfile(path):
            return None
        z = np.load(path)
        mse.append(float(z["mse"]))
        mae.append(float(z["mae"]))
    return float(np.mean(mse)), float(np.mean(mae))


def best_arm(results, variant, dataset, pred_len):
    """The (k, D) arm of `variant` with the lowest mean MSE: (run, (MSE, MAE))."""
    best = None
    for k in KS:
        for depth in DEPTHS:
            run = run_name(variant, k, depth)
            m = mean_metrics(results, run, dataset, pred_len)
            if m and (best is None or m[0] < best[1][0]):
                best = (run, m)
    return best


def table1(results):
    print("Table 1: MSE / MAE of each variant at its best (k, D); best arm in brackets")
    print(f"{'':>6} {'H':>3}  " + "  ".join(f"{v:>30}" for v in VARIANTS))
    for ds in DATASETS:
        for pl in PRED_LENS:
            cells = []
            for v in VARIANTS:
                b = best_arm(results, v, ds, pl)
                cells.append(
                    "-"
                    if b is None
                    else f"{b[1][0]:.3f} {b[1][1]:.3f} [{b[0][len(v) + 1 :]}]"
                )
            print(f"{ds:>6} {pl:>3}  " + "  ".join(f"{c:>30}" for c in cells))
    print()


def table2(results):
    print("Table 2: MSE averaged over the four datasets, best variant in brackets")
    for depth in DEPTHS:
        print(f"D = {depth}")
        print(f"{'':>5}" + "".join(f"{'H=' + str(pl):>12}" for pl in PRED_LENS))
        for k in KS:
            cells = []
            for pl in PRED_LENS:
                scores = {}
                for v in VARIANTS:
                    ms = [
                        mean_metrics(results, run_name(v, k, depth), ds, pl)
                        for ds in DATASETS
                    ]
                    if all(ms):
                        scores[v] = np.mean([m[0] for m in ms])
                if scores:
                    v = min(scores, key=scores.get)
                    cells.append(f"{scores[v]:.3f} [{v[len('QFWP-ANO-') :]}]")
                else:
                    cells.append("-")
            print(f"{'k=' + str(k):>5}" + "".join(f"{c:>12}" for c in cells))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results/mtsf")
    args = parser.parse_args()
    table1(args.results)
    table2(args.results)
