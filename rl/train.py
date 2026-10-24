"""Train one RL agent: one (arm, env, input scaling, seed) cell.

python -m rl.train --arm QFWP-ANO-RO --env_id Acrobot-v1 --seed 0 [--learnable_scaling 1]
"""

import argparse
import json
import logging
import os
import pickle
import time

from rl.a3c import GAMMA, UPDATE_GLOBAL_ITER, train
from rl.agents import ARMS, Net, param_split
from rl.envs import ENV_IDS, EPISODE_BUDGET, make_env

RESULTS = os.path.join("results", "rl")


def result_dir(env_id, arm, depth, seed, learnable_scaling):
    """results/rl/<env>/<arm>/d<D>[-inputscale]/seed<N>"""
    tag = f"d{depth}" + ("-inputscale" if learnable_scaling else "")
    return os.path.join(RESULTS, env_id, arm, tag, f"seed{seed}")


def build_parser():
    p = argparse.ArgumentParser(description="QFWP-ANO RL: one run")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--env_id", required=True, choices=list(ENV_IDS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--max_ep",
        type=int,
        default=0,
        help="episode budget; 0 uses the env's budget in rl.envs.EPISODE_BUDGET",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--depth", type=int, default=4, help="VQC depth D")
    p.add_argument("--n_local", type=int, default=3, help="ANO locality k")
    p.add_argument("--num_qubits", type=int, default=4)
    p.add_argument("--d_hidden", type=int, default=16)
    p.add_argument("--d_dec", type=int, default=16)
    p.add_argument(
        "--learnable_scaling",
        type=int,
        default=0,
        help="1 trains the input scaling w on the encoding gates",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.max_ep <= 0:
        args.max_ep = EPISODE_BUDGET[args.env_id]

    out = result_dir(
        args.env_id, args.arm, args.depth, args.seed, bool(args.learnable_scaling)
    )
    if os.path.exists(os.path.join(out, "episode_returns.pkl")):
        raise FileExistsError(f"{out} already holds results; delete it to re-run.")
    os.makedirs(out, exist_ok=True)

    net_kwargs = {
        "C": args.num_qubits,
        "D": args.depth,
        "n_local": args.n_local,
        "d_hidden": args.d_hidden,
        "d_dec": args.d_dec,
        "learnable_scaling": bool(args.learnable_scaling),
    }

    t0 = time.time()
    returns, steps = train(
        arm=args.arm,
        env_id=args.env_id,
        seed=args.seed,
        max_ep=args.max_ep,
        workers=args.workers,
        net_kwargs=net_kwargs,
        save_state_dict_to=os.path.join(out, "model.pth"),
    )
    elapsed = time.time() - t0

    probe = make_env(args.env_id)
    ref = Net(
        args.arm, probe.observation_space.shape[0], probe.action_space.n, **net_kwargs
    )

    with open(os.path.join(out, "episode_steps.pkl"), "wb") as f:
        pickle.dump(steps, f)
    config = dict(vars(args))
    config.update(
        gamma=GAMMA,
        update_global_iter=UPDATE_GLOBAL_ITER,
        param_split=param_split(ref),
        episodes=len(returns),
        total_env_steps=steps[-1] if steps else 0,
        elapsed_sec=round(elapsed, 1),
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    # Written last: it marks the cell complete.
    with open(os.path.join(out, "episode_returns.pkl"), "wb") as f:
        pickle.dump(returns, f)

    print(
        f"wrote {out}  episodes={len(returns)} "
        f"steps={config['total_env_steps']} elapsed={elapsed / 3600:.2f}h"
    )
    return out


if __name__ == "__main__":
    main()
