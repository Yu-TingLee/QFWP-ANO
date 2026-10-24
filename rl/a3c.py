"""A3C training loop (GAMMA = 0.9, 5-step updates, shared Adam, actor + critic loss).

Workers are forked processes that share the global net and optimizer moments.
"""

import logging
import os
import queue as pyqueue

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.multiprocessing as mp

from rl.agents import Net
from rl.envs import make_env

logger = logging.getLogger(__name__)

GAMMA = 0.9
UPDATE_GLOBAL_ITER = 5
MP = mp.get_context("fork")

# Set once loss.backward() has run in this process; train() refuses to fork
# after that (see its docstring).
_BACKWARD_RAN_IN_THIS_PROCESS = False


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


class SharedAdam(torch.optim.Adam):
    """Adam whose moment estimates live in shared memory.

    `step` is a tensor (torch>=2 requires it) and stays private to each worker,
    so bias correction is per worker.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def _worker_seed(run_seed, wid):
    """Base seed of worker `wid`: a disjoint 10**6-wide block per worker inside a
    10**8-wide block per run seed."""
    return run_seed * 10**8 + wid * 10**6


def _episode_seed(run_seed, wid, episode_idx):
    """Env-reset seed, unique per (run seed, worker, episode).

    One worker can claim far more than its share of the global episode budget,
    so each worker's block is wider than the whole budget. On MiniGrid the reset
    seed picks the layout, so a collision would correlate seeds."""
    return run_seed * 10**8 + wid * 10**6 + episode_idx


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    """One 5-step A3C update: local gradients onto the global net, then sync."""
    if done:
        v_s_ = 0.0
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    states = v_wrap(np.vstack(bs))
    actions = v_wrap(np.array(ba), dtype=np.int64)
    targets = v_wrap(np.array(buffer_v_target)[:, None])

    loss = lnet.loss_func(states, actions, targets)

    opt.zero_grad()
    # `opt` holds the GLOBAL net's parameters, so opt.zero_grad() does not clear
    # the local grads; with set_to_none=True (torch>=2) it also breaks the alias
    # below. Without this call loss.backward() would accumulate local gradients
    # across every update.
    lnet.zero_grad()
    loss.backward()
    global _BACKWARD_RAN_IN_THIS_PROCESS
    _BACKWARD_RAN_IN_THIS_PROCESS = True

    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()
    lnet.load_state_dict(gnet.state_dict())


class Worker(MP.Process):
    def __init__(
        self,
        gnet,
        opt,
        global_ep,
        res_queue,
        name,
        env_id,
        max_ep,
        arm,
        net_kwargs,
        seed,
    ):
        super().__init__()
        self.name = f"w{name:02d}"
        self.gnet, self.opt = gnet, opt
        self.g_ep, self.res_queue = global_ep, res_queue
        self.env_id, self.max_ep = env_id, max_ep
        self.arm, self.net_kwargs, self.seed, self.wid = arm, net_kwargs, seed, name

    def run(self):
        worker_seed = _worker_seed(self.seed, self.wid)
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        env = make_env(self.env_id)
        lnet = Net(
            self.arm,
            env.observation_space.shape[0],
            env.action_space.n,
            **self.net_kwargs,
        )
        lnet.load_state_dict(self.gnet.state_dict())  # calibrated global init
        total_step = 1
        episode_idx = 0
        while self.g_ep.value < self.max_ep:
            s, _ = env.reset(seed=_episode_seed(self.seed, self.wid, episode_idx))
            episode_idx += 1
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r, ep_steps = 0.0, 0
            while True:
                a = lnet.choose_action(v_wrap(s[None, :]))
                s_, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                ep_r += r
                ep_steps += 1
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(
                        self.opt,
                        lnet,
                        self.gnet,
                        done,
                        s_,
                        buffer_s,
                        buffer_a,
                        buffer_r,
                        GAMMA,
                    )
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        with self.g_ep.get_lock():
                            self.g_ep.value += 1
                        self.res_queue.put((ep_r, ep_steps))
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


def collect_random_states(env_id, n=512, seed=0):
    """Observations from a uniform-random policy, used to calibrate the
    hypernetworks' output scale to this env's input range."""
    env = make_env(env_id)
    # reset(seed=) does not seed action_space.sample(); seed it too so the batch
    # is reproducible.
    env.action_space.seed(seed)
    out, (s, _) = [], env.reset(seed=seed)
    while len(out) < n:
        out.append(s)
        s, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            s, _ = env.reset()
    return torch.from_numpy(np.asarray(out[:n], dtype=np.float32))


PROGRESS_EVERY = 500  # episodes between progress log lines


def _collect_episodes(
    res_queue, ws, workers, max_ep, timeout=120, progress_every=PROGRESS_EVERY
):
    """Drain `res_queue` until every worker is accounted for; returns
    (returns, cumulative steps).

    A worker is accounted for once: by its sentinel (None) when it exits
    cleanly, or by the timeout path when it crashes without sending one. Raises
    only if every worker is gone and the budget was not met.
    """
    returns, steps, sentinels, cum = [], [], 0, 0
    credited_crashed = set()
    while sentinels + len(credited_crashed) < workers:
        try:
            item = res_queue.get(timeout=timeout)
        except pyqueue.Empty:
            newly_crashed = [
                w
                for w in ws
                if w.exitcode not in (None, 0) and id(w) not in credited_crashed
            ]
            for w in newly_crashed:
                credited_crashed.add(id(w))
                logger.warning(
                    "worker %s crashed (exitcode=%s); treating its "
                    "outstanding sentinel as delivered so the run can "
                    "still finish if the episode budget is otherwise met",
                    w.name,
                    w.exitcode,
                )
            if newly_crashed:
                continue
            if not any(w.is_alive() for w in ws):
                if len(returns) >= max_ep:
                    break
                raise RuntimeError(
                    f"all workers died before finishing ({len(returns)} of "
                    f"{max_ep} episodes collected); check worker tracebacks"
                )
            continue
        if item is None:
            sentinels += 1
        elif len(returns) < max_ep:
            ep_r, ep_steps = item
            returns.append(ep_r)
            cum += ep_steps
            steps.append(cum)
            if progress_every and len(returns) % progress_every == 0:
                recent = returns[-100:]
                logger.info(
                    "progress: %d/%d episodes | %d env steps | "
                    "mean return (last %d) %.1f",
                    len(returns),
                    max_ep,
                    cum,
                    len(recent),
                    sum(recent) / len(recent),
                )
    return returns, steps


def train(
    arm, env_id, seed, max_ep, workers=16, net_kwargs=None, save_state_dict_to=None
):
    """Run one A3C job; returns (episode returns, cumulative env steps).

    `steps[i]` sums the lengths of the episodes completed so far, in the order
    they arrived from the workers. `save_state_dict_to` saves the trained global
    net.

    Call this before any loss.backward() in this process: fork() copies only the
    calling thread, so a lock held by an autograd thread would stay locked in
    every worker and the run would hang. train() raises instead.
    """
    if _BACKWARD_RAN_IN_THIS_PROCESS:
        raise RuntimeError(
            "train() cannot fork workers after autograd has run in this "
            "process; call it before any backward() or run it in a fresh "
            "process."
        )
    net_kwargs = dict(net_kwargs or {})
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe = make_env(env_id)
    gnet = Net(
        arm, probe.observation_space.shape[0], probe.action_space.n, **net_kwargs
    )
    gnet.calibrate(collect_random_states(env_id, n=512, seed=seed))
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))

    global_ep = MP.Value("i", 0)
    res_queue = MP.Queue()
    ws = [
        Worker(
            gnet, opt, global_ep, res_queue, i, env_id, max_ep, arm, net_kwargs, seed
        )
        for i in range(workers)
    ]
    for w in ws:
        w.start()

    returns, steps = _collect_episodes(res_queue, ws, workers, max_ep)

    for w in ws:
        w.join(timeout=30)

    if save_state_dict_to is not None:
        torch.save(gnet.state_dict(), save_state_dict_to)

    return returns, steps
