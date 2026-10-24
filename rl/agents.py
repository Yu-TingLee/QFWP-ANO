"""Actor-critic agents for the RL experiments.

Each agent embeds two QFWP-ANO instances (policy and value). The ANO-VQC
baseline has no hypernetwork; the QFWP-ANO arms' hypernetworks read the raw
state s_t while the circuit encodes h_t = W s_t + b.
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.qfwp_ano import INIT_STD, QFWPANO

ARMS = ("ANO-VQC", "QFWP-ANO-R", "QFWP-ANO-O", "QFWP-ANO-RO")
PROGRAMMED_ARMS = ("QFWP-ANO-R", "QFWP-ANO-O", "QFWP-ANO-RO")


class QFeature(QFWPANO):
    """QFWP-ANO with its hypernetwork's output heads scaled so the generated
    angles and observables start at the static arms' init std."""

    def __init__(
        self,
        arm,
        s_dim,
        C=4,
        D=4,
        n_local=3,
        learnable_scaling=False,
        d_hidden=16,
        d_dec=16,
        qml_device="default.qubit",
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        super().__init__(
            arm, s_dim, C, D, n_local, d_hidden, d_dec, learnable_scaling, qml_device
        )
        if self.hypernet is not None:
            # Provisional; Net.calibrate re-scales on real states before training.
            self.rescale_heads(torch.randn(512, s_dim))

    def init_stats(self, s):
        """Std of each generated weight block on a batch of states."""
        with torch.no_grad():
            var, obs = self.hypernet(s)
            out = {}
            if var is not None:
                out["var"] = var.std().item()
            if obs is not None:
                out["obs"] = obs.std().item()
            return out

    def rescale_heads(self, states):
        """Scale the output heads so generated weights have std INIT_STD on `states`.

        The encoder is linear, so the generated scale tracks input magnitude:
        calibrate on the states the agent will actually see."""
        stats = self.init_stats(states)
        with torch.no_grad():
            if stats.get("var", 0) > 0:
                g = INIT_STD / stats["var"]
                self.hypernet.head_var.weight.mul_(g)
                self.hypernet.head_var.bias.mul_(g)
            if stats.get("obs", 0) > 0:
                g = INIT_STD / stats["obs"]
                for dec in self.hypernet.obs_decoders:
                    dec[-1].weight.mul_(g)
                    dec[-1].bias.mul_(g)


class Net(nn.Module):
    """Shared state encoder, independent policy and value QFWP-ANO, linear heads."""

    def __init__(self, arm, s_dim, a_dim, C=4, **kw):
        super().__init__()
        self.arm, self.s_dim, self.a_dim = arm, s_dim, a_dim
        self.encoder = nn.Linear(s_dim, C)
        self.pi_func = QFeature(arm, s_dim, C=C, **kw)
        self.v_func = QFeature(arm, s_dim, C=C, **kw)
        self.pi_linear = nn.Linear(self.pi_func.num_obs, a_dim)
        self.v_linear = nn.Linear(self.v_func.num_obs, 1)
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        h = self.encoder(x)
        logits = self.pi_linear(self.pi_func(x, h).to(x.dtype))
        values = self.v_linear(self.v_func(x, h).to(x.dtype))
        return logits, values

    def calibrate(self, states):
        """Re-scale both hypernetworks' heads on a (N, s_dim) batch of real
        states. Returns the achieved stats; no-op for ANO-VQC."""
        stats = {"pi": {}, "v": {}}
        for key, func in (("pi", self.pi_func), ("v", self.v_func)):
            if func.hypernet is not None:
                func.rescale_heads(states)
                stats[key] = func.init_stats(states)
        return stats

    def choose_action(self, s):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(s)
            prob = F.softmax(logits, dim=1).data
            return self.distribution(prob).sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        """Actor + critic loss, no entropy term.

        squeeze(-1) on both operands keeps td per-timestep, shape (B,); a bare
        squeeze() of values against (B, 1) targets would broadcast td to (B, B).
        """
        self.train()
        logits, values = self.forward(s)
        values = values.squeeze(-1)
        v_t = v_t.squeeze(-1)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        if a.dim() > 1:
            a = a.squeeze()
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        return (c_loss + a_loss).mean()


def param_split(net):
    """Trainable parameter counts. Quantum = circuit angles, input scaling and the
    static observables; classical = hypernetworks, encoder, actor/critic heads."""
    quantum_leaves = ("var_weights", "scale_weights", "observables")
    quantum = classical = 0
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if any(f".{leaf}" in f".{name}" for leaf in quantum_leaves):
            quantum += p.numel()
        else:
            classical += p.numel()
    return {"quantum": quantum, "classical": classical, "total": quantum + classical}
