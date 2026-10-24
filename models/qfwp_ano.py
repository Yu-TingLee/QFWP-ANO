"""QFWP-ANO: a classical hypernetwork programs a data re-uploading VQC measured
with adaptive non-local observables (ANO), conditioned on each input.

The hypernetwork programs the rotation angles, the observables, or both:

                       static observables   programmed observables
    static angles      ANO-VQC              QFWP-ANO-O
    programmed angles  QFWP-ANO-R           QFWP-ANO-RO
"""

import itertools

import pennylane as qml
import torch
from torch import nn

from models.revin import RevIN

# variant -> (programs the rotation angles, programs the observables)
VARIANTS = {
    "ANO-VQC": (False, False),
    "QFWP-ANO-R": (True, False),
    "QFWP-ANO-O": (False, True),
    "QFWP-ANO-RO": (True, True),
}

INIT_STD = 0.1


def assemble_hermitians(diag, upper_real, upper_imag, iu_row, iu_col):
    """Hermitian matrices from their free parameters, over any leading batch dims.

    diag: (..., K); upper_real, upper_imag: (..., K(K-1)/2) -> (..., K, K) complex.
    """
    K = diag.shape[-1]
    real = torch.zeros(*diag.shape[:-1], K, K, device=diag.device, dtype=diag.dtype)
    imag = torch.zeros_like(real)
    real[..., iu_row, iu_col] = upper_real
    imag[..., iu_row, iu_col] = upper_imag
    real_part = real + real.transpose(-1, -2) + torch.diag_embed(diag)
    imag_part = imag - imag.transpose(-1, -2)
    return torch.complex(real_part, imag_part)


class ANOHermitians(nn.Module):
    """Bank of M trainable K x K Hermitian observables, stored as their free parameters."""

    def __init__(self, M, K, init_scale=INIT_STD):
        super().__init__()
        iu = torch.triu_indices(K, K, offset=1)
        self.register_buffer("iu_row", iu[0])
        self.register_buffer("iu_col", iu[1])
        num_upper = iu.shape[1]
        self.diag = nn.Parameter(torch.randn(M, K) * init_scale)
        self.upper_real = nn.Parameter(torch.randn(M, num_upper) * init_scale)
        self.upper_imag = nn.Parameter(torch.randn(M, num_upper) * init_scale)

    def assemble(self):
        return assemble_hermitians(
            self.diag, self.upper_real, self.upper_imag, self.iu_row, self.iu_col
        )  # (M, K, K)


class VQC(nn.Module):
    """Data re-uploading VQC with one k-local observable per k-subset of qubits.

    After a Hadamard layer, each of the D layers applies RZ, RY rotations, a CNOT
    ring, and RY, RZ encoding of the input. The rotation angles and the observables
    are either trainable parameters or passed in per sample by a hypernetwork.
    """

    def __init__(
        self,
        num_qubits,
        num_layers,
        n_local,
        program_angles=False,
        program_observables=False,
        learnable_scaling=False,
        qml_device="default.qubit",
    ):
        super().__init__()
        assert 1 <= n_local <= num_qubits, f"n_local must be in [1, {num_qubits}]"
        self.C, self.D = num_qubits, num_layers
        self.program_angles = program_angles
        self.program_observables = program_observables

        self.n_var = 2  # RZ, RY angle per qubit and layer
        self.var_dim = self.D * self.C * self.n_var
        if not program_angles:
            self.var_weights = nn.Parameter(
                torch.randn(self.D, self.C, self.n_var) * INIT_STD
            )
        # Input scaling w on the encoding gates, R_y(w x); trainable only when enabled.
        if learnable_scaling:
            self.scale_weights = nn.Parameter(torch.ones(self.D, self.C, 2))
        else:
            self.register_buffer("scale_weights", torch.ones(self.D, self.C, 2))

        self.subsets = list(itertools.combinations(range(self.C), n_local))
        self.num_obs = len(self.subsets)
        self.K_obs = 2**n_local
        self.num_upper = self.K_obs * (self.K_obs - 1) // 2
        self.per_obs = (
            self.K_obs + 2 * self.num_upper
        )  # free parameters of one observable
        if program_observables:
            iu = torch.triu_indices(self.K_obs, self.K_obs, offset=1)
            self.register_buffer("iu_row", iu[0])
            self.register_buffer("iu_col", iu[1])
        else:
            self.observables = ANOHermitians(self.num_obs, self.K_obs)

        dev = qml.device(qml_device, wires=self.C)
        C, D, subsets = self.C, self.D, self.subsets

        def circuit(inputs, var_w, scale_w):
            # var_w is (D, C, 2) when static and (B, D, C, 2) when programmed;
            # default.qubit broadcasts over the batch dim either way.
            for q in range(C):
                qml.Hadamard(wires=q)
            for d in range(D):
                for q in range(C):
                    qml.RZ(var_w[..., d, q, 0], wires=q)
                    qml.RY(var_w[..., d, q, 1], wires=q)
                for q in range(C):
                    qml.CNOT(wires=[q, (q + 1) % C])
                for q in range(C):
                    qml.RY(scale_w[d, q, 0] * inputs[..., q], wires=q)
                    qml.RZ(scale_w[d, q, 1] * inputs[..., q], wires=q)

        if program_observables:
            # One qml.Hermitian cannot differ across the batch, so read out each
            # subset's reduced density matrix and trace it against the sample's own
            # observable in forward().
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qnode(inputs, var_w, scale_w):
                circuit(inputs, var_w, scale_w)
                return [qml.density_matrix(wires=list(S)) for S in subsets]
        else:

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qnode(inputs, var_w, scale_w, herms):
                circuit(inputs, var_w, scale_w)
                return [
                    qml.expval(qml.Hermitian(herms[m], wires=list(S)))
                    for m, S in enumerate(subsets)
                ]

        self.qnode = qnode

    def forward(self, h, var_weights=None, obs_params=None):
        """h: (B, C). var_weights: (B, D, C, 2) programmed angles. obs_params:
        (B, M, per_obs) programmed observables. Returns (B, M) expectation values."""
        var_w = var_weights if self.program_angles else self.var_weights
        if not self.program_observables:
            herms = self.observables.assemble()
            return torch.stack(self.qnode(h, var_w, self.scale_weights, herms), dim=-1)

        K, nu = self.K_obs, self.num_upper
        herms = assemble_hermitians(
            obs_params[..., :K],
            obs_params[..., K : K + nu],
            obs_params[..., K + nu :],
            self.iu_row,
            self.iu_col,
        )
        rhos = self.qnode(h, var_w, self.scale_weights)  # M reduced DMs, (B, K, K) each
        herms = herms.to(rhos[0].dtype)
        exps = [
            torch.einsum("bij,bji->b", rhos[m], herms[:, m]).real
            for m in range(self.num_obs)
        ]
        return torch.stack(exps, dim=-1)


class Hypernetwork(nn.Module):
    """Shared encoder z = ReLU(W s + b), a linear decoder for the rotation angles,
    and one two-layer MLP decoder per observable."""

    def __init__(self, in_dim, hidden_dim, var_dim=0, num_obs=0, per_obs=0, d_dec=64):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.head_var = nn.Linear(hidden_dim, var_dim) if var_dim else None
        self.obs_decoders = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, d_dec),
                        nn.ReLU(),
                        nn.Linear(d_dec, per_obs),
                    )
                    for _ in range(num_obs)
                ]
            )
            if num_obs
            else None
        )

    def forward(self, s):
        """s: (B, in_dim) -> angles (B, var_dim) and observables (B, M, per_obs), or None."""
        z = self.act(self.encoder(s))
        var = self.head_var(z) if self.head_var is not None else None
        obs = (
            torch.stack([dec(z) for dec in self.obs_decoders], dim=1)
            if self.obs_decoders is not None
            else None
        )
        return var, obs


class QFWPANO(nn.Module):
    """One variant from VARIANTS: the VQC plus, unless static, its hypernetwork.

    forward(s, h): the hypernetwork reads `s` and the circuit encodes `h`.
    """

    def __init__(
        self,
        variant,
        s_dim,
        num_qubits,
        num_layers,
        n_local,
        d_hidden=64,
        d_dec=64,
        learnable_scaling=False,
        qml_device="default.qubit",
    ):
        super().__init__()
        program_angles, program_observables = VARIANTS[variant]
        self.variant = variant
        self.vqc = VQC(
            num_qubits,
            num_layers,
            n_local,
            program_angles,
            program_observables,
            learnable_scaling,
            qml_device,
        )
        self.num_obs = self.vqc.num_obs
        self.hypernet = None
        if program_angles or program_observables:
            self.hypernet = Hypernetwork(
                s_dim,
                d_hidden,
                var_dim=self.vqc.var_dim if program_angles else 0,
                num_obs=self.vqc.num_obs if program_observables else 0,
                per_obs=self.vqc.per_obs,
                d_dec=d_dec,
            )

    def forward(self, s, h):
        if self.hypernet is None:
            return self.vqc(h)
        var, obs = self.hypernet(s)
        if var is not None:
            var = var.reshape(-1, self.vqc.D, self.vqc.C, self.vqc.n_var)
        return self.vqc(h, var, obs)


class Model(nn.Module):
    """QFWP-ANO forecaster (configs.model picks the variant).

    RevIN -> flattened window -> linear projection h -> QFWP-ANO, whose
    hypernetwork reads the window -> linear head -> de-normalization.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        assert configs.num_qubits == self.C, "QFWP-ANO requires num_qubits == enc_in"

        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None
        self.input_proj = nn.Linear(self.C * self.seq_len, self.C)
        self.qfwp = QFWPANO(
            configs.model,
            self.C * self.seq_len,
            self.C,
            configs.num_layers,
            configs.n_local,
            configs.d_hidden,
            configs.d_dec,
            qml_device=configs.QML_device,
        )
        self.head = nn.Linear(self.qfwp.num_obs, self.C * self.pred_len)

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        x = x.transpose(1, 2)  # (B, C, L)
        x_norm = self.revin.normalize(x) if self.use_revin else x
        flat = x_norm.reshape(B, self.C * self.seq_len)
        h = self.input_proj(flat)  # (B, C)
        q_out = self.qfwp(flat, h).to(x.dtype)  # (B, M)
        y = self.head(q_out).reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
