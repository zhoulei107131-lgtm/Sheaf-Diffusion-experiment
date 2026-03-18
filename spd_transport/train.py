"""Training script for SPD transport experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

from .graph import build_transports, cycle_edges
from .linalg import polar_decomp, spd_invsqrtm, spd_logm, spd_sqrtm, symm


Edge = Tuple[int, int]


def frob_sq(M: torch.Tensor) -> torch.Tensor:
    return (M * M).sum()


def log_euclid_dist_sq(H1: torch.Tensor, H2: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """d_LE(H1,H2)^2 = ||log(H1) - log(H2)||_F^2."""
    return frob_sq(spd_logm(H1, eps=eps) - spd_logm(H2, eps=eps))


def energy(H: torch.Tensor, edges: Iterable[Edge], A, eps: float = 1e-10) -> torch.Tensor:
    """Sum over undirected edges: d(H_v, A_uv H_u A_uv^T)^2."""
    total = torch.zeros((), device=H.device, dtype=H.dtype)
    for (u, v) in edges:
        Au_v = A[(u, v)]
        Hu = H[u]
        Hv = H[v]
        transported = Au_v @ Hu @ Au_v.transpose(-1, -2)
        transported = symm(transported)
        total = total + log_euclid_dist_sq(Hv, transported, eps=eps)
    return total


class SPDPerNode(nn.Module):
    """
    Minimal SPD 'network': each node has learnable parameters producing H_v in SPD(2).
    Parameterization: H = L L^T with L lower-triangular having exp() diagonals.

    L = [[exp(a), 0],
         [b,      exp(c)]]
    """

    def __init__(self, n_nodes: int, device: torch.device, clamp_val: float = 6.0):
        super().__init__()
        self.n = n_nodes
        self.clamp_val = clamp_val
        self.device = device

        # Initialize so that H_v starts near Identity:
        # exp(a)=1, exp(c)=1, b=0  =>  H = I
        self.a = nn.Parameter(torch.zeros(n_nodes, device=device))
        self.b = nn.Parameter(torch.zeros(n_nodes, device=device))
        self.c = nn.Parameter(torch.zeros(n_nodes, device=device))

    def forward(self) -> torch.Tensor:
        a = torch.clamp(self.a, -self.clamp_val, self.clamp_val)
        c = torch.clamp(self.c, -self.clamp_val, self.clamp_val)
        b = torch.clamp(self.b, -self.clamp_val, self.clamp_val)
        
        ea = torch.exp(a)
        ec = torch.exp(c)

        # Build L for each node: shape (n, 2, 2)
        L = torch.zeros((self.n, 2, 2), device=self.device, dtype=torch.get_default_dtype())
        L[:, 0, 0] = ea
        L[:, 1, 0] = b
        L[:, 1, 1] = ec

        H = L @ L.transpose(-1, -2)
        # symmetrize for safety
        return symm(H)


@dataclass
class TrainConfig:
    n: int = 4
    steps: int = 2000
    lr: float = 5e-2
    mode: str = "rotation"
    s_max: float = 0.20
    seed: int = 0
    log_every: int = 200
    eps: float = 1e-10
    device: str | None = None


@torch.no_grad()
def _edge_diagnostics(H: torch.Tensor, edges: Iterable[Edge], A, eps: float = 1e-10) -> None:
    print("\nEdge diagnostics (u->v):  ||S_uv||_F where P_uv = exp(S_uv) from polar of tildeA_uv")
    for (u, v) in edges:
        Hu_sqrt = spd_sqrtm(H[u], eps=eps)
        Hv_invsqrt = spd_invsqrtm(H[v], eps=eps)
        tildeA = Hu_sqrt @ A[(u, v)] @ Hv_invsqrt
        _, P = polar_decomp(tildeA, eps=eps)
        S = spd_logm(P, eps=eps)
        print(f"edge {u}->{v} | ||S||_F = {torch.sqrt(frob_sq(S)).item():.6e}")


def run_training(config: TrainConfig) -> float:
    torch.set_default_dtype(torch.float64)
    device = (
        torch.device(config.device)
        if config.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("torch:", torch.__version__)
    print("device:", device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    edges = cycle_edges(config.n)
    A = build_transports(config.n, edges, device=device, mode=config.mode, s_max=config.s_max)

    print("Graph edges:", edges)
    print("Example A[(0,1)]:\n", A[(0, 1)])
    print("Check A[(1,0)] * A[(0,1)] ~ I:\n", A[(1, 0)] @ A[(0, 1)])

    model = SPDPerNode(config.n, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    last_loss = 0.0
    for step in range(1, config.steps + 1):
        opt.zero_grad(set_to_none=True)
        H = model()
        loss = energy(H, edges, A, eps=config.eps)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

        if step % config.log_every == 0 or step == 1 or step == config.steps:
            with torch.no_grad():
                # basic SPD sanity: minimum eigenvalue over nodes
                mineigs = []
                for i in range(config.n):
                    lam, _ = torch.linalg.eigh(symm(H[i]))
                    mineigs.append(lam.min().item())
                print(
                    f"step {step:4d} | loss {loss.item():.6e} | min_eig(H) {min(mineigs):.3e}"
                )

    print("Training done.")
    _edge_diagnostics(model(), edges, A, eps=config.eps)
    print(
        "\nTip: set A_mode='rotation' first; loss should be ~0 quickly. "
        "Then switch to 'aniso' for nontrivial behavior."
    )
    return last_loss


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SPD transport model.")
    parser.add_argument("--n", type=int, default=4, help="Number of nodes in cycle graph.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate.")
    parser.add_argument("--mode", type=str, default="aniso", help="Transport mode.")
    parser.add_argument("--s-max", type=float, default=0.20, help="Anisotropy magnitude.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=200, help="Logging frequency.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    args = parser.parse_args()
    return TrainConfig(
        n=args.n,
        steps=args.steps,
        lr=args.lr,
        mode=args.mode,
        s_max=args.s_max,
        seed=args.seed,
        log_every=args.log_every,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
