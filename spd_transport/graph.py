"""Graph utilities for SPD transport experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch


Edge = Tuple[int, int]


def cycle_edges(n: int) -> List[Edge]:
    """Undirected edges of C_n: (0,1),(1,2),...,(n-1,0)."""
    return [(i, (i + 1) % n) for i in range(n)]


def rot2(theta: torch.Tensor) -> torch.Tensor:
    """2x2 rotation matrix."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )


@torch.no_grad()
def build_transports(
    n: int,
    edges: Iterable[Edge],
    device: torch.device,
    mode: str = "rotation",
    s_max: float = 0.15,
) -> Dict[Edge, torch.Tensor]:
    """
    Build a dict A[(u,v)] for directed edges, with A[(v,u)] = inv(A[(u,v)]).

    mode:
      - "rotation": A_uv = R(theta)
      - "aniso":    A_uv = R(theta) @ diag(exp(s), exp(-s))
    """
    if n <= 0:
        raise ValueError("n must be positive")

    A: Dict[Edge, torch.Tensor] = {}
    for (u, v) in edges:
        theta = (2 * math.pi) * torch.rand((), device=device)
        R = rot2(theta).to(device)

        if mode == "rotation":
            M = R
        elif mode == "aniso":
            s = (2 * s_max) * (torch.rand((), device=device) - 0.5)  # in [-s_max, s_max]
            D = torch.diag(torch.tensor([torch.exp(s), torch.exp(-s)], device=device))
            M = R @ D
        else:
            raise ValueError("Unknown mode")

        A[(u, v)] = M
        A[(v, u)] = torch.linalg.inv(M)

    return A
