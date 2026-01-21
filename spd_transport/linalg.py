"""Linear algebra helpers for SPD matrices."""

from __future__ import annotations

import torch


def symm(M: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a matrix/tensor."""
    return 0.5 * (M + M.transpose(-1, -2))


def spd_eigh(M: torch.Tensor, eps: float = 1e-10):
    """
    Eigen-decomp for symmetric matrices with eigenvalue clipping to keep SPD.
    Returns (Q, lam) where lam is clipped to >= eps.
    """
    M = symm(M)
    lam, Q = torch.linalg.eigh(M)  # lam ascending
    lam = torch.clamp(lam, min=eps)
    return Q, lam


def spd_logm(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    Q, lam = spd_eigh(M, eps=eps)
    return Q @ torch.diag_embed(torch.log(lam)) @ Q.transpose(-1, -2)


def spd_sqrtm(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    Q, lam = spd_eigh(M, eps=eps)
    return Q @ torch.diag_embed(torch.sqrt(lam)) @ Q.transpose(-1, -2)


def spd_invsqrtm(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    Q, lam = spd_eigh(M, eps=eps)
    return Q @ torch.diag_embed(1.0 / torch.sqrt(lam)) @ Q.transpose(-1, -2)


@torch.no_grad()
def polar_decomp(M: torch.Tensor, eps: float = 1e-10):
    """
    Polar decomposition M = Q P
    where P = sqrt(M^T M) is SPD, Q = M P^{-1} is orthogonal (in exact arithmetic).
    """
    MtM = symm(M.transpose(-1, -2) @ M)
    MtM = MtM + eps * torch.eye(2, device=M.device, dtype=M.dtype)
    P = spd_sqrtm(MtM, eps=eps)
    Pinv = torch.linalg.inv(P)
    Q = M @ Pinv
    return Q, P
