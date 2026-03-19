import torch


def symm(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def _eye_like_2x2(M: torch.Tensor) -> torch.Tensor:
    I = torch.eye(2, dtype=M.dtype, device=M.device)
    return I.expand(M.shape[:-2] + (2, 2))


def _matrix_fun_2x2(
    M: torch.Tensor,
    f,
    df,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Stable 2x2 symmetric matrix function:
        f(M) = alpha I + beta M
    using only scalar invariants, avoiding eigenvectors.
    """

    M = symm(M)

    a = M[..., 0, 0]
    b = M[..., 0, 1]
    d = M[..., 1, 1]

    tr = a + d
    disc = (a - d) ** 2 + 4.0 * b * b

    # avoid sqrt(0) in backward
    mask = disc > (eps * eps)
    gap_safe = torch.sqrt(torch.clamp(disc, min=eps * eps))

    lam1 = torch.clamp(0.5 * (tr + gap_safe), min=eps)
    lam2 = torch.clamp(0.5 * (tr - gap_safe), min=eps)
    lam = torch.clamp(0.5 * tr, min=eps)

    den = lam1 - lam2  # safe because gap_safe >= eps in distinct branch

    f1 = f(lam1)
    f2 = f(lam2)

    beta_distinct = (f1 - f2) / den
    alpha_distinct = (lam1 * f2 - lam2 * f1) / den

    # repeated-eigenvalue limit:
    # f(M) ≈ (f(lam) - lam f'(lam)) I + f'(lam) M
    beta_repeat = df(lam)
    alpha_repeat = f(lam) - lam * beta_repeat

    beta = torch.where(mask, beta_distinct, beta_repeat)
    alpha = torch.where(mask, alpha_distinct, alpha_repeat)

    I = _eye_like_2x2(M)
    return alpha[..., None, None] * I + beta[..., None, None] * M


def spd_logm(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _matrix_fun_2x2(
        M,
        f=torch.log,
        df=lambda x: 1.0 / x,
        eps=eps,
    )


def spd_sqrtm(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _matrix_fun_2x2(
        M,
        f=torch.sqrt,
        df=lambda x: 0.5 / torch.sqrt(x),
        eps=eps,
    )


def spd_invsqrtm(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _matrix_fun_2x2(
        M,
        f=lambda x: 1.0 / torch.sqrt(x),
        df=lambda x: -0.5 / (x ** 1.5),
        eps=eps,
    )


@torch.no_grad()
def polar_decomp(M: torch.Tensor, eps: float = 1e-8):
    """
    Polar decomposition M = Q P with P SPD.
    Since this is only used in diagnostics, no_grad is fine.
    """
    MtM = symm(M.transpose(-1, -2) @ M)
    MtM = MtM + eps * torch.eye(2, device=M.device, dtype=M.dtype)
    P = spd_sqrtm(MtM, eps=eps)

    # use solve instead of inv for better numerical stability
    Q = torch.linalg.solve(
        P.transpose(-1, -2),
        M.transpose(-1, -2)
    ).transpose(-1, -2)

    return Q, P
  
