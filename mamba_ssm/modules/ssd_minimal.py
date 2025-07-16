# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

# Simple test
def print_diff(a, b, name:str):
    max_diff = round((a - b).abs().max().item(), 4)
    median_diff = round((a - b).abs().median().item(), 4)
    min_diff = round((a - b).abs().min().item(), 4)
    percentile_99 = round((a - b).abs().flatten().kthvalue(int(0.999 * a.numel()))[0].item(), 4)
    print(f"Diff in {name}: Max: {max_diff:.4f}, 99th Percentile: {percentile_99:.4f}, "
          f"Median: {median_diff:.4f}, Min: {min_diff:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 256, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 256  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    D = torch.randn(nheads, dtype=dtype, device=device).requires_grad_()

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)
    print_diff(y, y_min, 'y')

    # backward
    grad_out = torch.randn_like(y_min).requires_grad_()
    y_min.backward(grad_out, retain_graph=True)
    dx_ref = x.grad.clone()
    ddt_ref = dt.grad.clone()
    dA_ref = A.grad.clone()
    dB_ref = B.grad.clone()
    dC_ref = C.grad.clone()

    x.grad.zero_()
    dt.grad.zero_()
    A.grad.zero_()
    B.grad.zero_()
    C.grad.zero_()

    y.backward(grad_out, retain_graph=True)
    print_diff(dx_ref, x.grad, "dx")
    print_diff(dB_ref, B.grad, "dB")
    print_diff(dC_ref, C.grad, "dC")
    print_diff(dA_ref, A.grad, "dA")
    print_diff(ddt_ref, dt.grad, "ddt")