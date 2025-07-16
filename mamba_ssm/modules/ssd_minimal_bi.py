# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
from matplotlib import pyplot as plt
# Simple test
import time
from mamba_ssm.ops.triton_bi.rev_cumsum_general import rev_cumsum_fn


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

def segsum_bi(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x_low = x.masked_fill(~mask, 0)
    x_low = torch.cumsum(x_low, dim=-2)
    # we should remove flip with reverse cumsum
    # x_up = x.flip([-2]).masked_fill(~mask, 0)
    # x_up = torch.cumsum(x_up, dim=-2).flip([-2]).flip([-1])
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=bool), diagonal=1)
    x_up = x.masked_fill(~mask, 0)
    x_up = rev_cumsum(x_up, axis=-2)

    x_segsum = x_up + x_low
    x_segsum.diagonal(dim1=-2, dim2=-1).fill_(0)
    # mask = torch.diag(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    # x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete_bi(X, A, B, C, block_len, initial_states=None):
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

    A_orig = A
    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    '''
    b = batch size
    c = number of chunks
    l = chunk length
    s = chunk length
    h = number of heads
    n = d_state
    p = d_head
    '''
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)


    # 1. Compute the output for each intra-chunk (diagonal blocks)
    segsum_A = segsum_bi(A)
    L = torch.exp(segsum_A)
    L = L + torch.eye(L.shape[-2], L.shape[-1], device=L.device, dtype=L.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # L.diagonal(dim1=-2, dim2=-1).fill_(2)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)

    # C[0, 0, 0]*(B[0, 0, 0]*1*X[0,0,0]+B[0, 0, 1]*L[0,0,0,0,1]*X[0,0,1]+B[0, 0, 2]*L[0,0,0,0,2]*X[0,0,2])
    decay_states = (A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, torch.exp(decay_states), X)

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
    right_term = torch.roll(A_cumsum, shifts=1, dims=-1)
    right_term[:, :, :, 0] = 0

    state_decay_out = torch.exp(right_term + A)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Backward ssm
    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states_b = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, torch.exp(right_term), X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states_b[:, -1:])
    states_b = torch.cat([states_b, initial_states], dim=1)
    # decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(-1, -2).contiguous()
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk.transpose(-1, -2).contiguous(), states_b)
    states_b, final_state_b = new_states[:, 1:], new_states[:, 0]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    Y_off_b = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states_b, torch.exp(A + decay_states))

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off + Y_off_b, "b c l h p -> b (c l) h p")
    return Y, final_state, final_state_b

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
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 256, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 64  # (N) in the paper
    # dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    dt = torch.nn.functional.softplus(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    D = torch.randn(nheads, dtype=dtype, device=device).requires_grad_()

    y_ref_bi, _, _ = ssd_minimal_discrete_bi(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)


    y_min, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
    y_min_b = ssd_minimal_discrete(x.flip([1]) * dt.flip([1]).unsqueeze(-1), A * dt.flip([1]), B.flip([1]), C.flip([1]), chunk_size)[0].flip([1])
    y_ref = y_min + y_min_b

    print_diff(y_ref_bi, y_ref, 'y_minimal')
    # backward
    grad_out = torch.randn_like(y_ref).requires_grad_()
    y_ref.backward(grad_out, retain_graph=True)
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

    y_ref_bi.backward(grad_out, retain_graph=True)
    print_diff(dx_ref, x.grad, "dx")
    print_diff(dB_ref, B.grad, "dB")
    print_diff(dC_ref, C.grad, "dC")
    print_diff(dA_ref, A.grad, "dA")
    print_diff(ddt_ref, dt.grad, "ddt")