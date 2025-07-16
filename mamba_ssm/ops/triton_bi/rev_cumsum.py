import math
import time

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl


# This reverse cumsum is used for tensors of shape batch, nheads, nchunks, chunk_size only on the last dimension.

def rev_cumsum_ref(x):
    return torch.flip(torch.cumsum(torch.flip(x, [-1]), -1), [-1])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size'],
)
@triton.jit
def _rev_cumsum_kernel_fwd(
        x_ptr, y_ptr, chunk_size, seqlen,
        nheads, nheads_per_program, ngroups,
        stride_x_batch, stride_x_head, stride_x_chunk,
        stride_y_batch, stride_y_head, stride_y_chunk,
        BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    # pid_sg = tl.program_id(2)
    # pid_s = pid_sg // ngroups
    # pid_g = pid_sg - pid_s * ngroups

    # x_ptr += pid_b * stride_x_batch + pid_c * stride_x_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    x_ptr += pid_b * stride_x_batch + pid_c * stride_x_chunk + pid_h * stride_x_head
    y_ptr += pid_b * stride_y_batch + pid_c * stride_y_chunk + pid_h * stride_y_head

    num_blocks = (chunk_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    x_ptr += BLOCK_SIZE * (num_blocks - 1)
    y_ptr += BLOCK_SIZE * (num_blocks - 1)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    offset = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offset
    y_ptrs = y_ptr + offset

    carry = 0.0

    high, low = (num_blocks - 1) * BLOCK_SIZE, 0
    # for start in reversed(range(high, low - 1, BLOCK_SIZE)):
    for start in range(high, - 1, -BLOCK_SIZE):
        start = tl.multiple_of(start, BLOCK_SIZE)
        x = tl.load(x_ptrs, mask=(offset < chunk_size_limit - start))
        cumsum = tl.cumsum(x, axis=0, reverse=True) + carry
        tl.store(y_ptrs, cumsum, mask=(offset < chunk_size_limit - start))

        carry += tl.sum(x, axis=0)

        x_ptrs -= BLOCK_SIZE
        y_ptrs -= BLOCK_SIZE

def rev_cumsum_fwd(x: torch.Tensor, ngroups=1):
    batch, nheads, nchunks, chunk_size = x.shape
    assert chunk_size>=32, "Triton 3.0.0 has a bug with reverse cumsum when the length of the axis is < 32"

    if x.stride(-1) != 1:
        x = x.contiguous()
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    seqlen = chunk_size * nchunks

    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)

    # grid = lambda META: (tl.cdiv(chunk_size, META['BLOCK_SIZE']), batch * nchunks, nsplits * ngroups)
    grid = lambda META: (batch, nchunks, nheads)

    stride_x_batch, stride_x_nheads, stride_x_nchunks, stride_x_chunk = x.stride()
    stride_y_batch, stride_y_nheads, stride_y_nchunks, stride_y_chunk = output.stride()

    _rev_cumsum_kernel_fwd[grid](
        x, output, chunk_size, seqlen, nheads, nheads_per_program, ngroups,
        stride_x_batch, stride_x_nheads, stride_x_nchunks,
        stride_y_batch, stride_y_nheads, stride_y_nchunks,
    )

    return output


class RevCumSumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return rev_cumsum_fwd(x)  # torch.flip(torch.cumsum(torch.flip(x, [axis]), axis), [axis])

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.cumsum(grad_output, dim=-1)
        return grad_input, None  # No gradient required for axis


def rev_cumsum(x):
    return RevCumSumFn.apply(x)


def main_correctness():
    torch.manual_seed(42)
    # Testing with double precision to match gradcheck's requirements
    # Triton 3.0.0 has a bug with reverse cumsum when the length of the axis is < 32
    # Check https://github.com/triton-lang/triton/issues/4362
    batch, nheads, nchunks, chunk_size = 16, 32, 8, 256
    # batch, nheads, nchunks, chunk_size = 1, 1, 1, 256
    x = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float32, device="cuda").requires_grad_()

    # Test forward pass
    y = rev_cumsum(x)
    y_ref = rev_cumsum_ref(x)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4)
    print("Forward pass tests pass!")

    y_grad = torch.randn_like(y)

    # Test backward pass
    y.backward(y_grad)
    x_grad = x.grad

    x.grad.zero_()

    y_ref.backward(y_grad)
    x_ref_grad = x.grad

    assert torch.allclose(x_grad, x_ref_grad, atol=1e-4, rtol=1e-4)
    print("Backward pass tests pass!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['chunk_size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2 ** i for i in range(5, 10)],  # Different possible values for `x_name`.
        x_log=True,  # x-axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='Time (ms)',  # Label name for the y-axis.
        plot_name='rev-cumsum-performance',  # Name for the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(chunk_size, provider):
    batch, nheads, nchunks = 64, 3, 4
    x = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float32, device="cuda")
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rev_cumsum_ref(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rev_cumsum_fwd(x), quantiles=quantiles)
    return ms, min_ms, max_ms


def run_benchmark():
    benchmark.run(print_data=True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # main_correctness()
    run_benchmark()
