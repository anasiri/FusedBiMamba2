import math
import time

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl


def rev_cumsum_ref(x, axis=0):
    return torch.flip(torch.cumsum(torch.flip(x, [axis]), axis), [axis])

@triton.jit
def _rev_cumsum_kernel_fwd(X,  # pointer to the input tensor
                           Y,  # pointer to the output tensor
                           N,  # Number of elements along the specified axis
                           BLOCK_SIZE: tl.constexpr
                           ):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    offset = pid * N + offset
    X_block = tl.load(X + offset, mask=mask, other=0.0)
    rev_cumsum_block = tl.cumsum(X_block, axis=0, reverse=True)
    tl.store(Y + offset, rev_cumsum_block, mask=mask)


def rev_cumsum_fwd(x: torch.Tensor, axis: int):
    if axis < 0:
        axis += x.ndim
    shape = x.shape
    AXIS_SIZE = shape[axis]

    # This is a workaround for the tl.cumsum(reverse=True) bug.
    if AXIS_SIZE < 32:
        return rev_cumsum_ref(x, axis)
    perm = list(range(x.ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]

    # contiguous is used to make sure the stride is 1 for the last dimension
    x_permuted = x.permute(*perm)
    shape_x_perm = x_permuted.shape
    x_permuted = x_permuted.reshape(-1, AXIS_SIZE).contiguous()
    y_permuted = torch.empty_like(x_permuted)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(AXIS_SIZE))
    if AXIS_SIZE > BLOCK_SIZE:
        raise RuntimeError("This revcumsum doesn't support feature dim >= 64KB.")

    numel = x_permuted.numel()
    grid = (math.ceil(numel // AXIS_SIZE),)
    _rev_cumsum_kernel_fwd[grid](x_permuted, y_permuted, AXIS_SIZE, BLOCK_SIZE)
    y = y_permuted.reshape(shape_x_perm).permute(*perm)
    return y


class RevCumSumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return rev_cumsum_fwd(x, axis)  # torch.flip(torch.cumsum(torch.flip(x, [axis]), axis), [axis])

    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis
        grad_input = torch.cumsum(grad_output, axis)

        return grad_input, None  # No gradient required for axis


def rev_cumsum_fn(x, axis):
    return RevCumSumFn.apply(x, axis)


def main_correctness():
    torch.manual_seed(42)
    # Testing with double precision to match gradcheck's requirements
    # Triton 3.0.0 has a bug with reverse cumsum when the length of the axis is < 32
    # Check https://github.com/triton-lang/triton/issues/4362
    x = torch.randn([2, 1024], dtype=torch.double, requires_grad=True, device="cuda").double()
    axis = 1
    # Test forward pass
    y = rev_cumsum_fn(x, axis)
    y_ref = torch.flip(torch.cumsum(torch.flip(x, [axis]), axis), [axis])
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4)
    print("Forward pass tests pass!")
    # Test backward pass with gradcheck
    gradcheck_passed = torch.autograd.gradcheck(rev_cumsum_fn, (x, axis), rtol=1e-2, atol=1e-2)
    print(f"Gradcheck passed: {gradcheck_passed}")

    # Test second-order gradients with gradgradcheck
    gradgradcheck_passed = torch.autograd.gradgradcheck(rev_cumsum_fn, (x, axis), rtol=1e-4, atol=1e-4)
    print(f"Gradgradcheck passed: {gradgradcheck_passed}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['feature_size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2 ** i for i in range(1, 15)],  # Different possible values for `x_name`.
        x_log=True,  # x-axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='Time (ms)',  # Label name for the y-axis.
        plot_name='rev-cumsum-performance',  # Name for the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(feature_size, provider):
    BATCH_SIZE = 64
    x = torch.rand((BATCH_SIZE, feature_size), dtype=torch.float32, device='cuda')
    axis = 1  # Assuming we are benchmarking along axis 0
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rev_cumsum_ref(x, axis), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rev_cumsum_fwd(x, axis), quantiles=quantiles)

    return ms, min_ms, max_ms


def run_benchmark():
    benchmark.run(print_data=True)
    plt.axvline(x=32, color='r', linestyle='--', label="x = 32")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_correctness()
    # run_benchmark()