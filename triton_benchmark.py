import matplotlib.pyplot as plt

from mamba_ssm.ops.triton_bi.ssd_combined_bi import mamba_chunk_scan_combined_bi
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import triton
import torch
chunk_size = 128
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seqlen'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7, 12)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'triton_fused', 'triton (uni)'],  # possible values for `line_arg`
        line_names=[
            "Triton",
            "Triton Fused",
            "Triton (Uni)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="mamba-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'chunk_size': chunk_size},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(seqlen, chunk_size, provider):
    # Generate random inputs based on the dimensions of your mamba function
    batch, dim, headdim = 16, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 64  # (N) in the paper

    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    dt = torch.nn.functional.softplus(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()

    grad_out = torch.randn_like(x)


    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # Choose the correct method based on provider
    if provider == 'triton':
        # Measure forward + backward pass inside the lambda function
        ms = triton.testing.do_bench(
            lambda: (
                mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None) +
                mamba_chunk_scan_combined(x.flip([1]), dt.flip([1]), A, B.flip([1]), C.flip([1]), chunk_size, D=None).flip([1])
            ).backward(grad_out)
        )
    if provider == 'triton (uni)':
        # Measure forward + backward pass inside the lambda function
        ms = triton.testing.do_bench(
            lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None).backward(grad_out)
        )

    if provider == 'triton_fused':
        # Measure forward + backward pass inside the lambda function
        ms = triton.testing.do_bench(
            lambda: mamba_chunk_scan_combined_bi(x, dt, A, B, C, chunk_size, D=None).backward(grad_out)
        )

    # Calculate GB/s based on the data size and elapsed time
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=False, print_data=True)

plt.title("Different Implementations of Mamba based on chunk size {}".format(chunk_size))
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()