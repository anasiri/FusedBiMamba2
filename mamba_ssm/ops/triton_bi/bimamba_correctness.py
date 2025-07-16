from src.mamba_ssm.ops.triton_bi.ssd_combined_bi import *


def check_allclose(a, b, atol=1e-1, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (f"Mean Error: {(a - b).abs().mean()}, "
                                                        f"Median Error: {(a - b).abs().median()}, "
                                                        f"Max Error: {(a - b).abs().max()}")

def print_diff(a, b, name:str):
    max_diff = round((a - b).abs().max().item(), 4)
    median_diff = round((a - b).abs().median().item(), 4)
    min_diff = round((a - b).abs().min().item(), 4)
    percentile_99 = round((a - b).abs().flatten().kthvalue(int(0.999 * a.numel()))[0].item(), 4) if a.numel() > 1 else 0
    print(f"Diff in {name}: Max: {max_diff:.4f}, 99th Percentile: {percentile_99:.4f}, "
          f"Median: {median_diff:.4f}, Min: {min_diff:.4f}")


def main_correctness():
    torch.manual_seed(42)
    torch.set_printoptions(precision=7, sci_mode=False)

    ## Dimensions
    # batch, seqlen, chunk_size, dim, headdim = 1, 96, 32, 1, 1
    batch, seqlen, chunk_size, dim, headdim = 16, 2048, 256, 2048, 64
    # batch, seqlen, chunk_size, dim, headdim = 1, 512, 32, 1, 1
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 256  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    dt = torch.nn.functional.softplus(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()

    print("Running for the Reference Implementation")
    y_out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_b_out = (mamba_chunk_scan_combined(x.flip([1]), dt.flip([1]), A, B.flip([1]), C.flip([1]), chunk_size, D=None))

    print("Running for the Triton Implementation")
    y_triton_bi = mamba_chunk_scan_combined_bi(x, dt, A, B, C, chunk_size, D=None)

    out = y_out + y_b_out.flip([1])
    check_allclose(out, y_triton_bi)
    print_diff(out, y_triton_bi, "y")
    print("Forward Tests Passed")

    # Backward tests
    torch.autograd.set_detect_anomaly(True)
    print("Running Backward Tests")
    grad_out = torch.randn_like(y_triton_bi).requires_grad_()

    # Run backward to trigger the custom backward method
    out.backward(grad_out, retain_graph=True)
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

    y_triton_bi.backward(grad_out, retain_graph=True)

    print_diff(dx_ref, x.grad, "dx")
    print_diff(dB_ref, B.grad, "dB")
    print_diff(dC_ref, C.grad, "dC")
    print_diff(dA_ref, A.grad, "dA")
    print_diff(ddt_ref, dt.grad, "ddt")


def measure_time(func, *args, warmup=True, iterations=300, verbose=True):
    # Warmup
    if warmup:
        if verbose:
            print("Warming up")
            range_iter = tqdm(range(iterations))
        else:
            range_iter = range(iterations)

        for _ in range_iter:
            func(*args)

    torch.cuda.synchronize()
    # Timing
    start_time = time.time()
    if verbose:
        print("Timing")
        range_iter = tqdm(range(iterations))
    else:
        range_iter = range(iterations)

    for _ in range_iter:
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = (end_time - start_time)  # / iterations
    return total_time


def main_time():
    torch.manual_seed(42)

    ## Dimensions
    batch, seqlen, chunk_size, dim, headdim = 16, 2048, 128, 2048, 64
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
    D = torch.randn(nheads, dtype=dtype, device=device)

    # Measuring the time for the Triton-based method with warmup
    def run_triton():
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
        y_b = mamba_chunk_scan_combined(x.flip([1]), dt.flip([1]), A, B.flip([1]), C.flip([1]), chunk_size,
                                        D=None).flip([1])
        return y + y_b

    def run_triton_fused():
        return mamba_chunk_scan_combined_bi(x, dt, A, B, C, chunk_size, D=None)

    triton_time = measure_time(run_triton, warmup=True)
    triton_fused_time = measure_time(run_triton_fused, warmup=True)
    print(f"Triton-based method time: {triton_time:.6f} seconds")
    print(f"Triton-based fused method time: {triton_fused_time:.6f} seconds")
    print(f"Speedup between Fused and Triton: {round((triton_time / triton_fused_time - 1) * 100, 2)}%")

    # Verifying the correctness
    y_triton = run_triton()
    torch.cuda.synchronize()
    y_triton_fused = run_triton_fused()
    torch.cuda.synchronize()

    print("Diff Triton Fused vs Triton:", round(torch.abs(y_triton_fused - y_triton).abs().mean().item(), 4))

    print("********************Backward Pass************************")
    # Backward pass measurement
    grad_out = torch.randn_like(y_triton).requires_grad_()

    def backward_triton():
        y_triton.backward(grad_out, retain_graph=True)

    def backward_triton_fused():
        y_triton_fused.backward(grad_out, retain_graph=True)

    # Time backward pass for Triton method
    triton_backward_time = measure_time(backward_triton, warmup=True)

    # Clear gradients before running fused version
    x.grad.zero_()
    dt.grad.zero_()
    A.grad.zero_()
    B.grad.zero_()
    C.grad.zero_()

    # Time backward pass for fused method
    triton_fused_backward_time = measure_time(backward_triton_fused, warmup=True)

    print(f"Triton-based method backward pass time: {triton_backward_time:.6f} seconds")
    print(f"Triton-based fused method backward pass time: {triton_fused_backward_time:.6f} seconds")
    print(
        f"Speedup between Fused and Triton (backward): {round((triton_backward_time / triton_fused_backward_time - 1) * 100, 2)}%")


def main_with_plot():
    torch.manual_seed(42)

    ## Dimensions
    batch, dim, headdim = 1, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"

    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    chunk_sizes = [32, 64, 128, 256]

    results = {}

    # Compute times once and store them
    for chunk_size in chunk_sizes:
        fused_times = []
        base_times = []
        fused_backward_times = []
        base_backward_times = []
        tested_seq_lengths = []
        for seqlen in sequence_lengths:
            if seqlen < chunk_size:
                continue
            x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
            dt = torch.nn.functional.softplus(
                torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
            A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
            B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
            C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
            D = torch.randn(nheads, dtype=dtype, device=device)

            grad_out = torch.randn_like(x).requires_grad_()

            def run_triton():
                y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
                y_b = mamba_chunk_scan_combined(x.flip([1]), dt.flip([1]), A, B.flip([1]), C.flip([1]), chunk_size,
                                                D=None).flip([1])
                return y + y_b

            def run_triton_fused():
                return mamba_chunk_scan_combined_bi(x, dt, A, B, C, chunk_size, D=None)

            fused_time = measure_time(run_triton_fused, warmup=True, verbose=False)
            base_time = measure_time(run_triton, warmup=True, verbose=False)

            # Perform forward pass once, store the output
            y_triton_fused = run_triton_fused()
            y_triton = run_triton()

            # Backward pass for Triton (using stored forward pass result)
            def backward_triton():
                y_triton.backward(grad_out, retain_graph=True)

            # Backward pass for Triton Fused (using stored forward pass result)
            def backward_triton_fused():
                y_triton_fused.backward(grad_out, retain_graph=True)

            # Measure backward pass (separately)
            fused_backward_time = measure_time(backward_triton_fused, warmup=True, verbose=False)
            base_backward_time = measure_time(backward_triton, warmup=True, verbose=False)

            fused_times.append(fused_time)
            base_times.append(base_time)
            fused_backward_times.append(fused_backward_time)
            base_backward_times.append(base_backward_time)
            tested_seq_lengths.append(seqlen)
            print(
                f"Chunk Size: {chunk_size}, Sequence Length: {seqlen}, Fused Time: {fused_time:.6f} s, Base Time: {base_time:.6f} s, Fused Backward Time: {fused_backward_time:.6f} s, Base Backward Time: {base_backward_time:.6f} s")
        results[chunk_size] = {
            'sequence_lengths': tested_seq_lengths,
            'fused_times': fused_times,
            'base_times': base_times,
            'fused_backward_times': fused_backward_times,
            'base_backward_times': base_backward_times,
        }

    # Set up colormap
    color_mapping = {
        32: 'blue',
        64: 'green',
        128: 'red',
        256: 'purple'
    }
    # Plot speedup (forward pass)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    for chunk_size in chunk_sizes:
        speedups = [m / f for f, m in zip(results[chunk_size]['fused_times'], results[chunk_size]['base_times'])]
        plt.plot(results[chunk_size]['sequence_lengths'], speedups, marker='o', label=f"Chunk Size {chunk_size}",
                 color=color_mapping[chunk_size])

    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup (Forward)")
    plt.title("Forward Pass Speedup between Fused and Minimal Methods")
    plt.grid(True)
    plt.legend()

    # Plot actual execution times (forward pass)
    plt.subplot(2, 2, 2)
    for chunk_size in chunk_sizes:
        plt.plot(results[chunk_size]['sequence_lengths'], results[chunk_size]['fused_times'], marker='o',
                 label=f"Fused Chunk Size {chunk_size}", color=color_mapping[chunk_size])
        plt.plot(results[chunk_size]['sequence_lengths'], results[chunk_size]['base_times'], marker='x', linestyle='--',
                 label=f"Base Chunk Size {chunk_size}", color=color_mapping[chunk_size])

    plt.xlabel("Sequence Length")
    plt.ylabel("Execution Time (Forward, seconds)")
    plt.title("Execution Time for Fused and Base Methods (Forward)")
    plt.grid(True)
    plt.legend()

    # Plot speedup (backward pass)
    plt.subplot(2, 2, 3)
    for chunk_size in chunk_sizes:
        backward_speedups = [m / f for f, m in zip(results[chunk_size]['fused_backward_times'],
                                                   results[chunk_size]['base_backward_times'])]
        plt.plot(results[chunk_size]['sequence_lengths'], backward_speedups, marker='o',
                 label=f"Chunk Size {chunk_size}",
                 color=color_mapping[chunk_size])

    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup (Backward)")
    plt.title("Backward Pass Speedup between Fused and Minimal Methods")
    plt.grid(True)
    plt.legend()

    # Plot actual execution times (backward pass)
    plt.subplot(2, 2, 4)
    for chunk_size in chunk_sizes:
        plt.plot(results[chunk_size]['sequence_lengths'], results[chunk_size]['fused_backward_times'], marker='o',
                 label=f"Fused Chunk Size {chunk_size}", color=color_mapping[chunk_size])
        plt.plot(results[chunk_size]['sequence_lengths'], results[chunk_size]['base_backward_times'], marker='x',
                 linestyle='--',
                 label=f"Base Chunk Size {chunk_size}", color=color_mapping[chunk_size])

    plt.xlabel("Sequence Length")
    plt.ylabel("Execution Time (Backward, seconds)")
    plt.title("Execution Time for Fused and Base Methods (Backward)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"speedup_plot_{dstate}.png")
    plt.show()


if __name__ == '__main__':
    main_correctness()
    # main_time()
    # main_with_plot()
