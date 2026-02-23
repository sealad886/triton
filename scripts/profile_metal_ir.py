#!/usr/bin/env python3
"""Profile make_metal_ir hot paths to quantify bottlenecks."""

import cProfile
import io
import pstats
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from triton_metal_backend_compiler_perf import (
    generate_sample_llvm_ir,
    run_make_metal_ir,
)


def main():
    # Generate sample LLVM IR of various sizes
    sizes = [50, 200, 500]
    for n_lines in sizes:
        ir_text = generate_sample_llvm_ir(n_lines)
        line_count = ir_text.count("\n")
        print(f"\n{'='*60}")
        print(f"LLVM IR with ~{line_count} lines ({n_lines} instruction groups)")
        print(f"{'='*60}")

        # Warm up
        run_make_metal_ir(ir_text)

        # Time it
        times = []
        for _ in range(20):
            t0 = time.perf_counter_ns()
            run_make_metal_ir(ir_text)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)  # ms

        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[-1]
        mean = sum(times) / len(times)
        print(f"  p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  mean={mean:.2f}ms")

        # Profile with cProfile
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(10):
            run_make_metal_ir(ir_text)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    main()
