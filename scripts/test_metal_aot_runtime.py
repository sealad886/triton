#!/usr/bin/env python3
"""
Metal AOT Runtime Integration Test

Compiles a simple Triton vector_add kernel to .metallib using the Metal
backend, then invokes the Objective-C AOT harness to load and run it on
a real Metal GPU.

Requires: macOS, Xcode CLI tools, MPS-capable GPU.

Usage:
    python scripts/test_metal_aot_runtime.py
    python scripts/test_metal_aot_runtime.py --num-elements 2048
    python scripts/test_metal_aot_runtime.py --json -o report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HARNESS_SRC = (
    Path(__file__).resolve().parent.parent
    / "third_party"
    / "metal"
    / "tools"
    / "metal"
    / "test_aot_runtime.m"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compile_harness(harness_src: Path, output_bin: Path) -> tuple[bool, str]:
    """Compile the ObjC AOT harness with clang."""
    cmd = [
        "clang",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        "-framework",
        "CoreGraphics",
        "-o",
        str(output_bin),
        str(harness_src),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"clang failed: {result.stderr.strip()}"
        return True, "OK"
    except FileNotFoundError:
        return False, "clang not found"
    except subprocess.TimeoutExpired:
        return False, "clang timed out"


def _compile_triton_kernel_to_metallib(
    work_dir: Path,
) -> tuple[bool, Path | None, dict[str, int | str] | str]:
    """Compile a vector_add kernel through Triton JIT → MSL → metallib."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import triton
        import triton.compiler
        import triton.language as tl
        from triton.compiler.compiler import GPUTarget

        block_size = 256

        @triton.jit
        def vector_add_kernel(
            x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=vector_add_kernel,
            signature={
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n_elements": "i32",
            },
            constexprs={"BLOCK_SIZE": block_size},
        )
        target = GPUTarget("metal", "apple8", 32)
        compiled = triton.compile(src=src, target=target)

        metallib_data = compiled.asm.get("metallib")
        if not metallib_data or metallib_data[:4] != b"MTLB":
            return False, None, "Compilation succeeded but no valid metallib produced"

        metallib_path = work_dir / "vector_add.metallib"
        metallib_path.write_bytes(metallib_data)

        # Extract actual kernel name from MSL source
        kernel_name = "vector_add_kernel"
        msl_src = compiled.asm.get("metal", b"")
        if isinstance(msl_src, bytes):
            msl_src = msl_src.decode("utf-8", errors="replace")
        for line in msl_src.splitlines():
            if "kernel void" in line:
                parts = line.split("kernel void")[1].strip().split("(")
                if parts:
                    kernel_name = parts[0].strip()
                    break

        return True, metallib_path, {
            "kernel_name": kernel_name,
            "block_size": block_size,
            "threads_per_threadgroup": int(compiled.metadata.num_warps * target.warp_size),
        }
    except Exception as exc:
        return False, None, str(exc)


def _run_harness(
    harness_bin: Path,
    metallib_path: Path,
    kernel_name: str,
    num_elements: int,
    block_size: int,
    threads_per_threadgroup: int,
) -> tuple[bool, str]:
    """Run the compiled AOT harness."""
    cmd = [
        str(harness_bin),
        str(metallib_path),
        kernel_name,
        str(num_elements),
        str(block_size),
        str(threads_per_threadgroup),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and "RESULT: PASS" in output
        return passed, output.strip()
    except subprocess.TimeoutExpired:
        return False, "harness timed out"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Metal AOT runtime integration test")
    parser.add_argument(
        "--num-elements", type=int, default=1024, help="Number of elements"
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="vector_add_kernel",
        help="Kernel function name",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Write JSON to file"
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("SKIP: Metal AOT runtime requires macOS")
        return 0

    report: dict = {
        "ts_utc": _utc_now_iso(),
        "steps": {},
        "overall_passed": False,
    }

    with tempfile.TemporaryDirectory(prefix="metal-aot-runtime-") as tmpdir:
        work_dir = Path(tmpdir)

        # Step 1: Compile ObjC harness
        harness_bin = work_dir / "test_aot_runtime"
        ok, detail = _compile_harness(HARNESS_SRC, harness_bin)
        report["steps"]["compile_harness"] = {"passed": ok, "detail": detail}
        if not ok:
            print(f"FAIL: compile harness — {detail}")
            _output_report(report, args)
            return 1
        print(f"[1/3] Compiled AOT harness: {detail}")

        # Step 2: Compile Triton kernel → metallib
        ok, metallib_path, launch_info_or_err = _compile_triton_kernel_to_metallib(
            work_dir
        )
        report["steps"]["compile_kernel"] = {"passed": ok, "detail": launch_info_or_err}
        if not ok or metallib_path is None:
            print(f"FAIL: compile kernel — {launch_info_or_err}")
            _output_report(report, args)
            return 1
        assert isinstance(launch_info_or_err, dict)
        print(
            "[2/3] Compiled kernel to metallib: "
            f"{metallib_path.stat().st_size} bytes "
            f"(kernel={launch_info_or_err['kernel_name']}, "
            f"block_size={launch_info_or_err['block_size']}, "
            f"threads_per_tg={launch_info_or_err['threads_per_threadgroup']})"
        )

        # Step 3: Run harness using the Triton launch geometry required by the
        # generated kernel rather than generic Metal defaults.
        ok, output = _run_harness(
            harness_bin,
            metallib_path,
            str(launch_info_or_err["kernel_name"]),
            args.num_elements,
            int(launch_info_or_err["block_size"]),
            int(launch_info_or_err["threads_per_threadgroup"]),
        )
        report["steps"]["run_harness"] = {"passed": ok, "detail": output}
        report["overall_passed"] = ok
        if ok:
            print(f"[3/3] AOT runtime: PASS")
        else:
            print(f"[3/3] AOT runtime: FAIL")
            print(output)

        _output_report(report, args)
        return 0 if ok else 1


def _output_report(report: dict, args: argparse.Namespace) -> None:
    if args.json or args.output:
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    raise SystemExit(main())
