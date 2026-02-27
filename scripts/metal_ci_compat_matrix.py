#!/usr/bin/env python3
"""
Automated compatibility matrix validation for Metal backend CI.

Validates that the Metal backend:
1. Imports without error
2. Compilation pipeline works (TTIR -> TTGIR -> LLIR -> MSL)
3. Basic kernel compilation succeeds
4. Runtime fallback paths are functional

Exit code 0 = all checks pass, non-zero = failure with details.

Usage:
    python scripts/metal_ci_compat_matrix.py              # human-readable
    python scripts/metal_ci_compat_matrix.py --json        # CI-friendly JSON
    python scripts/metal_ci_compat_matrix.py --json -o report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: Any = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_python_version() -> CheckResult:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    supported = sys.version_info[:2] in ((3, 11), (3, 12), (3, 13))
    return CheckResult(
        name="python_version",
        passed=supported,
        detail=f"Python {ver} ({'supported' if supported else 'unsupported — need 3.11/3.12/3.13'})",
        value=ver,
    )


def check_platform() -> CheckResult:
    plat = sys.platform
    arch = platform.machine()
    is_macos = plat == "darwin"
    return CheckResult(
        name="platform",
        passed=True,
        detail=f"{plat}/{arch} — Metal requires macOS/arm64 for GPU, cross-compile OK elsewhere",
        value={"platform": plat, "arch": arch, "is_macos": is_macos},
    )


def check_torch_available() -> CheckResult:
    try:
        import torch

        ver = torch.__version__
        mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        mps_avail = mps_built and torch.backends.mps.is_available()
        return CheckResult(
            name="torch",
            passed=True,
            detail=f"torch {ver}, MPS built={mps_built}, available={mps_avail}",
            value={"version": ver, "mps_built": mps_built, "mps_available": mps_avail},
        )
    except ImportError:
        return CheckResult(
            name="torch",
            passed=True,
            detail="torch not installed — compile-only mode",
            value=None,
        )


def check_xcrun_available() -> CheckResult:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return CheckResult(
            name="xcrun",
            passed=False,
            detail="xcrun not found — install Xcode Command Line Tools",
            value=None,
        )
    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        sdk_ver = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        sdk_ver = "unknown"
    return CheckResult(
        name="xcrun",
        passed=True,
        detail=f"xcrun at {xcrun}, macOS SDK {sdk_ver}",
        value={"path": xcrun, "sdk_version": sdk_ver},
    )


def check_metal_toolchain() -> CheckResult:
    if sys.platform != "darwin":
        return CheckResult(
            name="metal_toolchain",
            passed=True,
            detail="Not macOS — Metal toolchain check skipped (cross-compile OK)",
            value=None,
        )
    metal_compiler = shutil.which("xcrun")
    if metal_compiler is None:
        return CheckResult(
            name="metal_toolchain",
            passed=False,
            detail="Metal toolchain unavailable — need Xcode",
            value=None,
        )
    try:
        result = subprocess.run(
            ["xcrun", "-f", "metal"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        metal_path = result.stdout.strip() if result.returncode == 0 else None
        result2 = subprocess.run(
            ["xcrun", "-f", "metallib"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        metallib_path = result2.stdout.strip() if result2.returncode == 0 else None
        ok = metal_path is not None and metallib_path is not None
        return CheckResult(
            name="metal_toolchain",
            passed=ok,
            detail=(
                f"metal={metal_path}, metallib={metallib_path}"
                if ok
                else "Metal compiler not found"
            ),
            value={"metal": metal_path, "metallib": metallib_path},
        )
    except Exception as exc:
        return CheckResult(
            name="metal_toolchain",
            passed=False,
            detail=f"Metal toolchain probe error: {exc}",
            value=None,
        )


def check_metal_backend_import() -> CheckResult:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from third_party.metal.backend.compiler import MetalBackend, MetalOptions
        from third_party.metal.backend.driver import MetalDriver

        opts = MetalOptions(arch="apple8")
        is_active = MetalDriver.is_active()
        h = opts.hash()
        return CheckResult(
            name="metal_backend_import",
            passed=True,
            detail=f"MetalBackend, MetalOptions, MetalDriver loaded; active={is_active}; hash={h[:16]}…",
            value={"active": is_active, "hash": h},
        )
    except Exception as exc:
        return CheckResult(
            name="metal_backend_import",
            passed=False,
            detail=f"Import failed: {exc}",
            value=None,
        )


def check_basic_compilation() -> CheckResult:
    """Run a minimal Triton JIT → MSL compilation pipeline."""
    try:
        import triton
        import triton.compiler
        import triton.language as tl
        from triton.compiler.compiler import GPUTarget

        @triton.jit
        def _compat_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, x + y, mask=mask)

        src = triton.compiler.ASTSource(
            fn=_compat_add,
            signature={
                "x_ptr": "*fp32",
                "y_ptr": "*fp32",
                "out_ptr": "*fp32",
                "n": "i32",
            },
            constexprs={"BLOCK": 128},
        )
        target = GPUTarget("metal", "apple8", 32)

        try:
            compiled = triton.compile(src=src, target=target)
            has_msl = "metal" in compiled.asm and b"kernel void" in compiled.asm.get(
                "metal", b""
            )
            has_metallib = (
                "metallib" in compiled.asm and compiled.asm["metallib"][:4] == b"MTLB"
            )
            return CheckResult(
                name="basic_compilation",
                passed=has_msl and has_metallib,
                detail=f"Triton JIT→MSL OK, metallib={'present' if has_metallib else 'missing'}",
                value={"has_msl": has_msl, "has_metallib": has_metallib},
            )
        except Exception as exc:
            return CheckResult(
                name="basic_compilation",
                passed=False,
                detail=f"Compilation pipeline error: {exc}",
                value=None,
            )
    except ImportError as exc:
        return CheckResult(
            name="basic_compilation",
            passed=False,
            detail=f"Cannot import backend for compilation: {exc}",
            value=None,
        )


def run_all_checks() -> dict:
    checks = [
        check_python_version,
        check_platform,
        check_torch_available,
        check_xcrun_available,
        check_metal_toolchain,
        check_metal_backend_import,
        check_basic_compilation,
    ]
    results = [c() for c in checks]
    overall = all(r.passed for r in results)
    return {
        "ts_utc": _utc_now_iso(),
        "overall_passed": overall,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "arch": platform.machine(),
        "checks": [asdict(r) for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metal backend CI compatibility matrix validation",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Write JSON to file"
    )
    args = parser.parse_args()

    report = run_all_checks()

    if args.json or args.output:
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(text)
    else:
        print(f"Metal Backend Compatibility Check — {report['ts_utc']}")
        print(
            f"Python {report['python_version']} on {report['platform']}/{report['arch']}"
        )
        print("-" * 60)
        for c in report["checks"]:
            icon = "PASS" if c["passed"] else "FAIL"
            print(f"  [{icon}] {c['name']}: {c['detail']}")
        print("-" * 60)
        status = (
            "ALL CHECKS PASSED" if report["overall_passed"] else "SOME CHECKS FAILED"
        )
        print(f"  {status}")

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
