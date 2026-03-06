import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import triton
from triton._internal_testing import is_metal

pytestmark = [
    pytest.mark.skipif(not is_metal(), reason="Requires active Metal backend"),
    pytest.mark.skipif(shutil.which("xcrun") is None, reason="xcrun not found"),
]


def _has_mps_runtime() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


KERNEL_SRC = """
import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(Y + offs, x, mask=mask)
"""


def test_metal_aot_compile_emits_objc_launcher():
    repo_root = Path(__file__).resolve().parents[4]
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "metal"
    target_spec = f"{target.backend}:{target.arch}:{target.warp_size}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "kernel.py")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(KERNEL_SRC)

        compiler_path = repo_root / "python" / "triton" / "tools" / "compile.py"
        out_name = "copy_kernel"
        out_path = os.path.join(tmp_dir, out_name)
        cmd = [
            sys.executable,
            str(compiler_path),
            "-n",
            "kernel",
            "--signature",
            "*fp32, *fp32, i32, 128",
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            "4",
            "-g",
            "N/128, 1, 1",
            "-t",
            target_spec,
            kernel_path,
        ]
        env = os.environ.copy()
        python_path_parts = [str(repo_root), str(repo_root / "python")]
        if env.get("PYTHONPATH"):
            python_path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
        subprocess.run(cmd, check=True, cwd=tmp_dir, env=env)

        headers = glob.glob(os.path.join(tmp_dir, "copy_kernel.*.h"))
        impls = glob.glob(os.path.join(tmp_dir, "copy_kernel.*.m"))
        assert len(headers) == 1
        assert len(impls) == 1

        with open(headers[0], "r", encoding="utf-8") as f:
            header_text = f.read()
        with open(impls[0], "r", encoding="utf-8") as f:
            impl_text = f.read()

        assert "MTLBufferPtr" in header_text
        assert "id<MTLComputeCommandEncoder>" in impl_text
        assert "newLibraryWithData" in impl_text or "newLibraryWithURL" in impl_text


@pytest.mark.skipif(not _has_mps_runtime(), reason="MPS runtime not available")
def test_metal_aot_runtime_harness():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "test_metal_aot_runtime.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "aot-runtime-report.json"
        env = os.environ.copy()
        python_path_parts = [str(repo_root / "python")]
        if env.get("PYTHONPATH"):
            python_path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

        result = subprocess.run(
            [sys.executable, str(script_path), "--json", "-o", str(report_path)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "Metal AOT runtime harness failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["overall_passed"] is True
        assert report["steps"]["compile_harness"]["passed"] is True
        assert report["steps"]["compile_kernel"]["passed"] is True
        assert report["steps"]["run_harness"]["passed"] is True
