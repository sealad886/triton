import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

import triton
from triton._internal_testing import is_metal

pytestmark = [
    pytest.mark.skipif(not is_metal(), reason="Requires active Metal backend"),
    pytest.mark.skipif(shutil.which("xcrun") is None, reason="xcrun not found"),
]


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
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "metal"
    target_spec = f"{target.backend}:{target.arch}:{target.warp_size}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "kernel.py")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(KERNEL_SRC)

        compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")
        out_name = "copy_kernel"
        out_path = os.path.join(tmp_dir, out_name)
        cmd = [
            sys.executable,
            compiler_path,
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
        subprocess.run(cmd, check=True, cwd=tmp_dir)

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
