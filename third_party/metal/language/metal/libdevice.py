"""
Metal-specific libdevice-equivalent math functions.

Provides Triton language bindings for Metal Shading Language standard library
math functions, analogous to NVIDIA's ``libdevice``.  Each function maps to
an LLVM intrinsic that the Metal backend's LLVM→MSL translator
(``compiler.py``) already knows how to lower.

MSL function column documents the Metal Shading Language standard library
function that the LLVM intrinsic ultimately becomes.
"""

from triton.language import core

# ── Mapping table ─────────────────────────────────────────────────────
# Each entry: (metal_extern_name, msl_function)
#
# The extern name is what Triton emits into the LLVM IR; the MSL function
# is what compiler.py's ``lower_intrinsic`` produces.

METAL_LIBDEVICE_MAP: dict[str, tuple[str, str]] = {
    "clz": ("__metal_clz", "clz"),
    "popc": ("__metal_popcount", "popcount"),
    "abs_i32": ("__metal_abs", "abs"),
    "abs_f32": ("__metal_fabs", "fabs"),
    "min_i32": ("__metal_min", "min"),
    "min_f32": ("__metal_fmin", "fmin"),
    "max_i32": ("__metal_max", "max"),
    "max_f32": ("__metal_fmax", "fmax"),
    "fma": ("__metal_fma", "fma"),
    "rsqrt": ("__metal_rsqrt", "rsqrt"),
    "exp2": ("__metal_exp2", "exp2"),
    "log2": ("__metal_log2", "log2"),
    "sin": ("__metal_sin", "sin"),
    "cos": ("__metal_cos", "cos"),
    "ceil": ("__metal_ceil", "ceil"),
    "floor": ("__metal_floor", "floor"),
    "trunc": ("__metal_trunc", "trunc"),
    "round": ("__metal_round", "rint"),
    "saturate": ("__metal_saturate", "saturate"),
    "sqrt": ("__metal_sqrt", "sqrt"),
}


# ── Single-argument functions ─────────────────────────────────────────


@core.extern
def clz(arg0, _semantic=None):
    """Count leading zeros — MSL ``clz()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("int32"),): ("__metal_clz", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def popc(arg0, _semantic=None):
    """Population count — MSL ``popcount()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("int32"),): ("__metal_popcount", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def abs(arg0, _semantic=None):
    """Absolute value — MSL ``abs()`` / ``fabs()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("int32"),): ("__metal_abs", core.dtype("int32")),
            (core.dtype("fp32"),): ("__metal_fabs", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_fabs", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def floor(arg0, _semantic=None):
    """Floor — MSL ``floor()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_floor", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_floor", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def ceil(arg0, _semantic=None):
    """Ceiling — MSL ``ceil()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_ceil", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_ceil", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def trunc(arg0, _semantic=None):
    """Truncation — MSL ``trunc()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_trunc", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_trunc", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def round(arg0, _semantic=None):
    """Round to nearest even — MSL ``rint()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_round", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_round", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def rsqrt(arg0, _semantic=None):
    """Reciprocal square root — MSL ``rsqrt()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_rsqrt", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_rsqrt", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def sqrt(arg0, _semantic=None):
    """Square root — MSL ``sqrt()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_sqrt", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_sqrt", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def exp2(arg0, _semantic=None):
    """Base-2 exponential — MSL ``exp2()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_exp2", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_exp2", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def log2(arg0, _semantic=None):
    """Base-2 logarithm — MSL ``log2()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_log2", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_log2", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def sin(arg0, _semantic=None):
    """Sine — MSL ``sin()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_sin", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_sin", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def cos(arg0, _semantic=None):
    """Cosine — MSL ``cos()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_cos", core.dtype("fp32")),
            (core.dtype("fp64"),): ("__metal_cos", core.dtype("fp64")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


# ── Two-argument functions ────────────────────────────────────────────


@core.extern
def min(arg0, arg1, _semantic=None):
    """Minimum — MSL ``min()`` / ``fmin()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0, arg1],
        {
            (core.dtype("int32"), core.dtype("int32")): (
                "__metal_min",
                core.dtype("int32"),
            ),
            (core.dtype("uint32"), core.dtype("uint32")): (
                "__metal_umin",
                core.dtype("uint32"),
            ),
            (core.dtype("fp32"), core.dtype("fp32")): (
                "__metal_fmin",
                core.dtype("fp32"),
            ),
            (core.dtype("fp64"), core.dtype("fp64")): (
                "__metal_fmin",
                core.dtype("fp64"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def max(arg0, arg1, _semantic=None):
    """Maximum — MSL ``max()`` / ``fmax()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0, arg1],
        {
            (core.dtype("int32"), core.dtype("int32")): (
                "__metal_max",
                core.dtype("int32"),
            ),
            (core.dtype("uint32"), core.dtype("uint32")): (
                "__metal_umax",
                core.dtype("uint32"),
            ),
            (core.dtype("fp32"), core.dtype("fp32")): (
                "__metal_fmax",
                core.dtype("fp32"),
            ),
            (core.dtype("fp64"), core.dtype("fp64")): (
                "__metal_fmax",
                core.dtype("fp64"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )


# ── Three-argument functions ──────────────────────────────────────────


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    """Fused multiply-add — MSL ``fma()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0, arg1, arg2],
        {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): (
                "__metal_fma",
                core.dtype("fp32"),
            ),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): (
                "__metal_fma",
                core.dtype("fp64"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )


# ── Metal-specific functions ──────────────────────────────────────────


@core.extern
def saturate(arg0, _semantic=None):
    """Clamp to [0, 1] — MSL ``saturate()``."""
    return core.extern_elementwise(
        "",
        "",
        [arg0],
        {
            (core.dtype("fp32"),): ("__metal_saturate", core.dtype("fp32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
