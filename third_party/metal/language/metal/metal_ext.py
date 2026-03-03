"""Metal-specific Triton language extensions.

Provides Metal equivalents for hardware identity queries available
via the MSL threadgroup/simdgroup built-in variables.

MSL surfaces these through function parameter attributes rather than
inline assembly, so these helpers use ``core.inline_asm_elementwise``
with symbolic MSL built-in names.  The Metal LLVM->MSL translator is
responsible for recognising these tokens and emitting correct MSL
attribute-qualified parameters.
"""

from triton.language import core

# ---------------------------------------------------------------------------
# Hardware-identity externs
# ---------------------------------------------------------------------------


@core.extern
def thread_position_in_grid(_semantic=None):
    """Return the global thread position (Metal: ``thread_position_in_grid``)."""
    return core.inline_asm_elementwise(
        "thread_position_in_grid.x",
        "=r",
        [],
        dtype=core.uint32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def simdgroup_index(_semantic=None):
    """Return the simdgroup index within the threadgroup."""
    return core.inline_asm_elementwise(
        "simdgroup_index_in_threadgroup",
        "=r",
        [],
        dtype=core.uint32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def threadgroup_position(_semantic=None):
    """Return the threadgroup position in the grid."""
    return core.inline_asm_elementwise(
        "threadgroup_position_in_grid.x",
        "=r",
        [],
        dtype=core.uint32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


# ---------------------------------------------------------------------------
# MSL built-in reference (semantic constants)
# ---------------------------------------------------------------------------
# These document the Metal hardware ID built-ins that kernels may query.
# The actual plumbing through the compiler is via inline_asm_elementwise
# with the symbolic names above.

METAL_BUILTINS = {
    "thread_position_in_grid": "uint3  – global thread position",
    "thread_position_in_threadgroup": "uint3  – local thread position",
    "threadgroup_position_in_grid": "uint3  – workgroup position",
    "threads_per_threadgroup": "uint3  – workgroup dimensions",
    "simdgroup_index_in_threadgroup": "uint   – SIMD-group index",
    "thread_index_in_simdgroup": "uint   – lane index within SIMD-group",
}
