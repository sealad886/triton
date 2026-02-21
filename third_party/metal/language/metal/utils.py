"""Metal-specific language extensions for Triton.

Provides Metal GPU builtins accessible via ``triton.language.extra.metal``.
"""

from triton.language import core


@core.builtin
def num_threads(_semantic=None):
    """Return the total number of threads (num_warps * simd_width)."""
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    """Return the number of SIMD groups (warps) configured for this kernel."""
    return core.constexpr(_semantic.builder.options.num_warps)


@core.builtin
def simd_size(_semantic=None):
    """Return the SIMD group width (always 32 on Apple Silicon)."""
    return core.constexpr(32)
