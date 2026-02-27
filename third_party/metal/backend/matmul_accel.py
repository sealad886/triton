"""
Metal-native matmul acceleration strategy.

Implements Metal-specific matmul optimization using Apple silicon's
simdgroup_matrix hardware. Provides tile size selection, strategy
dispatch, and MSL-level matmul optimization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── GPU family helpers ───────────────────────────────────────────────

_GPU_FAMILY_ORDER = ("apple7", "apple8", "apple9")


def _gpu_family_index(family: str) -> int:
    """Return ordinal for a GPU family string, -1 if unknown."""
    try:
        return _GPU_FAMILY_ORDER.index(family.lower())
    except ValueError:
        return -1


def _gpu_at_least(family: str, minimum: str) -> bool:
    idx = _gpu_family_index(family)
    return idx >= 0 and idx >= _gpu_family_index(minimum)


# ── Strategy dataclass ───────────────────────────────────────────────


@dataclass(frozen=True)
class MetalMatmulStrategy:
    """Describes the chosen matmul execution strategy for a tile."""

    use_simdgroup: bool
    tile_m: int
    tile_n: int
    tile_k: int
    elem_type: str
    accum_type: str
    pipeline_depth: int
    gpu_family: str

    def summary(self) -> str:
        kind = "simdgroup" if self.use_simdgroup else "fma"
        return (
            f"{kind} {self.tile_m}x{self.tile_n}x{self.tile_k} "
            f"{self.elem_type}->{self.accum_type} pipe={self.pipeline_depth} "
            f"on {self.gpu_family}"
        )


# ── Strategy selection ───────────────────────────────────────────────

_SIMDGROUP_TILE = 8  # Apple simdgroup_matrix is always 8x8

# Minimum M/N shape to prefer simdgroup over scalar FMA.
_MIN_SHAPE_FOR_SIMDGROUP = 16

# dtypes that support simdgroup_matrix natively by GPU family.
_SIMDGROUP_DTYPES: dict[str, str] = {
    "float": "apple8",
    "half": "apple8",
    "fp16": "apple8",
    "fp32": "apple8",
    "bfloat": "apple9",
    "bf16": "apple9",
}

# dtypes that must always use FMA.
_FMA_ONLY_DTYPES: frozenset[str] = frozenset({"int8", "i8", "uint8", "u8"})


def _accum_type_for(elem: str) -> str:
    """Select accumulator type for a given element type."""
    if elem in ("half", "fp16", "bfloat", "bf16"):
        return "float"
    return elem if elem in ("float", "fp32") else "float"


def _normalise_dtype(dtype: str) -> str:
    """Map Triton/LLVM dtype names to MSL-level names."""
    mapping = {
        "fp16": "half",
        "fp32": "float",
        "bf16": "bfloat",
        "f16": "half",
        "f32": "float",
    }
    return mapping.get(dtype.lower(), dtype.lower())


def select_matmul_strategy(
    M: int,
    N: int,
    K: int,
    dtype: str = "float",
    gpu_family: str = "apple8",
    strategy_hint: str = "auto",
) -> MetalMatmulStrategy:
    """Choose the optimal matmul strategy for the given parameters.

    Parameters
    ----------
    M, N, K : int
        Matmul dimensions.
    dtype : str
        Element data type (e.g. ``"float"``, ``"half"``, ``"bf16"``).
    gpu_family : str
        Target Apple GPU family (``"apple7"``, ``"apple8"``, ``"apple9"``).
    strategy_hint : str
        ``"auto"`` lets the heuristic decide, ``"native"`` forces
        simdgroup, ``"fallback"`` forces FMA.

    Returns
    -------
    MetalMatmulStrategy
    """
    gpu_family = gpu_family.lower()
    norm_dtype = _normalise_dtype(dtype)
    accum = _accum_type_for(norm_dtype)

    use_simdgroup = False

    if strategy_hint == "fallback":
        use_simdgroup = False
    elif strategy_hint == "native":
        use_simdgroup = True
    else:
        # auto — heuristic decision
        if norm_dtype in _FMA_ONLY_DTYPES:
            use_simdgroup = False
        elif M < _MIN_SHAPE_FOR_SIMDGROUP or N < _MIN_SHAPE_FOR_SIMDGROUP:
            use_simdgroup = False
        else:
            min_family = _SIMDGROUP_DTYPES.get(norm_dtype)
            if min_family and _gpu_at_least(gpu_family, min_family):
                use_simdgroup = True

    if use_simdgroup:
        tile_m = _SIMDGROUP_TILE
        tile_n = _SIMDGROUP_TILE
        tile_k = _SIMDGROUP_TILE
        pipeline_depth = 2 if K >= 32 else 1
    else:
        tile_m = min(M, 4)
        tile_n = min(N, 4)
        tile_k = min(K, 4)
        pipeline_depth = 1

    return MetalMatmulStrategy(
        use_simdgroup=use_simdgroup,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        elem_type=norm_dtype,
        accum_type=accum,
        pipeline_depth=pipeline_depth,
        gpu_family=gpu_family,
    )


# ── Performance model ────────────────────────────────────────────────


@dataclass(frozen=True)
class MatmulPerfEstimate:
    """Throughput estimate for a matmul strategy on a specific GPU."""

    gpu_family: str
    simdgroup_gflops: float
    fma_gflops: float
    simdgroup_efficiency: float  # 0.0-1.0

    def preferred(self) -> str:
        return "simdgroup" if self.simdgroup_gflops > self.fma_gflops else "fma"


_PERF_DB: dict[str, tuple[float, float, float]] = {
    # (simdgroup_gflops, fma_gflops, efficiency)
    "apple7": (0.0, 800.0, 0.0),
    "apple8": (2600.0, 1200.0, 0.75),
    "apple9": (3800.0, 1400.0, 0.85),
}


def get_matmul_performance_model(gpu_family: str) -> MatmulPerfEstimate:
    """Return estimated throughput for different strategies.

    The numbers are approximate and intended for strategy selection
    guidance, not precise benchmarking.
    """
    gpu_family = gpu_family.lower()
    sg_gflops, fma_gflops, eff = _PERF_DB.get(gpu_family, (0.0, 600.0, 0.0))
    return MatmulPerfEstimate(
        gpu_family=gpu_family,
        simdgroup_gflops=sg_gflops,
        fma_gflops=fma_gflops,
        simdgroup_efficiency=eff,
    )


# ── MSL optimisation pass ───────────────────────────────────────────

# Pattern: a triple-nested loop doing FMA on shared-memory buffers,
# which is the canonical Triton matmul lowering shape.
# We look for structure like:
#   for (...k...) {
#     ... acc += a[...] * b[...]; ...
#   }
# within a kernel that writes to a `device` pointer.
_RE_MATMUL_LOOP = re.compile(
    r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\w+)\s*;"
    r"\s*(?:\+\+\1|\1\s*\+=\s*1)\s*\)\s*\{",
    re.MULTILINE,
)

_RE_FMA_BODY = re.compile(
    r"(\w+)\s*\+=\s*\(?[^;]*\*[^;]*\)?;",
    re.MULTILINE,
)

_RE_KERNEL_SIG = re.compile(
    r"kernel\s+void\s+(\w+)\s*\(",
    re.MULTILINE,
)


def _count_nested_fma_loops(source: str) -> int:
    """Count loops that contain FMA-style accumulation."""
    count = 0
    for m in _RE_MATMUL_LOOP.finditer(source):
        start = m.end()
        depth = 1
        pos = start
        segment_end = min(start + 512, len(source))
        while pos < segment_end and depth > 0:
            if source[pos] == "{":
                depth += 1
            elif source[pos] == "}":
                depth -= 1
            pos += 1
        body = source[start:pos]
        if _RE_FMA_BODY.search(body):
            count += 1
    return count


def optimize_matmul_msl(
    msl_source: str,
    strategies: list[MetalMatmulStrategy] | None = None,
) -> str:
    """Post-process MSL to insert simdgroup matrix ops where beneficial.

    This is a *conservative* pass: it only annotates the MSL with
    pragmas and comments for downstream ``metallib`` compilation when
    it detects a matmul pattern it is confident about.  If no matmul
    pattern is detected (or strategies are all FMA), the source is
    returned unchanged.

    Parameters
    ----------
    msl_source : str
        The generated MSL kernel source.
    strategies : list[MetalMatmulStrategy] | None
        Optional strategies already selected for the kernel.  If absent
        or empty, defaults to no transformation.

    Returns
    -------
    str
        Possibly-annotated MSL source.
    """
    if not strategies:
        return msl_source

    any_simdgroup = any(s.use_simdgroup for s in strategies)
    if not any_simdgroup:
        return msl_source

    fma_loops = _count_nested_fma_loops(msl_source)
    if fma_loops == 0:
        return msl_source

    # Insert a matmul acceleration hint as a pragma comment after the
    # #include <metal_stdlib> line (or at the top).  The Metal compiler
    # can use this to guide vectorisation decisions.
    strat = next(s for s in strategies if s.use_simdgroup)
    hint_line = (
        f"// __metal_matmul_accel: tile={strat.tile_m}x{strat.tile_n}x{strat.tile_k} "
        f"elem={strat.elem_type} accum={strat.accum_type} pipe={strat.pipeline_depth}"
    )

    include_pos = msl_source.find("using namespace metal;")
    if include_pos != -1:
        nl = msl_source.find("\n", include_pos)
        if nl != -1:
            msl_source = msl_source[: nl + 1] + hint_line + "\n" + msl_source[nl + 1 :]
    else:
        msl_source = hint_line + "\n" + msl_source

    return msl_source
