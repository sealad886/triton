from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_GPU_FAMILY_ORDER = ("apple7", "apple8", "apple9")


def _gpu_family_index(family: str) -> int:
    try:
        return _GPU_FAMILY_ORDER.index(str(family).lower())
    except ValueError:
        return -1


def _gpu_at_least(family: str, minimum: str) -> bool:
    idx = _gpu_family_index(family)
    return idx >= 0 and idx >= _gpu_family_index(minimum)


SUPPORTED = "supported"
LIMITED = "limited"
UNSUPPORTED = "unsupported"

_SIMDGROUP_NATIVE_MIN_GPU = {
    "float": "apple7",
    "half": "apple7",
    "bfloat": "apple9",
}

_GENERIC_FMA_DTYPES = frozenset(
    {
        "bool",
        "i1",
        "int8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "float",
        "half",
        "bfloat",
        "fp8e5",
        "fp8e4b15",
    }
)

_STRICT_FIRST_CLASS_BLOCKERS = (
    "launch_pdl is accepted for ABI parity but remains a no-op",
    "profile_scratch is metadata-only until a Metal profiler runtime path exists",
    "cross-backend HIP numerical parity is still supplemental rather than always-on",
    "self-hosted Apple GPU runtime, throughput, and soak validation remains supplemental",
)


@dataclass(frozen=True)
class CapabilityStatus:
    level: str
    detail: str


@dataclass(frozen=True)
class DotCapability:
    lhs_dtype: str
    rhs_dtype: str
    gpu_family: str
    supported: bool
    min_dot_size: tuple[int, int, int]
    supports_fma_fallback: bool
    supports_simdgroup: bool
    native_tile: tuple[int, int, int] | None
    required_accumulator: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _status(level: str, detail: str) -> dict[str, str]:
    return asdict(CapabilityStatus(level=level, detail=detail))


def _safe_predicate(obj: Any, name: str) -> bool:
    attr = getattr(obj, name, None)
    if not callable(attr):
        return False
    try:
        result = attr()
    except Exception:
        return False
    return isinstance(result, bool) and result


def _normalise_dtype_name(name: str | None) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    norm = name.strip().lower()
    aliases = {
        "bool": "bool",
        "i1": "bool",
        "int1": "bool",
        "i8": "int8",
        "int8": "int8",
        "uint8": "u8",
        "u8": "u8",
        "i16": "i16",
        "int16": "i16",
        "uint16": "u16",
        "u16": "u16",
        "i32": "i32",
        "int32": "i32",
        "uint32": "u32",
        "u32": "u32",
        "i64": "i64",
        "int64": "i64",
        "uint64": "u64",
        "u64": "u64",
        "f16": "half",
        "fp16": "half",
        "float16": "half",
        "half": "half",
        "f32": "float",
        "fp32": "float",
        "float32": "float",
        "float": "float",
        "bf16": "bfloat",
        "bfloat16": "bfloat",
        "bfloat": "bfloat",
        "fp8e5": "fp8e5",
        "fp8e5m2": "fp8e5",
        "fp8e4b15": "fp8e4b15",
    }
    return aliases.get(norm, norm)


def _infer_scalar_dtype(operand_type: Any) -> str:
    scalar = getattr(operand_type, "scalar", operand_type)

    for attr_name in ("name", "dtype", "scalar_name", "typename"):
        dtype_name = _normalise_dtype_name(getattr(scalar, attr_name, None))
        if dtype_name is not None:
            return dtype_name

    predicate_map = (
        ("is_fp16", "half"),
        ("is_f16", "half"),
        ("is_fp32", "float"),
        ("is_f32", "float"),
        ("is_bf16", "bfloat"),
        ("is_bool", "bool"),
        ("is_int8", "int8"),
        ("is_uint8", "u8"),
        ("is_int16", "i16"),
        ("is_uint16", "u16"),
        ("is_int32", "i32"),
        ("is_uint32", "u32"),
        ("is_int64", "i64"),
        ("is_uint64", "u64"),
    )
    for predicate_name, dtype_name in predicate_map:
        if _safe_predicate(scalar, predicate_name):
            return dtype_name

    bitwidth = getattr(scalar, "primitive_bitwidth", None)
    if isinstance(bitwidth, int):
        if bitwidth == 1:
            return "bool"
        if _safe_predicate(scalar, "is_floating"):
            if bitwidth == 16:
                return "half"
            if bitwidth == 32:
                return "float"
        signedness = _normalise_dtype_name(getattr(scalar, "signedness", None))
        if bitwidth == 8:
            return "u8" if signedness == "unsigned" else "int8"
        if bitwidth == 16:
            return "u16" if signedness == "unsigned" else "i16"
        if bitwidth == 32:
            return "u32" if signedness == "unsigned" else "i32"
        if bitwidth == 64:
            return "u64" if signedness == "unsigned" else "i64"

    dtype_name = _normalise_dtype_name(str(scalar))
    if dtype_name is not None:
        return dtype_name
    return "unknown"


def describe_dot_capability(
    lhs_type: Any,
    rhs_type: Any,
    gpu_family: str = "apple8",
) -> DotCapability:
    lhs_dtype = _infer_scalar_dtype(lhs_type)
    rhs_dtype = _infer_scalar_dtype(rhs_type)
    family = str(gpu_family or "apple8").lower()

    if lhs_dtype in {"i64", "u64"} or rhs_dtype in {"i64", "u64"}:
        detail = (
            "Metal does not support fp64/i64 dot operands "
            f"(got lhs={lhs_dtype}, rhs={rhs_dtype})"
        )
        return DotCapability(
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            gpu_family=family,
            supported=False,
            min_dot_size=(0, 0, 0),
            supports_fma_fallback=False,
            supports_simdgroup=False,
            native_tile=None,
            required_accumulator=None,
            detail=detail,
        )

    supports_fma = lhs_dtype in _GENERIC_FMA_DTYPES and rhs_dtype in _GENERIC_FMA_DTYPES
    if not supports_fma:
        detail = (
            "Metal dot lowering only supports common <=32-bit scalar operand families; "
            f"got lhs={lhs_dtype}, rhs={rhs_dtype}."
        )
        return DotCapability(
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            gpu_family=family,
            supported=False,
            min_dot_size=(0, 0, 0),
            supports_fma_fallback=False,
            supports_simdgroup=False,
            native_tile=None,
            required_accumulator=None,
            detail=detail,
        )

    native_min_gpu = _SIMDGROUP_NATIVE_MIN_GPU.get(lhs_dtype)
    supports_simdgroup = (
        lhs_dtype == rhs_dtype
        and native_min_gpu is not None
        and _gpu_at_least(family, native_min_gpu)
    )

    if supports_simdgroup:
        detail = (
            "Generic FMA dot lowering is available, and native Metal simdgroup "
            f"matmul can be selected on {family} for {lhs_dtype} tiles when M/N are >= 16 "
            "and K is a multiple of 8."
        )
    elif lhs_dtype == rhs_dtype and native_min_gpu is not None:
        detail = (
            "Generic FMA dot lowering is available, but native Metal simdgroup "
            f"matmul for {lhs_dtype} requires {native_min_gpu}+ (current target: {family})."
        )
    else:
        detail = (
            "Generic FMA dot lowering is available; native Metal simdgroup matmul is reserved "
            f"for matching float/half/bfloat operand pairs, got lhs={lhs_dtype}, rhs={rhs_dtype}."
        )

    return DotCapability(
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        gpu_family=family,
        supported=True,
        min_dot_size=(1, 1, 1),
        supports_fma_fallback=True,
        supports_simdgroup=supports_simdgroup,
        native_tile=(16, 16, 8) if supports_simdgroup else None,
        required_accumulator="float" if supports_simdgroup else None,
        detail=detail,
    )


def runtime_mode_snapshot(
    *,
    torch_compile_shader: bool,
    mps_available: bool,
    pyobjc_available: bool,
) -> dict[str, dict[str, str]]:
    if torch_compile_shader and mps_available:
        torch_status = _status(
            SUPPORTED,
            "`torch.mps.compile_shader` is available and can execute runtime probes.",
        )
    elif torch_compile_shader:
        torch_status = _status(
            LIMITED,
            "`torch.mps.compile_shader` is present, but MPS execution is unavailable on this host.",
        )
    else:
        torch_status = _status(
            UNSUPPORTED,
            "`torch.mps.compile_shader` is unavailable in this environment.",
        )

    pyobjc_status = _status(
        SUPPORTED if pyobjc_available else UNSUPPORTED,
        (
            "PyObjC metallib fallback can compile/load a pipeline directly."
            if pyobjc_available
            else "PyObjC Metal/Foundation bindings are unavailable."
        ),
    )

    return {
        "torch_mps": torch_status,
        "pyobjc_metallib": pyobjc_status,
        "compile_only": _status(
            SUPPORTED,
            "Compile-only validation remains available via xcrun metal/metallib.",
        ),
    }


def metal_capability_snapshot(gpu_family: str = "apple8") -> dict[str, Any]:
    family = str(gpu_family or "apple8").lower()
    bf16_detail = (
        "bf16 compute and native simdgroup matmul are available on apple9+."
        if _gpu_at_least(family, "apple9")
        else "bf16 requires apple9+ for runtime-confidence and native simdgroup matmul."
    )
    return {
        "gpu_family": family,
        "repo_first_class_bar": _status(
            SUPPORTED,
            "Hosted correctness gating in CI/release is the repository's current first-class bar.",
        ),
        "strict_first_class_bar": _status(
            LIMITED,
            "Launch-contract gaps and supplemental hardware validation still preclude a stricter parity bar.",
        ),
        "launch_contract": {
            "launch_cooperative_grid": _status(
                SUPPORTED,
                "Metal accepts cooperative-grid launches through the standard dispatch path.",
            ),
            "launch_pdl": _status(
                LIMITED,
                "Accepted for ABI compatibility, but currently a no-op on Metal.",
            ),
            "profile_scratch": _status(
                LIMITED,
                "Metadata is preserved for tooling, but the runtime does not consume profile scratch yet.",
            ),
            "global_scratch": _status(
                SUPPORTED,
                "Global scratch allocation is wired through the runtime launch path.",
            ),
        },
        "runtime_modes": runtime_mode_snapshot(
            torch_compile_shader=False,
            mps_available=False,
            pyobjc_available=False,
        ),
        "validation": {
            "hosted_correctness_ci": _status(
                SUPPORTED,
                "Hosted correctness gating is always-on in primary CI and release workflows.",
            ),
            "self_hosted_gpu_runtime": _status(
                LIMITED,
                "GPU runtime, throughput, and soak coverage remain supplemental self-hosted lanes.",
            ),
            "apple_family_throughput_guardrails": _status(
                LIMITED,
                "Throughput guardrails exist, but always-on apple7/apple8/apple9 coverage is still supplemental.",
            ),
            "cross_backend_cuda": _status(
                LIMITED,
                "Deterministic CPU-reference comparisons cover MPS and optional CUDA.",
            ),
            "cross_backend_hip": _status(
                LIMITED,
                "HIP-backed parity remains incomplete.",
            ),
        },
        "matmul": {
            "generic_fma_dot": _status(
                SUPPORTED,
                "Generic FMA dot lowering remains the portable baseline across supported <=32-bit operand families.",
            ),
            "native_simdgroup_float": _status(
                SUPPORTED,
                "float/half native simdgroup matmul is available on apple7+ when tile shapes align.",
            ),
            "native_simdgroup_bfloat": _status(
                SUPPORTED if _gpu_at_least(family, "apple9") else LIMITED,
                bf16_detail,
            ),
        },
        "strict_first_class_blockers": list(_STRICT_FIRST_CLASS_BLOCKERS),
    }
