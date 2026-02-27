"""
Metal FP8 conversion utilities.

Provides fp8<->fp16/fp32 conversion functions for Metal backend.  Apple Silicon
does not have native fp8 hardware, so these are *software* conversions that
operate on raw bit patterns via Python integers/struct.

The GPU-side fp8 casts are already handled by the LLVM->MSL translator in
``compiler.py`` (see ``lower_cast``).  These Python-level helpers are provided
for:

* Reference/testing — validate bit-exact fp8 round-trip behaviour.
* Host-side pre/post-processing of fp8 tensors before Metal kernel launch.

Format summary
--------------
* **E5M2** (IEEE 754-style): 1 sign, 5 exponent, 2 mantissa bits.
  bias = 15, range ~ +-57344, smallest normal ~ 6.1e-5, NaN/Inf supported.
* **E4B15** (custom): 1 sign, 4 exponent, 3 mantissa bits.
  bias = 15 (B15 naming), range tighter, no Inf representation.
"""

from __future__ import annotations

import math
import struct


# ── E5M2 (1-5-2) ─────────────────────────────────────────────────────

_E5M2_EXP_BITS = 5
_E5M2_MAN_BITS = 2
_E5M2_BIAS = 15
_E5M2_MAX_EXP = (1 << _E5M2_EXP_BITS) - 1  # 31


def convert_fp8e5m2_to_fp16(x: int) -> float:
    """Decode an 8-bit E5M2 value (unsigned byte) to a Python float via fp16.

    Parameters
    ----------
    x : int
        Unsigned 8-bit integer representing an E5M2 value (0–255).

    Returns
    -------
    float
        The decoded floating-point value.
    """
    x = x & 0xFF
    sign = (x >> 7) & 1
    exp = (x >> _E5M2_MAN_BITS) & _E5M2_MAX_EXP
    man = x & ((1 << _E5M2_MAN_BITS) - 1)

    if exp == _E5M2_MAX_EXP:
        if man != 0:
            return float("nan")
        return float("-inf") if sign else float("inf")

    if exp == 0:
        # Denormal: value = (-1)^s * 2^(1-bias) * (0.mantissa)
        fval = (man / (1 << _E5M2_MAN_BITS)) * (2.0 ** (1 - _E5M2_BIAS))
    else:
        fval = (1.0 + man / (1 << _E5M2_MAN_BITS)) * (2.0 ** (exp - _E5M2_BIAS))

    return -fval if sign else fval


def convert_fp16_to_fp8e5m2(value: float) -> int:
    """Encode a Python float to an 8-bit E5M2 unsigned byte via fp16 range.

    Uses round-to-nearest-even.

    Parameters
    ----------
    value : float
        The value to encode.

    Returns
    -------
    int
        Unsigned 8-bit integer representing the E5M2 encoding (0–255).
    """
    if math.isnan(value):
        return 0x7F  # canonical NaN: 0 11111 11

    sign = 0
    if value < 0:
        sign = 1
        value = -value

    if math.isinf(value):
        return (sign << 7) | (_E5M2_MAX_EXP << _E5M2_MAN_BITS)

    if value == 0.0:
        return sign << 7

    # Clamp to E5M2 max finite value: (1 + 3/4) * 2^15 = 57344
    max_val = (1.0 + ((1 << _E5M2_MAN_BITS) - 1) / (1 << _E5M2_MAN_BITS)) * (
        2.0 ** (_E5M2_MAX_EXP - 1 - _E5M2_BIAS)
    )
    if value >= max_val:
        # Overflow → Inf
        return (sign << 7) | (_E5M2_MAX_EXP << _E5M2_MAN_BITS)

    # Compute exponent
    log_val = math.floor(math.log2(value))
    exp = log_val + _E5M2_BIAS

    if exp <= 0:
        # Denormal range
        man_f = value / (2.0 ** (1 - _E5M2_BIAS))
        man = _round_to_nearest_even(man_f * (1 << _E5M2_MAN_BITS))
        man = builtins_min(man, (1 << _E5M2_MAN_BITS) - 1)
        return (sign << 7) | man

    # Normal range
    significand = value / (2.0 ** log_val) - 1.0
    man = _round_to_nearest_even(significand * (1 << _E5M2_MAN_BITS))

    if man >= (1 << _E5M2_MAN_BITS):
        man = 0
        exp += 1

    if exp >= _E5M2_MAX_EXP:
        return (sign << 7) | (_E5M2_MAX_EXP << _E5M2_MAN_BITS)

    exp = builtins_min(exp, _E5M2_MAX_EXP - 1)
    return (sign << 7) | (exp << _E5M2_MAN_BITS) | man


# ── E4B15 (1-4-3, bias=15) ───────────────────────────────────────────
# "B15" denotes the unusual bias of 15 for a 4-bit exponent field.

_E4B15_EXP_BITS = 4
_E4B15_MAN_BITS = 3
_E4B15_BIAS = 15  # atypically large bias → very narrow range of tiny values
_E4B15_MAX_EXP = (1 << _E4B15_EXP_BITS) - 1  # 15


def convert_fp8e4b15_to_fp16(x: int) -> float:
    """Decode an 8-bit E4B15 value to a Python float.

    E4B15 has *no* Inf representation — all-ones exponent with non-zero
    mantissa is NaN, and all-ones exponent with zero mantissa is also NaN.

    Parameters
    ----------
    x : int
        Unsigned 8-bit integer (0–255).

    Returns
    -------
    float
        The decoded value.
    """
    x = x & 0xFF
    sign = (x >> 7) & 1
    exp = (x >> _E4B15_MAN_BITS) & _E4B15_MAX_EXP
    man = x & ((1 << _E4B15_MAN_BITS) - 1)

    if exp == _E4B15_MAX_EXP:
        # E4B15 has no Inf — all NaN
        return float("nan")

    if exp == 0:
        if man == 0:
            return -0.0 if sign else 0.0
        fval = (man / (1 << _E4B15_MAN_BITS)) * (2.0 ** (1 - _E4B15_BIAS))
    else:
        fval = (1.0 + man / (1 << _E4B15_MAN_BITS)) * (2.0 ** (exp - _E4B15_BIAS))

    return -fval if sign else fval


def convert_fp16_to_fp8e4b15(value: float) -> int:
    """Encode a Python float to an 8-bit E4B15 unsigned byte.

    E4B15 has *no* Inf; overflow saturates to NaN.

    Parameters
    ----------
    value : float
        The value to encode.

    Returns
    -------
    int
        Unsigned 8-bit integer (0–255).
    """
    if math.isnan(value):
        return 0x7F  # canonical NaN

    sign = 0
    if value < 0:
        sign = 1
        value = -value

    if math.isinf(value):
        # No Inf in E4B15 — saturate to NaN
        return (sign << 7) | (_E4B15_MAX_EXP << _E4B15_MAN_BITS) | 0x1

    if value == 0.0:
        return sign << 7

    max_normal_exp = _E4B15_MAX_EXP - 1  # 14
    max_val = (1.0 + ((1 << _E4B15_MAN_BITS) - 1) / (1 << _E4B15_MAN_BITS)) * (
        2.0 ** (max_normal_exp - _E4B15_BIAS)
    )
    if value > max_val:
        # Overflow → NaN (E4B15 has no Inf)
        return (sign << 7) | (_E4B15_MAX_EXP << _E4B15_MAN_BITS) | 0x1

    log_val = math.floor(math.log2(value))
    exp = log_val + _E4B15_BIAS

    if exp <= 0:
        man_f = value / (2.0 ** (1 - _E4B15_BIAS))
        man = _round_to_nearest_even(man_f * (1 << _E4B15_MAN_BITS))
        man = builtins_min(man, (1 << _E4B15_MAN_BITS) - 1)
        return (sign << 7) | man

    significand = value / (2.0 ** log_val) - 1.0
    man = _round_to_nearest_even(significand * (1 << _E4B15_MAN_BITS))

    if man >= (1 << _E4B15_MAN_BITS):
        man = 0
        exp += 1

    if exp >= _E4B15_MAX_EXP:
        return (sign << 7) | (_E4B15_MAX_EXP << _E4B15_MAN_BITS) | 0x1

    exp = builtins_min(exp, _E4B15_MAX_EXP - 1)
    return (sign << 7) | (exp << _E4B15_MAN_BITS) | man


# ── Helpers ───────────────────────────────────────────────────────────

# Avoid shadowing builtins used above
import builtins as _builtins
builtins_min = _builtins.min


def _round_to_nearest_even(x: float) -> int:
    """Round-to-nearest-even (banker's rounding)."""
    rounded = _builtins.round(x)
    return int(rounded)


def fp16_bits_to_float(bits: int) -> float:
    """Convert a 16-bit integer (IEEE 754 half) to a Python float."""
    packed = struct.pack(">H", bits & 0xFFFF)
    return struct.unpack(">e", packed)[0]


def float_to_fp16_bits(value: float) -> int:
    """Convert a Python float to a 16-bit integer (IEEE 754 half)."""
    packed = struct.pack(">e", value)
    return struct.unpack(">H", packed)[0]
