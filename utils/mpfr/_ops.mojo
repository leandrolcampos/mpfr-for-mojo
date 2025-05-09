# ===----------------------------------------------------------------------=== #
# Copyright 2025 The MPFR for Mojo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import math

from sys.ffi import c_int, c_long
from sys.intrinsics import likely, unlikely
from utils.numerics import FPUtils

from mpfr._bindings import mpfr_exp_t
from mpfr._wrapper import get_mpfr_rounding_mode, MpfrFloat, MpfrFloatPtr
from rounding import RoundingMode


# ===----------------------------------------------------------------------=== #
# Custom Operators
# ===----------------------------------------------------------------------=== #


fn ulp_error[
    dtype: DType, rounding_mode: RoundingMode, //
](
    output: MpfrFloatPtr[dtype, rounding_mode],
    expected: MpfrFloat[dtype, rounding_mode],
    actual: Scalar[dtype],
):
    alias PREC = FPUtils[dtype].mantissa_width() + 1
    alias EMAX = FPUtils[dtype].max_exponent() - 1
    alias EMIN = 1 - EMAX
    alias MIN_ULP_EXPONENT = EMIN - PREC + 1

    if unlikely(math.isnan(actual)):
        if likely(expected.is_nan()):
            set_zero(output)
        else:
            set_inf(output)

    elif unlikely(math.isinf(actual)):
        if likely(expected[] == actual):
            set_zero(output)
        else:
            set_inf(output)

    elif unlikely(not expected.is_number()):
        set_inf(output)

    else:
        var ulp_exponent = MIN_ULP_EXPONENT if expected.is_zero() else max(
            MIN_ULP_EXPONENT, expected.get_exponent() - PREC
        )

        output._lib.call["mpfr_sub_d"](
            output.as_mutable(),
            expected.as_immutable_ptr(),
            actual.cast[DType.float64](),
            get_mpfr_rounding_mode[RoundingMode.AWAY_FROM_ZERO](),
        )
        output._lib.call["mpfr_abs"](
            output.as_mutable(),
            output.as_immutable(),
            get_mpfr_rounding_mode[RoundingMode.TO_NEAREST](),
        )
        output._lib.call["mpfr_mul_2si"](
            output.as_mutable(),
            output.as_immutable(),
            -ulp_exponent,
            get_mpfr_rounding_mode[RoundingMode.TO_NEAREST](),
        )


# ===----------------------------------------------------------------------=== #
# Arithmetic Operators
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn add[
    dtype: DType, rounding_mode: RoundingMode, //
](
    rop: MpfrFloatPtr[dtype, rounding_mode],
    op1: MpfrFloat[dtype, rounding_mode],
    op2: __type_of(op1),
) -> c_int:
    return rop._lib.call["mpfr_add", c_int](
        rop.as_mutable(),
        op1.as_immutable_ptr(),
        op2.as_immutable_ptr(),
        rop._MPFR_ROUNDING_MODE,
    )


@always_inline("nodebug")
fn sqrt[
    dtype: DType, rounding_mode: RoundingMode, //
](
    rop: MpfrFloatPtr[dtype, rounding_mode],
    op: MpfrFloat[dtype, rounding_mode],
) -> c_int:
    return rop._lib.call["mpfr_sqrt", c_int](
        rop.as_mutable(), op.as_immutable_ptr(), op._MPFR_ROUNDING_MODE
    )


# ===----------------------------------------------------------------------=== #
# Assigmment Operators
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn set_d(rop: MpfrFloatPtr, op: Float64) -> c_int:
    return rop._lib.call["mpfr_set_d", c_int](
        rop.as_mutable(), op, rop._MPFR_ROUNDING_MODE
    )


@always_inline("nodebug")
fn set_flt(rop: MpfrFloatPtr, op: Float32) -> c_int:
    return rop._lib.call["mpfr_set_flt", c_int](
        rop.as_mutable(), op, rop._MPFR_ROUNDING_MODE
    )


@always_inline("nodebug")
fn set_inf(rop: MpfrFloatPtr, sign: c_int = 1):
    rop._lib.call["mpfr_set_inf"](rop.as_mutable(), sign)


@always_inline("nodebug")
fn set_nan(rop: MpfrFloatPtr):
    rop._lib.call["mpfr_set_nan"](rop.as_mutable())


@always_inline("nodebug")
fn set_str(rop: MpfrFloatPtr, s: StringLiteral, base: c_int = 0) -> c_int:
    return rop._lib.call["mpfr_set_str", c_int](
        rop.as_mutable(), s.unsafe_cstr_ptr(), base, rop._MPFR_ROUNDING_MODE
    )


@always_inline("nodebug")
fn set_str(rop: MpfrFloatPtr, owned s: String, base: c_int = 0) -> c_int:
    return rop._lib.call["mpfr_set_str", c_int](
        rop.as_mutable(), s.unsafe_cstr_ptr(), base, rop._MPFR_ROUNDING_MODE
    )


@always_inline("nodebug")
fn set_zero(rop: MpfrFloatPtr, sign: c_int = 1):
    rop._lib.call["mpfr_set_zero"](rop.as_mutable(), sign)


# ===----------------------------------------------------------------------=== #
# Comparison Operators
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn cmp(op1: MpfrFloat, op2: MpfrFloat) -> c_int:
    return op1._lib.call["mpfr_cmp", c_int](
        op1.as_immutable_ptr(), op2.as_immutable_ptr()
    )


@always_inline("nodebug")
fn cmp_d(op1: MpfrFloat, op2: Float64) -> c_int:
    return op1._lib.call["mpfr_cmp_d", c_int](op1.as_immutable_ptr(), op2)


@always_inline("nodebug")
fn cmp_si_2exp(op1: MpfrFloat, op2: c_long, e: mpfr_exp_t) -> c_int:
    return op1._lib.call["mpfr_cmp_si_2exp", c_int](
        op1.as_immutable_ptr(), op2, e
    )


# ===----------------------------------------------------------------------=== #
# Conversion Operators
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn get_d[
    in_rounding_mode: RoundingMode, //,
    rounding_mode: RoundingMode = in_rounding_mode,
](op: MpfrFloat[_, in_rounding_mode]) -> Float64:
    alias MPFR_ROUNDING_MODE = get_mpfr_rounding_mode[rounding_mode]()

    return op._lib.call["mpfr_get_d", Float64](
        op.as_immutable_ptr(), MPFR_ROUNDING_MODE
    )


@always_inline("nodebug")
fn get_flt[
    in_rounding_mode: RoundingMode, //,
    rounding_mode: RoundingMode = in_rounding_mode,
](op: MpfrFloat[_, in_rounding_mode]) -> Float32:
    alias MPFR_ROUNDING_MODE = get_mpfr_rounding_mode[rounding_mode]()

    return op._lib.call["mpfr_get_flt", Float32](
        op.as_immutable_ptr(), MPFR_ROUNDING_MODE
    )
