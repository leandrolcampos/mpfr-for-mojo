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

from sys.ffi import c_int, c_long

from mpfr._bindings import mpfr_exp_t
from mpfr._wrapper import MpfrFloat, MpfrFloatPtr


@always_inline("nodebug")
fn add[
    type: DType, rounding_mode: RoundingMode, //
](
    rop: MpfrFloatPtr[type, rounding_mode],
    op1: MpfrFloat[type, rounding_mode],
    op2: __type_of(op1),
) -> c_int:
    return rop._lib.call["mpfr_add", c_int](
        rop._value,
        op1._as_immutable_ptr(),
        op2._as_immutable_ptr(),
        op1._mpfr_rounding_mode,
    )


@always_inline("nodebug")
fn cmp_si_2exp(op1: MpfrFloat, op2: c_long, e: mpfr_exp_t) -> c_int:
    return op1._lib.call["mpfr_cmp_si_2exp", c_int](
        op1._as_immutable_ptr(), op2, e
    )


@always_inline("nodebug")
fn get_d(op: MpfrFloat) -> Float64:
    return op._lib.call["mpfr_get_d", Float64](
        op._as_immutable_ptr(), op._mpfr_rounding_mode
    )


@always_inline("nodebug")
fn get_flt(op: MpfrFloat) -> Float32:
    return op._lib.call["mpfr_get_flt", Float32](
        op._as_immutable_ptr(), op._mpfr_rounding_mode
    )


@always_inline("nodebug")
fn set_d(rop: MpfrFloatPtr, op: Float64) -> c_int:
    return rop._lib.call["mpfr_set_d", c_int](
        rop._value, op, rop._mpfr_rounding_mode
    )


@always_inline("nodebug")
fn set_str(rop: MpfrFloatPtr, s: StringLiteral, base: c_int = 0) -> c_int:
    return rop._lib.call["mpfr_set_str", c_int](
        rop._value, s.unsafe_cstr_ptr(), base, rop._mpfr_rounding_mode
    )


@always_inline("nodebug")
fn set_str(rop: MpfrFloatPtr, s: String, base: c_int = 0) -> c_int:
    return rop._lib.call["mpfr_set_str", c_int](
        rop._value, s.unsafe_cstr_ptr(), base, rop._mpfr_rounding_mode
    )


@always_inline("nodebug")
fn sqrt[
    type: DType, rounding_mode: RoundingMode, //
](
    rop: MpfrFloatPtr[type, rounding_mode],
    op: MpfrFloat[type, rounding_mode],
) -> c_int:
    return rop._lib.call["mpfr_sqrt", c_int](
        rop._value, op._as_immutable_ptr(), op._mpfr_rounding_mode
    )
