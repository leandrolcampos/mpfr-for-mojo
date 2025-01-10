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


from mpfr_wrapper._bindings import (
    mpfr_exp_t,
    mpfr_prec_t,
    mpfr_ptr,
    mpfr_rnd_t,
    MPFR_RNDA,
    MPFR_RNDD,
    MPFR_RNDN,
    MPFR_RNDU,
    MPFR_RNDZ,
    mpfr_srcptr,
    mpfr_t,
)
from mpfr_wrapper._library import MpfrLibrary, MpfrLibraryProvider


@value
@register_passable("trivial")
struct RoundingMode:
    var _value: Int

    alias AWAY_FROM_ZERO = RoundingMode(-1)
    alias TOWARD_ZERO = RoundingMode(0)
    alias TO_NEAREST = RoundingMode(1)
    alias UPWARD = RoundingMode(2)
    alias DOWNWARD = RoundingMode(3)
    alias TO_NEAREST_AWAY = RoundingMode(4)

    @always_inline("nodebug")
    fn __eq__(self, other: RoundingMode) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: RoundingMode) -> Bool:
        return self._value != other._value

    @always_inline("nodebug")
    fn __is__(self, other: RoundingMode) -> Bool:
        return self == other

    @always_inline("nodebug")
    fn __isnot__(self, other: RoundingMode) -> Bool:
        return self != other


fn _get_mpfr_rounding_mode[mode: RoundingMode]() -> mpfr_rnd_t:
    @parameter
    if mode is RoundingMode.AWAY_FROM_ZERO:
        return MPFR_RNDA
    elif mode is RoundingMode.TOWARD_ZERO:
        return MPFR_RNDZ
    elif mode is RoundingMode.TO_NEAREST:
        return MPFR_RNDN
    elif mode is RoundingMode.DOWNWARD:
        return MPFR_RNDD
    else:
        constrained[mode is RoundingMode.UPWARD, "unsupported rounding mode"]()
        return mpfr_rnd_t(MPFR_RNDU)


fn _get_mpfr_precision[type: DType]() -> mpfr_prec_t:
    @parameter
    if type is DType.float64:
        return 256
    else:
        return 128


fn _mpfr_float_construction_checks[type: DType]():
    constrained[
        type.is_floating_point(), "type must be a floating point type"
    ]()
    constrained[
        type in [DType.bfloat16, DType.float16, DType.float32, DType.float64],
        "unsupported floating point type",
    ]()


struct MpfrFloat[
    type: DType,
    rounding_mode: RoundingMode = RoundingMode.TO_NEAREST,
]:
    alias _precision = _get_mpfr_precision[type]()
    alias _mpfr_rounding_mode = _get_mpfr_rounding_mode[rounding_mode]()

    var _lib: MpfrLibrary
    var _value: mpfr_t

    fn __init__(out self):
        _mpfr_float_construction_checks[type]()

        self._lib = MpfrLibraryProvider.get_or_create_ptr()[].library()
        self._value = mpfr_t()
        self._lib.call["mpfr_init2"](
            self._as_mutable_ptr(),
            Self._precision,
        )

    fn __init__(out self, value: Scalar[type]):
        self = Self()
        self[] = value

    fn __init__(out self, existing: Self):
        self = Self()
        self._lib.call["mpfr_set"](
            self._as_mutable_ptr(),
            existing._as_immutable_ptr(),
            Self._mpfr_rounding_mode,
        )

    fn __del__(owned self):
        self._lib.call["mpfr_clear"](self._as_mutable_ptr())

    @always_inline("nodebug")
    fn __setitem__(mut self, value: Scalar[type]):
        @parameter
        if type is DType.float64:
            self._lib.call["mpfr_set_d"](
                self._as_mutable_ptr(),
                value,
                Self._mpfr_rounding_mode,
            )
        elif type is DType.float32:
            self._lib.call["mpfr_set_flt"](
                self._as_mutable_ptr(),
                value,
                Self._mpfr_rounding_mode,
            )
        else:
            self._lib.call["mpfr_set_flt"](
                self._as_mutable_ptr(),
                value.cast[DType.float32](),
                Self._mpfr_rounding_mode,
            )

    @always_inline("nodebug")
    fn __getitem__(self) -> Scalar[type]:
        @parameter
        if type is DType.float64:
            return self._lib.call["mpfr_get_d", Scalar[type]](
                self._as_immutable_ptr(),
                Self._mpfr_rounding_mode,
            )
        else:
            # TODO: Add support for other floating point types
            # without double-rounding errors.
            constrained[
                type is DType.float32, "unsupported floating point type"
            ]()
            return self._lib.call["mpfr_get_flt", Scalar[type]](
                self._as_immutable_ptr(),
                Self._mpfr_rounding_mode,
            )

    @always_inline("nodebug")
    fn get_exponent(self) -> mpfr_exp_t:
        return self._lib.call["mpfr_get_exp", mpfr_exp_t](
            self._as_immutable_ptr()
        )

    @always_inline("nodebug")
    fn get_sign(self) -> c_int:
        return self._lib.call["mpfr_sgn", c_int](self._as_immutable_ptr())

    @always_inline("nodebug")
    fn is_regular(self) -> Bool:
        return self._lib.call["mpfr_regular_p", Bool](self._as_immutable_ptr())

    @always_inline("nodebug")
    fn is_singular(self) -> Bool:
        return not self.is_regular()

    @always_inline("nodebug")
    fn _as_immutable_ptr(ref [ImmutableAnyOrigin]self) -> mpfr_srcptr:
        return Pointer.address_of(self._value)

    @always_inline("nodebug")
    fn _as_mutable_ptr(ref [MutableAnyOrigin]self) -> mpfr_ptr:
        return Pointer.address_of(self._value)


@register_passable("trivial")
struct MpfrFloatPtr[type: DType, rounding_mode: RoundingMode]:
    """Represents a mutable pointer to an MPFR float.

    This type is used for passing a mutable pointer to an MPFR float into
    functions that require one. It is not intended to be used directly.

    It is assumed that MPFR handles potential aliasing correctly.
    """

    alias _precision = _get_mpfr_precision[type]()
    alias _mpfr_rounding_mode = _get_mpfr_rounding_mode[rounding_mode]()

    var _lib: MpfrLibrary
    var _value: mpfr_ptr

    @implicit
    @always_inline
    fn __init__(out self, mut src: MpfrFloat[type, rounding_mode]):
        self._lib = src._lib
        self._value = src._as_mutable_ptr()


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
