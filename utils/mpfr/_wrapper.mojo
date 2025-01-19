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
from sys.intrinsics import unlikely
from utils.numerics import FPUtils


from mpfr._bindings import (
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
from mpfr._library import MpfrLibrary, MpfrLibraryProvider
from mpfr._ops import cmp_si_2exp, get_flt, set_str
from rounding import RoundingMode


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
](Defaultable, ExplicitlyCopyable):
    alias _precision = _get_mpfr_precision[type]()
    alias _mpfr_rounding_mode = _get_mpfr_rounding_mode[rounding_mode]()

    var _lib: MpfrLibrary
    var _value: mpfr_t

    @always_inline("nodebug")
    fn __init__(out self):
        _mpfr_float_construction_checks[type]()

        self._lib = MpfrLibraryProvider.get_or_create_ptr()[].library()
        self._value = mpfr_t()
        self._lib.call["mpfr_init2"](
            self._as_mutable_ptr(),
            Self._precision,
        )

    @always_inline("nodebug")
    fn __init__(out self, value: Scalar[type]):
        self = Self()
        self[] = value

    @always_inline("nodebug")
    fn __init__(out self, value: StringLiteral, base: c_int = 0):
        self = Self()
        _ = set_str(self, value, base)

    @always_inline("nodebug")
    fn __init__(out self, value: String, base: c_int = 0):
        self = Self()
        _ = set_str(self, value, base)

    @always_inline("nodebug")
    fn __del__(owned self):
        self._lib.call["mpfr_clear"](self._as_mutable_ptr())

    @always_inline("nodebug")
    fn copy(self, out result: Self):
        result = Self()
        result._lib.call["mpfr_set"](
            result._as_mutable_ptr(),
            self._as_immutable_ptr(),
            Self._mpfr_rounding_mode,
        )

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

    @always_inline
    fn __getitem__(self) -> Scalar[type]:
        @parameter
        if type is DType.float64:
            return self._lib.call["mpfr_get_d", Scalar[type]](
                self._as_immutable_ptr(),
                Self._mpfr_rounding_mode,
            )
        elif type is DType.float32:
            return self._lib.call["mpfr_get_flt", Scalar[type]](
                self._as_immutable_ptr(),
                Self._mpfr_rounding_mode,
            )
        else:
            return _get_value(self).cast[type]()

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
    @always_inline("nodebug")
    fn __init__(out self, mut src: MpfrFloat[type, rounding_mode]):
        self._lib = src._lib
        self._value = src._as_mutable_ptr()


@always_inline("nodebug")
fn _eval[T: AnyType, //, val: T]() -> T:
    return val


@always_inline
fn _get_value(src: MpfrFloat) -> Float32:
    alias INF = math.inf[DType.float32]()
    alias ONE = Scalar[DType.float32](1.0)
    alias PREC = FPUtils[src.type].mantissa_width() + 1  # 24
    alias EMAX = FPUtils[src.type].max_exponent() - 1  # 127
    alias EMIN_NORMAL = 1 - EMAX  # -126
    alias EMIN_SUBNORMAL = EMIN_NORMAL + 1 - PREC  # -149
    alias EPSILON = math.ldexp(ONE, -PREC + 1)
    alias EPSILON_NEG = math.ldexp(ONE, -PREC)
    alias SMALLEST_SUBNORMAL = EPSILON * math.ldexp(ONE, EMIN_NORMAL)
    alias MAX_FINITE = math.ldexp(2.0 * (1.0 - EPSILON_NEG), EMAX)

    if unlikely(src.is_singular()):
        return get_flt(src)

    var mpfr_rounding_mode = src._mpfr_rounding_mode
    var exponent = src.get_exponent()
    var is_negative = src.get_sign() < 0
    var result: Float32

    if unlikely(mpfr_rounding_mode == MPFR_RNDA):
        mpfr_rounding_mode = MPFR_RNDD if is_negative else MPFR_RNDU

    if unlikely(exponent <= EMIN_SUBNORMAL):
        result = is_negative.select(
            (
                (mpfr_rounding_mode == MPFR_RNDD)
                | (
                    (mpfr_rounding_mode == MPFR_RNDN)
                    & (cmp_si_2exp(src, -1, _eval[EMIN_SUBNORMAL - 1]()) < 0)
                )
            ).select(-SMALLEST_SUBNORMAL, -0.0),
            (
                (mpfr_rounding_mode == MPFR_RNDU)
                | (
                    (mpfr_rounding_mode == MPFR_RNDN)
                    & (cmp_si_2exp(src, 1, _eval[EMIN_SUBNORMAL - 1]()) > 0)
                )
            ).select(SMALLEST_SUBNORMAL, 0.0),
        )
    elif unlikely(exponent > _eval[EMAX + 1]()):
        result = is_negative.select(
            (
                (mpfr_rounding_mode == MPFR_RNDZ)
                | (mpfr_rounding_mode == MPFR_RNDU)
            ).select(-MAX_FINITE, -INF),
            (
                (mpfr_rounding_mode == MPFR_RNDZ)
                | (mpfr_rounding_mode == MPFR_RNDD)
            ).select(MAX_FINITE, INF),
        )
    else:
        var tmp = src.copy()
        var prec = PREC + (exponent <= EMIN_NORMAL).select(
            -(EMIN_NORMAL + 1) + exponent, 0
        )
        tmp._lib.call["mpfr_prec_round"](
            tmp._as_mutable_ptr(), prec, mpfr_rounding_mode
        )
        result = get_flt(tmp)

    return result
