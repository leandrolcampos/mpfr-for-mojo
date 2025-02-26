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
from mpfr._ops import (
    cmp,
    cmp_d,
    cmp_si_2exp,
    get_d,
    get_flt,
    set_d,
    set_flt,
    set_str,
)
from rounding import RoundingMode


fn get_mpfr_rounding_mode[mode: RoundingMode]() -> mpfr_rnd_t:
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
        type.is_floating_point(), "type must be a floating-point type"
    ]()
    constrained[
        type in [DType.bfloat16, DType.float16, DType.float32, DType.float64],
        "unsupported floating-point type",
    ]()


struct MpfrFloat[
    type: DType,
    rounding_mode: RoundingMode = RoundingMode.TO_NEAREST,
](Defaultable, ExplicitlyCopyable):
    alias _MPFR_PRECISION = _get_mpfr_precision[type]()
    alias _MPFR_ROUNDING_MODE = get_mpfr_rounding_mode[rounding_mode]()

    var _lib: MpfrLibrary
    var _value: mpfr_t

    @always_inline("nodebug")
    fn __init__(out self):
        _mpfr_float_construction_checks[type]()

        self._lib = MpfrLibraryProvider.get_or_create_ptr()[].library()
        self._value = mpfr_t()
        self._lib.call["mpfr_init2"](
            self.as_mutable_ptr(),
            Self._MPFR_PRECISION,
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
        self._lib.call["mpfr_clear"](self.as_mutable_ptr())

    @always_inline("nodebug")
    fn copy(self, out result: Self):
        result = Self()
        result._lib.call["mpfr_set"](
            result.as_mutable_ptr(),
            self.as_immutable_ptr(),
            Self._MPFR_ROUNDING_MODE,
        )

    @always_inline("nodebug")
    fn __setitem__(mut self, value: Scalar[type]):
        @parameter
        if type is DType.float64:
            _ = set_d(self, rebind[Scalar[DType.float64]](value))
        elif type is DType.float32:
            _ = set_flt(self, rebind[Scalar[DType.float32]](value))
        else:
            _ = set_flt(self, value.cast[DType.float32]())

    @always_inline
    fn __getitem__(self) -> Scalar[type]:
        @parameter
        if type is DType.float64:
            return rebind[Scalar[type]](get_d(self))
        elif type is DType.float32:
            return rebind[Scalar[type]](get_flt(self))
        else:
            return _get_value(self).cast[type]()

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) == 0

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) == 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) == 0

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) != 0

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) != 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) != 0

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) < 0

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) < 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) < 0

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) > 0

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) > 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) > 0

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) <= 0

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) <= 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) <= 0

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        if unlikely(self.is_nan() or other.is_nan()):
            return False

        return cmp(self, other) >= 0

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[type]) -> Bool:
        if unlikely(self.is_nan() or math.isnan(other)):
            return False

        @parameter
        if type is DType.float64:
            return cmp_d(self, rebind[Scalar[DType.float64]](other)) >= 0
        else:
            return cmp_d(self, other.cast[DType.float64]()) >= 0

    @always_inline("nodebug")
    fn get_exponent(self) -> mpfr_exp_t:
        return self._lib.call["mpfr_get_exp", mpfr_exp_t](
            self.as_immutable_ptr()
        )

    @always_inline("nodebug")
    fn get_sign(self) -> c_int:
        return self._lib.call["mpfr_sgn", c_int](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn is_inf(self) -> Bool:
        return self._lib.call["mpfr_inf_p", Bool](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn is_nan(self) -> Bool:
        return self._lib.call["mpfr_nan_p", Bool](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn is_number(self) -> Bool:
        return self._lib.call["mpfr_number_p", Bool](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn is_regular(self) -> Bool:
        return self._lib.call["mpfr_regular_p", Bool](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn is_singular(self) -> Bool:
        return not self.is_regular()

    @always_inline("nodebug")
    fn is_zero(self) -> Bool:
        return self._lib.call["mpfr_zero_p", Bool](self.as_immutable_ptr())

    @always_inline("nodebug")
    fn as_immutable_ptr(ref [ImmutableAnyOrigin]self) -> mpfr_srcptr:
        return Pointer.address_of(self._value)

    @always_inline("nodebug")
    fn as_mutable_ptr(ref [MutableAnyOrigin]self) -> mpfr_ptr:
        return Pointer.address_of(self._value)


@register_passable("trivial")
struct MpfrFloatPtr[type: DType, rounding_mode: RoundingMode]:
    """Represents a pointer to an MPFR value.

    This type is used for passing a pointer to an MPFR value into functions
    that require one.

    It is useful for avoiding diagnosis of "argument exclusivity" violations
    due to aliasing references, assuming that MPFR handles aliasing correctly.

    It is not intended to be used directly: when needed, a `MpfrFloat` object
    will be implicitly converted to a `MpfrFloatPtr` object.
    """

    alias _MPFR_PRECISION = _get_mpfr_precision[type]()
    alias _MPFR_ROUNDING_MODE = get_mpfr_rounding_mode[rounding_mode]()

    var _lib: MpfrLibrary
    var _as_immutable: mpfr_srcptr
    var _as_mutable: mpfr_ptr

    @implicit
    @always_inline("nodebug")
    fn __init__(out self, mut src: MpfrFloat[type, rounding_mode]):
        self._lib = src._lib
        self._as_immutable = src.as_immutable_ptr()
        self._as_mutable = src.as_mutable_ptr()

    @always_inline("nodebug")
    fn as_immutable(self) -> mpfr_srcptr:
        return self._as_immutable

    @always_inline("nodebug")
    fn as_mutable(self) -> mpfr_ptr:
        return self._as_mutable


@always_inline("nodebug")
fn _eval[T: AnyType, //, val: T]() -> T:
    return val


@always_inline
fn _get_value(src: MpfrFloat) -> Float32:
    alias INF = math.inf[DType.float32]()
    alias ONE = Scalar[DType.float32](1.0)
    alias PREC = FPUtils[src.type].mantissa_width() + 1
    alias EMAX = FPUtils[src.type].max_exponent() - 1
    alias EMIN_NORMAL = 1 - EMAX
    alias EMIN_SUBNORMAL = EMIN_NORMAL + 1 - PREC
    alias EPSILON = math.ldexp(ONE, -PREC + 1)
    alias EPSILON_NEG = math.ldexp(ONE, -PREC)
    alias SMALLEST_SUBNORMAL = EPSILON * math.ldexp(ONE, EMIN_NORMAL)
    alias MAX_FINITE = math.ldexp(2.0 * (1.0 - EPSILON_NEG), EMAX)

    if unlikely(src.is_singular()):
        return get_flt(src)

    var mpfr_rounding_mode = src._MPFR_ROUNDING_MODE
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
            tmp.as_mutable_ptr(), prec, mpfr_rounding_mode
        )
        result = get_flt(tmp)

    return result
