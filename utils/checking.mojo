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

from builtin._location import __call_location, _SourceLocation
from builtin.dtype import _unsigned_integral_type_of
from builtin.format_int import hex
from memory import bitcast
from sys.ffi import c_int
from utils.numerics import FPUtils

import demo_math
import mpfr

from mpfr import MpfrFloat, MpfrFloatPtr
from rounding import RoundingContext, RoundingMode


alias UnaryMpfrOperator = fn[type: DType, rounding_mode: RoundingMode] (
    MpfrFloatPtr[type, rounding_mode],
    MpfrFloat[type, rounding_mode],
) -> c_int

alias UnaryOperator = fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
    type, width
]


@register_passable("trivial")
struct _MatchResult[type: DType](Boolable):
    var _cr_expected: Scalar[type]
    var _actual: Scalar[type]
    var _ulp_error: Float64
    var _success: Bool

    @always_inline("nodebug")
    fn __init__(out self, cr_expected: Scalar[type], actual: Scalar[type]):
        self._cr_expected = cr_expected
        self._actual = actual
        self._ulp_error = 0.0
        self._success = cr_expected == actual

    @always_inline("nodebug")
    fn set_ulp_error(mut self, ulp_error: Float64, is_tolerable: Bool):
        self._ulp_error = ulp_error
        self._success = (self._cr_expected == self._actual) or is_tolerable

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return self._success

    @always_inline("nodebug")
    fn cr_expected(self) -> Scalar[type]:
        return self._cr_expected

    @always_inline("nodebug")
    fn actual(self) -> Scalar[type]:
        return self._actual

    @always_inline("nodebug")
    fn ulp_error(self) -> Float64:
        return self._ulp_error


struct UnaryOperatorChecker[
    type: DType,
    oracle_fn: UnaryMpfrOperator,
    tested_fn: UnaryOperator,
    *,
    rounding_mode: RoundingMode = RoundingMode.TO_NEAREST,
    default_ulp_tolerance: Float64 = 1.0,
]:
    alias MAX_FINITE = Scalar[type].MAX_FINITE
    alias SMALLEST_NORMAL = _smallest_normal[type]()
    alias SMALLEST_SUBNORMAL = _smallest_subnormal[type]()
    alias UINT_TYPE = _unsigned_integral_type_of[type]()

    var _rounding_context: RoundingContext

    @always_inline
    fn __init__(out self) raises:
        self._rounding_context = RoundingContext(rounding_mode)
        self._rounding_context.__enter__()

    @always_inline
    fn __del__(owned self):
        self._rounding_context.__exit__()

    @always_inline
    fn assert_match(
        self,
        /,
        x: Scalar[type],
        *,
        ulp_tolerance: Float64 = default_ulp_tolerance,
    ) raises:
        var mpfr_scratch = MpfrFloat[type, rounding_mode]()
        var match_result = _match[oracle_fn, tested_fn](
            x, ulp_tolerance, mpfr_scratch
        )

        if not match_result:
            raise _assert_error[rounding_mode](
                x, ulp_tolerance, match_result, __call_location()
            )

    @always_inline
    fn assert_special_values(
        self,
        /,
        *,
        ulp_tolerance: Float64 = default_ulp_tolerance,
    ) raises:
        alias VALUES = List[Scalar[type], hint_trivial_type=True](
            0.0, -0.0, math.inf[type](), -math.inf[type](), math.nan[type]()
        )

        var mpfr_scratch = MpfrFloat[type, rounding_mode]()

        @parameter
        for i in range(len(VALUES)):
            alias VALUE = VALUES[i]

            var match_result = _match[oracle_fn, tested_fn](
                VALUE, ulp_tolerance, mpfr_scratch
            )

            if not match_result:
                raise _assert_error[rounding_mode](
                    VALUE, ulp_tolerance, match_result, __call_location()
                )

    @always_inline
    fn _unsafe_assert_range[
        start: Scalar[Self.UINT_TYPE],
        stop: Scalar[Self.UINT_TYPE],
        *,
        endpoint: Bool = True,
        count: Scalar[Self.UINT_TYPE] = _range_length[start, stop, endpoint](),
    ](self, ulp_tolerance: Float64, location: _SourceLocation,) raises:
        alias RANGE_LENGTH = _range_length[start, stop, endpoint]()

        constrained[RANGE_LENGTH != 0, "start and stop must be different"]()
        constrained[
            count > (1 if endpoint else 0),
            (
                "count must be greater than 1 if endpoint is included, "
                "or greater than 0 otherwise"
            ),
        ]()
        constrained[
            count <= RANGE_LENGTH,
            "count must be less than or equal to the range length",
        ]()

        alias COUNT = count - 1 if endpoint else count
        alias STEP = RANGE_LENGTH / COUNT

        @always_inline
        fn _assert(
            value: Scalar[Self.UINT_TYPE],
            mut mpfr_scratch: MpfrFloat[type, rounding_mode],
        ) raises:
            var x = bitcast[type](value)

            var match_result = _match[oracle_fn, tested_fn](
                x, ulp_tolerance, mpfr_scratch
            )

            if not match_result:
                raise _assert_error[rounding_mode](
                    x, ulp_tolerance, match_result, location
                )

        @always_inline
        fn _next(current: Scalar[Self.UINT_TYPE]) -> Scalar[Self.UINT_TYPE]:
            @parameter
            if start < stop:
                return current + STEP
            else:
                return current - STEP

        var current = start
        var mpfr_scratch = MpfrFloat[type, rounding_mode]()

        for _ in range(COUNT):
            _assert(current, mpfr_scratch)
            current = _next(current)

        @parameter
        if endpoint:
            _assert(stop, mpfr_scratch)

    @always_inline
    fn assert_range[
        start: Scalar[type],
        stop: Scalar[type],
        *,
        endpoint: Bool = True,
        count: Scalar[Self.UINT_TYPE] = _range_length[
            bitcast[Self.UINT_TYPE](start),
            bitcast[Self.UINT_TYPE](stop),
            endpoint,
        ](),
    ](self, *, ulp_tolerance: Float64 = default_ulp_tolerance) raises:
        self._unsafe_assert_range[
            bitcast[Self.UINT_TYPE](start),
            bitcast[Self.UINT_TYPE](stop),
            endpoint=endpoint,
            count=count,
        ](ulp_tolerance, __call_location())

    @always_inline
    fn assert_positive_normals[
        *,
        count: Scalar[Self.UINT_TYPE] = _range_length[
            bitcast[Self.UINT_TYPE](Self.SMALLEST_NORMAL),
            bitcast[Self.UINT_TYPE](Self.MAX_FINITE),
            True,
        ](),
    ](self, *, ulp_tolerance: Float64 = default_ulp_tolerance) raises:
        self._unsafe_assert_range[
            bitcast[Self.UINT_TYPE](Self.SMALLEST_NORMAL),
            bitcast[Self.UINT_TYPE](Self.MAX_FINITE),
            endpoint=True,
            count=count,
        ](ulp_tolerance, __call_location())

    @always_inline
    fn assert_negative_normals[
        *,
        count: Scalar[Self.UINT_TYPE] = _range_length[
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_NORMAL),
            bitcast[Self.UINT_TYPE](-Self.MAX_FINITE),
            True,
        ](),
    ](self, *, ulp_tolerance: Float64 = default_ulp_tolerance) raises:
        self._unsafe_assert_range[
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_NORMAL),
            bitcast[Self.UINT_TYPE](-Self.MAX_FINITE),
            endpoint=True,
            count=count,
        ](ulp_tolerance, __call_location())

    @always_inline
    fn assert_positive_subnormals[
        *,
        count: Scalar[Self.UINT_TYPE] = _range_length[
            bitcast[Self.UINT_TYPE](Self.SMALLEST_SUBNORMAL),
            bitcast[Self.UINT_TYPE](Self.SMALLEST_NORMAL) - 1,
            True,
        ](),
    ](self, *, ulp_tolerance: Float64 = default_ulp_tolerance) raises:
        self._unsafe_assert_range[
            bitcast[Self.UINT_TYPE](Self.SMALLEST_SUBNORMAL),
            bitcast[Self.UINT_TYPE](Self.SMALLEST_NORMAL) - 1,
            endpoint=True,
            count=count,
        ](ulp_tolerance, __call_location())

    @always_inline
    fn assert_negative_subnormals[
        *,
        count: Scalar[Self.UINT_TYPE] = _range_length[
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_SUBNORMAL),
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_NORMAL) - 1,
            True,
        ](),
    ](self, *, ulp_tolerance: Float64 = default_ulp_tolerance) raises:
        self._unsafe_assert_range[
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_SUBNORMAL),
            bitcast[Self.UINT_TYPE](-Self.SMALLEST_NORMAL) - 1,
            endpoint=True,
            count=count,
        ](ulp_tolerance, __call_location())


@always_inline
fn _smallest_subnormal[type: DType]() -> Scalar[type]:
    return FPUtils[type].bitcast_from_integer(1)


@always_inline
fn _smallest_normal[type: DType]() -> Scalar[type]:
    return FPUtils[type].bitcast_from_integer(
        1 << FPUtils[type].mantissa_width()
    )


@always_inline
fn _range_length[
    type: DType, //, start: Scalar[type], stop: Scalar[type], endpoint: Bool
]() -> Scalar[type]:
    @parameter
    if start < stop:
        return stop - start + (1 if endpoint else 0)
    else:
        return start - stop + (1 if endpoint else 0)


@always_inline
fn _match[
    type: DType,
    rounding_mode: RoundingMode, //,
    oracle_fn: UnaryMpfrOperator,
    tested_fn: UnaryOperator,
](
    x: Scalar[type],
    ulp_tolerance: Float64,
    mut mpfr_scratch: MpfrFloat[type, rounding_mode],
) -> _MatchResult[type]:
    mpfr_scratch[] = x
    _ = oracle_fn(mpfr_scratch, mpfr_scratch)

    var result = _MatchResult(mpfr_scratch[], tested_fn(x))

    if not result:
        mpfr.ulp_error(mpfr_scratch, mpfr_scratch, result.actual())
        ulp_error = mpfr.get_d[RoundingMode.AWAY_FROM_ZERO](mpfr_scratch)
        result.set_ulp_error(ulp_error, ulp_error <= ulp_tolerance)

    return result


@always_inline
fn _assert_error[
    type: DType, //,
    rounding_mode: RoundingMode,
](
    x: Scalar[type],
    ulp_tolerance: Float64,
    match_result: _MatchResult[type],
    location: _SourceLocation,
) -> String:
    alias UINT_TYPE = _unsigned_integral_type_of[type]()
    alias TYPE_AS_STRING = String(type)
    alias ROUNDING_MODE_AS_STRING = String(rounding_mode)

    var error_message = String(
        "AssertionError",
        "\n  - Type:          ",
        TYPE_AS_STRING,
        "\n  - Rounding mode: ",
        ROUNDING_MODE_AS_STRING,
        "\n  - Input:         ",
        hex(bitcast[UINT_TYPE](x)),
        "\n  - Expected:      ",
        match_result.cr_expected(),
        " (correctly rounded)",
        "\n  - Actual:        ",
        match_result.actual(),
        "\n  - ULP error:     ",
        match_result.ulp_error(),
        "\n  - ULP tolerance: ",
        ulp_tolerance,
    )

    return location.prefix(String.write(error_message))
