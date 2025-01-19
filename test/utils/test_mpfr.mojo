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

from collections import InlineArray
from sys import has_neon
from testing import assert_equal, assert_true
from utils.numerics import FPUtils

import mpfr

from mpfr import MpfrFloat
from mpfr._library import MpfrLibrary
from rounding import (
    available_rounding_modes,
    quick_get_rounding_mode,
    RoundingContext,
    RoundingMode,
)


fn test_version() raises:
    var lib = MpfrLibrary()
    var version = lib.get_version()

    assert_true(
        version.major() == 4 and version.minor() == 2,
        "The MPFR version should be 4.2.x; got " + str(version),
    )


fn test_copy() raises:
    var a = MpfrFloat[DType.float32](1.0)
    var b = a.copy()

    assert_equal(a[], b[])

    b[] = 2.0

    assert_equal(a[], 1.0)
    assert_equal(b[], 2.0)


fn test_add() raises:
    var a = MpfrFloat[DType.float32](1.0)
    var b = MpfrFloat[DType.float32](3.0)
    _ = mpfr.add(a, a, b)

    assert_equal(a[], 4.0)


fn _append_singular_values[
    type: DType, //
](mut values: List[Scalar[type], hint_trivial_type=True],):
    alias NAN = math.nan[type]()
    alias INF = math.inf[type]()

    values.append(NAN)
    values.append(INF)
    values.append(-INF)
    values.append(0.0)
    values.append(-0.0)


fn _append_boundary_values[
    type: DType
](mut values: List[Float64, hint_trivial_type=True]):
    alias INF = math.inf[DType.float64]()
    alias PREC = FPUtils[type].mantissa_width() + 1
    alias EMAX = FPUtils[type].max_exponent() - 1
    alias EMIN_NORMAL = 1 - EMAX
    alias EPSILON = math.ldexp(1.0, -PREC + 1)
    alias EPSILON_NEG = math.ldexp(1.0, -PREC)
    alias SMALLEST_SUBNORMAL = EPSILON * math.ldexp(1.0, EMIN_NORMAL)
    alias MAX_FINITE = math.ldexp(2.0 * (1.0 - EPSILON_NEG), EMAX)

    var boundary_values = List[Float64, hint_trivial_type=True](
        2.0 * MAX_FINITE,
        math.nextafter(MAX_FINITE, INF),
        MAX_FINITE,
        math.nextafter(MAX_FINITE, 0.0),
        math.nextafter(SMALLEST_SUBNORMAL, 1.0),
        SMALLEST_SUBNORMAL,
        math.nextafter(SMALLEST_SUBNORMAL, 0.0),
        0.50 * SMALLEST_SUBNORMAL,
        0.25 * SMALLEST_SUBNORMAL,
    )

    for i in range(len(boundary_values)):
        var value = boundary_values[i]
        values.append(value)
        values.append(-value)


fn _get_intermediate_values[
    type: DType, //
](lower_bound: Scalar[type], upper_bound: Scalar[type]) -> InlineArray[
    Float64, 4
]:
    var base_value = lower_bound.cast[DType.float64]()
    var delta = 0.25 * (upper_bound.cast[DType.float64]() - base_value)
    var result = InlineArray[Float64, 4](unsafe_uninitialized=True)

    @parameter
    for i in range(result.size):
        result[i] = base_value + i * delta

    return result


fn _append_in_range_values[
    type: DType
](mut values: List[Float64, hint_trivial_type=True]):
    alias ONE = Scalar[type](1.0)
    alias EMAX = FPUtils[type].max_exponent() - 1
    alias EMIN_NORMAL = 1 - EMAX

    # The `math.ldexp` operation cannot be executed in compile-time
    # for `DType.bfloat16`
    var smallest_normal = math.ldexp(ONE, EMIN_NORMAL)

    # The `math.nextafter` operation cannot be executed for `DType.bfloat16`
    # and `DType.float16`
    fn _next_after[*, upward: Bool](value: Scalar[type]) -> Scalar[type]:
        # This function assumes that `value` is a regular(, non-zero) value
        var as_int = FPUtils[type].bitcast_to_integer(value)
        var inc = 1 if value >= 0.0 else -1

        @parameter
        if upward:
            return FPUtils[type].bitcast_from_integer(as_int + inc)

        return FPUtils[type].bitcast_from_integer(as_int - inc)

    var lower_bounds = List[Scalar[type], hint_trivial_type=True](
        -Scalar[type].MAX_FINITE,
        -_next_after[upward=True](ONE),
        -ONE,
        -_next_after[upward=True](smallest_normal),
        -smallest_normal,
    )

    for i in range(len(lower_bounds)):
        var lower_bound = lower_bounds[i]
        var upper_bound = _next_after[upward=True](lower_bound)
        var intermediate_values = _get_intermediate_values(
            lower_bound, upper_bound
        )

        for j in range(intermediate_values.size):
            var value = intermediate_values[j]
            values.append(value)
            values.append(-value)


fn _test_get_value[rounding_mode: RoundingMode]() raises:
    @parameter
    fn _get_supported_types() -> List[DType, hint_trivial_type=True]:
        var result = List[DType, hint_trivial_type=True](DType.float16)

        @parameter
        if not has_neon():
            result.append(DType.bfloat16)

        return result

    alias FLOAT_TYPES = _get_supported_types()

    @parameter
    for i in range(len(FLOAT_TYPES)):
        alias ERROR_MESSAGE = (
            "incorrect rounding for ('"
            + str(rounding_mode)
            + "', '"
            + str(FLOAT_TYPES[i])
            + "')"
        )

        var singular_values = List[
            Scalar[FLOAT_TYPES[i]], hint_trivial_type=True
        ]()
        _append_singular_values(singular_values)

        var regular_values = List[Float64, hint_trivial_type=True]()
        _append_boundary_values[FLOAT_TYPES[i]](regular_values)
        _append_in_range_values[FLOAT_TYPES[i]](regular_values)

        var a = MpfrFloat[FLOAT_TYPES[i], rounding_mode]()

        with RoundingContext(rounding_mode):
            if rounding_mode != quick_get_rounding_mode[FLOAT_TYPES[i]]():
                # It means that the active rounding mode is not supported
                # by `FLOAT_TYPES[i]` in the current target
                continue

            for j in range(len(singular_values)):
                var singular_value = singular_values[j]
                a[] = singular_value
                assert_equal(a[], singular_value, ERROR_MESSAGE)

            for j in range(len(regular_values)):
                var value = regular_values[j]
                _ = mpfr.set_d(a, value)
                assert_equal(a[], Scalar[FLOAT_TYPES[i]](value), ERROR_MESSAGE)


fn test_get_value() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        _test_get_value[ROUNDING_MODES[i]]()
