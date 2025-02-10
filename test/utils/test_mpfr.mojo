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

from float_types import available_half_float_types
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
    alias FLOAT_TYPES = available_half_float_types()

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


fn _get_expected_values() -> List[Float64, hint_trivial_type=True]:
    alias FP32_EMAX = FPUtils[DType.float32].max_exponent() - 1
    alias FP32_PREC = FPUtils[DType.float32].mantissa_width() + 1

    return List[Float64, hint_trivial_type=True](
        math.nan[DType.float64](),
        Float64.MIN,  # -inf
        Float64.MAX,  # +inf
        math.ldexp(-1.0, FP32_EMAX + 1),  # -huge
        math.ldexp(+1.0, FP32_EMAX + 1),  # +huge
        1.0 + math.ldexp(1.0, -FP32_PREC),
    )


fn _get_actual_values() -> List[Float32, hint_trivial_type=True]:
    alias FP32_PREC = FPUtils[DType.float32].mantissa_width() + 1

    return List[Float32, hint_trivial_type=True](
        math.nan[DType.float32](),
        Float32.MIN,  # -inf
        Float32.MAX,  # +inf
        1.0 + math.ldexp(Scalar[DType.float32](1.0), 1 - FP32_PREC),
    )


fn _test_ulp_error[rounding_mode: RoundingMode]() raises:
    alias ERROR_MESSAGE = (
        "incorrect ULP error for ('"
        + str(rounding_mode)
        + "', '"
        + str(DType.float32)
        + "')"
    )

    var expected_values = _get_expected_values()
    var actual_values = _get_actual_values()

    var expected = MpfrFloat[DType.float32, rounding_mode]()
    var output = __type_of(expected)()

    for i in range(len(expected_values)):
        for j in range(len(actual_values)):
            _ = mpfr.set_d(expected, expected_values[i])
            var actual = actual_values[j]

            mpfr.ulp_error(output, expected, actual)
            assert_true(
                mpfr.cmp_d(expected, expected_values[i]) == 0,
                msg="The `expected` variable was mutated.",
            )

            if expected.is_nan() and math.isnan(actual):
                assert_equal(output[], 0.0, ERROR_MESSAGE)
            elif expected.is_nan() or math.isnan(actual):
                assert_equal(output[], Float32.MAX, ERROR_MESSAGE)
            elif expected.is_inf() or math.isinf(actual):
                assert_equal(
                    output[],
                    0.0 if expected[] == actual else Float32.MAX,
                    ERROR_MESSAGE,
                )
            elif expected < Float32.MIN_FINITE:
                # output = abs(-huge - actual) / ulp(-huge)
                # ulp(-huge) = 2**(128 - 24 + 1)
                @parameter
                if rounding_mode is RoundingMode.UPWARD:
                    assert_equal(output[], 8388609, ERROR_MESSAGE)
                else:
                    assert_equal(output[], 8388608, ERROR_MESSAGE)
            elif expected > Float32.MAX_FINITE:
                # output = abs(+huge - actual) / ulp(+huge)
                # ulp(+huge) = 2**(128 - 24 + 1)
                @parameter
                if rounding_mode in [
                    RoundingMode.TO_NEAREST,
                    RoundingMode.UPWARD,
                ]:
                    assert_equal(output[], 8388608, ERROR_MESSAGE)
                else:
                    assert_equal(output[], 8388607.5, ERROR_MESSAGE)
            else:
                assert_equal(output[], 0.5, ERROR_MESSAGE)


fn _test_ulp_error_with_aliasing[rounding_mode: RoundingMode]() raises:
    alias ERROR_MESSAGE = (
        "incorrect ULP error for ('"
        + str(rounding_mode)
        + "', '"
        + str(DType.float32)
        + "')"
    )

    var expected_values = _get_expected_values()
    var actual_values = _get_actual_values()

    var expected = MpfrFloat[DType.float32, rounding_mode]()
    var output = __type_of(expected)()

    for i in range(len(expected_values)):
        for j in range(len(actual_values)):
            _ = mpfr.set_d(output, expected_values[i])
            _ = mpfr.set_d(expected, expected_values[i])
            var actual = actual_values[j]

            mpfr.ulp_error(output, output, actual)

            if expected.is_nan() and math.isnan(actual):
                assert_equal(output[], 0.0, ERROR_MESSAGE)
            elif expected.is_nan() or math.isnan(actual):
                assert_equal(output[], Float32.MAX, ERROR_MESSAGE)
            elif expected.is_inf() or math.isinf(actual):
                assert_equal(
                    output[],
                    0.0 if expected[] == actual else Float32.MAX,
                    ERROR_MESSAGE,
                )
            elif expected < Float32.MIN_FINITE:
                # output = abs(-huge - actual) / ulp(-huge)
                # ulp(-huge) = 2**(128 - 24 + 1)
                @parameter
                if rounding_mode is RoundingMode.UPWARD:
                    assert_equal(output[], 8388609, ERROR_MESSAGE)
                else:
                    assert_equal(output[], 8388608, ERROR_MESSAGE)
            elif expected > Float32.MAX_FINITE:
                # output = abs(+huge - actual) / ulp(+huge)
                # ulp(+huge) = 2**(128 - 24 + 1)
                @parameter
                if rounding_mode in [
                    RoundingMode.TO_NEAREST,
                    RoundingMode.UPWARD,
                ]:
                    assert_equal(output[], 8388608, ERROR_MESSAGE)
                else:
                    assert_equal(output[], 8388607.5, ERROR_MESSAGE)
            else:
                assert_equal(output[], 0.5, ERROR_MESSAGE)


fn test_ulp_error() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        _test_ulp_error[ROUNDING_MODES[i]]()
        _test_ulp_error_with_aliasing[ROUNDING_MODES[i]]()
