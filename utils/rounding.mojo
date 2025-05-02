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

from memory import UnsafePointer
from sys import is_defined, llvm_intrinsic
from utils.numerics import FPUtils


@value
@register_passable("trivial")
struct RoundingMode(EqualityComparable, Stringable, Writable):
    var _value: Int32

    alias AWAY_FROM_ZERO = RoundingMode(Int32.MIN)
    alias INDETERMINATE = RoundingMode(-1)
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

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self is RoundingMode.AWAY_FROM_ZERO:
            writer.write("RoundingMode.AWAY_FROM_ZERO")
        elif self is RoundingMode.INDETERMINATE:
            writer.write("RoundingMode.INDETERMINATE")
        elif self is RoundingMode.TOWARD_ZERO:
            writer.write("RoundingMode.TOWARD_ZERO")
        elif self is RoundingMode.TO_NEAREST:
            writer.write("RoundingMode.TO_NEAREST")
        elif self is RoundingMode.UPWARD:
            writer.write("RoundingMode.UPWARD")
        elif self is RoundingMode.DOWNWARD:
            writer.write("RoundingMode.DOWNWARD")
        elif self is RoundingMode.TO_NEAREST_AWAY:
            writer.write("RoundingMode.TO_NEAREST_AWAY")
        else:
            writer.write("RoundingMode(", self.value(), ")")

    @always_inline("nodebug")
    fn value(self) -> Int32:
        return self._value


@always_inline("nodebug")
fn available_rounding_modes() -> List[RoundingMode, hint_trivial_type=True]:
    @parameter
    if is_defined["ALL_ROUNDING_MODES"]():
        return List[RoundingMode, hint_trivial_type=True](
            RoundingMode.TOWARD_ZERO,
            RoundingMode.TO_NEAREST,
            RoundingMode.UPWARD,
            RoundingMode.DOWNWARD,
        )
    else:
        return List[RoundingMode, hint_trivial_type=True](
            RoundingMode.TO_NEAREST,
        )


@always_inline("nodebug")
fn get_rounding_mode() -> RoundingMode:
    return RoundingMode(
        llvm_intrinsic["llvm.get.rounding", Int32, has_side_effect=False]()
    )


@always_inline("nodebug")
fn set_rounding_mode(rounding_mode: RoundingMode) -> Bool:
    llvm_intrinsic["llvm.set.rounding", NoneType](rounding_mode.value())

    if rounding_mode in [
        RoundingMode.TOWARD_ZERO,
        RoundingMode.TO_NEAREST,
        RoundingMode.UPWARD,
        RoundingMode.DOWNWARD,
        RoundingMode.TO_NEAREST_AWAY,
    ]:
        return rounding_mode == quick_get_rounding_mode()

    return rounding_mode == get_rounding_mode()


@always_inline
fn quick_get_rounding_mode[dtype: DType = DType.float32]() -> RoundingMode:
    alias ONE = Scalar[dtype](1.0)
    alias PREC = FPUtils[dtype].mantissa_width() + 1

    @parameter
    fn _get_eps() -> Scalar[dtype]:
        @parameter
        if dtype == DType.bfloat16:
            # The `math.ldexp` operation cannot be executed in compile-time
            # for `DType.bfloat16`
            return Scalar[dtype](0.0078125)

        return math.ldexp(ONE, 1 - PREC)

    alias EPS = _get_eps()
    alias EPSNEG = 0.5 * EPS
    alias ONE_PLUS_EPS = 1.0 + EPS

    # The following line forces a volatile load of `EPSNEG`, ensuring that
    # its value is actually read each time and preventing the compiler from
    # optimizing away the subsequent floating-point operations.
    var epsneg = UnsafePointer(to=EPSNEG).load[volatile=True]()
    var diff = (ONE_PLUS_EPS + epsneg) + (-1.0 - epsneg)

    if diff == 0.0:
        return RoundingMode.DOWNWARD

    if diff == EPS:
        return RoundingMode.TOWARD_ZERO

    if (2.0 + epsneg) == 2.0:
        if diff == EPSNEG:
            return RoundingMode.TO_NEAREST_AWAY

        return RoundingMode.TO_NEAREST

    if diff == EPSNEG:
        return RoundingMode.AWAY_FROM_ZERO

    return RoundingMode.UPWARD


struct RoundingContext:
    var _old_rounding_mode: RoundingMode
    var _new_rounding_mode: RoundingMode

    @always_inline
    fn __init__(out self, rounding_mode: RoundingMode):
        self._old_rounding_mode = RoundingMode.INDETERMINATE
        self._new_rounding_mode = rounding_mode

    @always_inline
    fn __enter__(mut self) raises:
        self._old_rounding_mode = get_rounding_mode()

        if self._old_rounding_mode != self._new_rounding_mode:
            if not set_rounding_mode(self._new_rounding_mode):
                raise String("unable to set '", self._new_rounding_mode, "'")

    @always_inline
    fn __exit__(self):
        if self._old_rounding_mode != self._new_rounding_mode:
            _ = set_rounding_mode(self._old_rounding_mode)
