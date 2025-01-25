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

from testing import assert_equal

from demo_math import sqrt
from float_types import available_float_types


fn _test_simple_cases[type: DType]() raises:
    alias InType = Scalar[type]
    alias INF = math.inf[type]()
    alias NAN = math.nan[type]()

    assert_equal(sqrt(NAN), NAN)
    assert_equal(sqrt(-INF), NAN)
    assert_equal(sqrt(INF), INF)
    assert_equal(sqrt(InType(-0)), -0)
    assert_equal(sqrt(InType(0)), 0)
    assert_equal(sqrt(InType(-1)), NAN)
    assert_equal(sqrt(InType(1)), 1)
    assert_equal(sqrt(InType(4)), 2)
    assert_equal(sqrt(InType(9)), 3)


fn test_simple_cases() raises:
    alias FLOAT_TYPES = available_float_types()

    @parameter
    for i in range(len(FLOAT_TYPES)):
        _test_simple_cases[FLOAT_TYPES[i]]()
