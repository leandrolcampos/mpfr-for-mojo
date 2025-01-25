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

from testing import assert_true

from float_types import available_float_types, available_half_float_types


fn test_float_types() raises:
    alias FLOAT_TYPES = available_float_types()

    var num_types = len(FLOAT_TYPES)
    assert_true(num_types > 0, "no floating-point types available")

    for i in range(num_types):
        var type = FLOAT_TYPES[i]
        assert_true(
            type.is_floating_point(),
            str(type) + " is not a floating-point type",
        )


fn test_half_float_types() raises:
    alias HALF_FLOAT_TYPES = available_half_float_types()

    var num_types = len(HALF_FLOAT_TYPES)
    assert_true(
        num_types > 0, "no half-precision floating-point types available"
    )

    for i in range(num_types):
        var type = HALF_FLOAT_TYPES[i]
        assert_true(
            type.is_half_float(),
            str(type) + " is not a half-precision floating-point type",
        )
