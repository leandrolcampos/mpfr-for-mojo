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

from float_dtypes import available_float_dtypes, available_half_float_dtypes


fn test_float_dtypes() raises:
    alias FLOAT_DTYPES = available_float_dtypes()

    var num_dtypes = len(FLOAT_DTYPES)
    assert_true(num_dtypes > 0, "no floating-point data types available")

    for i in range(num_dtypes):
        var dtype = FLOAT_DTYPES[i]
        assert_true(
            dtype.is_floating_point(),
            String(dtype, " is not a floating-point data type"),
        )


fn test_half_float_dtypes() raises:
    alias HALF_FLOAT_DTYPES = available_half_float_dtypes()

    var num_dtypes = len(HALF_FLOAT_DTYPES)
    assert_true(
        num_dtypes > 0, "no half-precision floating-point data types available"
    )

    for i in range(num_dtypes):
        var dtype = HALF_FLOAT_DTYPES[i]
        assert_true(
            dtype.is_half_float(),
            String(dtype, " is not a half-precision floating-point data type"),
        )
