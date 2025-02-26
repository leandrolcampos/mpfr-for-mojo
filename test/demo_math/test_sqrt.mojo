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

from sys import has_neon

from demo_math import sqrt

import mpfr

from checking import UnaryOperatorChecker
from rounding import available_rounding_modes


fn test_sqrt_bf16() raises:
    @parameter
    if has_neon():
        # bf16 is not supported for ARM architectures
        return

    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        alias ROUNDING_MODE = ROUNDING_MODES[i]

        var sqrt_checker = UnaryOperatorChecker[
            DType.bfloat16,
            mpfr.sqrt,
            sqrt,
            rounding_mode=ROUNDING_MODE,
            default_ulp_tolerance=0.5,
        ]()

        sqrt_checker.assert_special_values()
        sqrt_checker.assert_negative_normals[count=101]()
        sqrt_checker.assert_negative_subnormals[count=11]()
        sqrt_checker.assert_positive_subnormals[count=101]()
        sqrt_checker.assert_positive_normals[count=1_001]()


fn test_sqrt_fp16() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        alias ROUNDING_MODE = ROUNDING_MODES[i]

        var sqrt_checker = UnaryOperatorChecker[
            DType.float16,
            mpfr.sqrt,
            sqrt,
            rounding_mode=ROUNDING_MODE,
            default_ulp_tolerance=0.5,
        ]()

        sqrt_checker.assert_special_values()
        sqrt_checker.assert_negative_normals[count=101]()
        sqrt_checker.assert_negative_subnormals[count=11]()
        sqrt_checker.assert_positive_subnormals[count=1_001]()
        sqrt_checker.assert_positive_normals[count=1_001]()


fn test_sqrt_fp32() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        alias ROUNDING_MODE = ROUNDING_MODES[i]

        var sqrt_checker = UnaryOperatorChecker[
            DType.float32,
            mpfr.sqrt,
            sqrt,
            rounding_mode=ROUNDING_MODE,
            default_ulp_tolerance=0.5,
        ]()

        sqrt_checker.assert_special_values()
        sqrt_checker.assert_negative_normals[count=1_001]()
        sqrt_checker.assert_negative_subnormals[count=101]()
        sqrt_checker.assert_positive_subnormals[count=10_001]()
        sqrt_checker.assert_positive_normals[count=10_001]()


fn test_sqrt_fp64() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    @parameter
    for i in range(len(ROUNDING_MODES)):
        alias ROUNDING_MODE = ROUNDING_MODES[i]

        var sqrt_checker = UnaryOperatorChecker[
            DType.float64,
            mpfr.sqrt,
            sqrt,
            rounding_mode=ROUNDING_MODE,
            default_ulp_tolerance=0.5,
        ]()

        sqrt_checker.assert_special_values()
        sqrt_checker.assert_negative_normals[count=10_001]()
        sqrt_checker.assert_negative_subnormals[count=1_001]()
        sqrt_checker.assert_positive_subnormals[count=100_001]()
        sqrt_checker.assert_positive_normals[count=100_001]()
