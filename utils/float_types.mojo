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


@always_inline("nodebug")
fn available_float_types() -> List[DType, hint_trivial_type=True]:
    var result = List[DType, hint_trivial_type=True](
        DType.float16,
        DType.float32,
        DType.float64,
    )

    @parameter
    if not has_neon():
        result.append(DType.bfloat16)

    return result


@always_inline("nodebug")
fn available_half_float_types() -> List[DType, hint_trivial_type=True]:
    var result = List[DType, hint_trivial_type=True]()
    var float_types = available_float_types()

    for i in range(len(float_types)):
        if float_types[i].is_half_float():
            result.append(float_types[i])

    return result
