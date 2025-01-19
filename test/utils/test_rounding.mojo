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

from testing import assert_equal

from rounding import (
    available_rounding_modes,
    quick_get_rounding_mode,
    RoundingContext,
)


fn test_rounding_modes() raises:
    alias ROUNDING_MODES = available_rounding_modes()

    for i in range(len(ROUNDING_MODES)):
        var rounding_mode = ROUNDING_MODES[i]

        with RoundingContext(rounding_mode):
            assert_equal(rounding_mode, quick_get_rounding_mode())
