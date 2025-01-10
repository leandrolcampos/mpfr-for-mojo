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

from testing import assert_equal, assert_true

from mpfr_wrapper import add, MpfrFloat, RoundingMode
from mpfr_wrapper._library import MpfrLibrary


fn test_version() raises:
    var lib = MpfrLibrary()
    var version = lib.get_version()

    assert_true(
        version.major() == 4 and version.minor() == 2,
        "The MPFR version should be 4.2.x; got " + str(version),
    )


fn test_add() raises:
    var a = MpfrFloat[DType.float32](1.0)
    var b = MpfrFloat[DType.float32](3.0)
    _ = add(a, a, b)

    assert_equal(a[], 4.0)
