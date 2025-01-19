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

from memory import UnsafePointer
from sys.ffi import c_char, c_int, c_long


alias mpfr_rnd_t = c_int

alias MPFR_RNDN: mpfr_rnd_t = 0
alias MPFR_RNDZ: mpfr_rnd_t = 1
alias MPFR_RNDU: mpfr_rnd_t = 2
alias MPFR_RNDD: mpfr_rnd_t = 3
alias MPFR_RNDA: mpfr_rnd_t = 4
alias MPFR_RNDF: mpfr_rnd_t = 5

alias mpfr_exp_t = c_long
alias mpfr_prec_t = c_long
alias mpfr_sign_t = c_int

alias mp_limb_t = UInt64


struct mpfr_t:
    var _mpfr_prec: mpfr_prec_t
    var _mpfr_sign: mpfr_sign_t
    var _mpfr_exp: mpfr_exp_t
    var _mpfr_d: UnsafePointer[mp_limb_t]

    @always_inline("nodebug")
    fn __init__(out self):
        self._mpfr_prec = mpfr_prec_t()
        self._mpfr_sign = mpfr_sign_t()
        self._mpfr_exp = mpfr_exp_t()
        self._mpfr_d = UnsafePointer[mp_limb_t]()


alias mpfr_ptr = Pointer[mpfr_t, MutableAnyOrigin]
alias mpfr_srcptr = Pointer[mpfr_t, ImmutableAnyOrigin]
