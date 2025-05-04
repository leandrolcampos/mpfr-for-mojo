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

from collections import InlineArray
from collections.string import StringSlice
from memory import UnsafePointer
from os import abort, getenv
from pathlib import Path
from sys import DLHandle, os_is_linux, os_is_macos
from sys.ffi import _Global, c_char


@value
@register_passable("trivial")
struct MpfrVersion(Stringable, Writable):
    """Represents a MPFR version with major, minor, and patch numbers."""

    var _major: Int
    var _minor: Int
    var _patch: Int

    fn __init__(out self, version: String):
        """Initializes a `MpfrVersion` object from a string.

        The string is parsed to extract major, minor, and patch numbers. If
        parsing fails for any component, it defaults to -1.

        Args:
            version: A string representing the MPFR version (e.g., "4.2.1").
        """
        var splits: List[String]
        var components = InlineArray[Int, 3](-1)

        try:
            splits = version.split(".")
        except:
            splits = List[String]()

        for i in range(min(3, len(splits))):
            try:
                components[i] = atol(splits[i])
            except:
                components[i] = -1

        self = MpfrVersion(components[0], components[1], components[2])

    fn __str__(self) -> String:
        """Gets a string representation of the MPFR version."""
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats the MPFR version to the provided writer.

        Parameters:
            W: A type conforming to the `Writer` trait.

        Args:
            writer: The object to write to.
        """
        writer.write(self._major, ".", self._minor, ".", self._patch)

    @always_inline("nodebug")
    fn major(self) -> Int:
        """Gets the major version number."""
        return self._major

    @always_inline("nodebug")
    fn minor(self) -> Int:
        """Gets the minor version number."""
        return self._minor

    @always_inline("nodebug")
    fn patch(self) -> Int:
        """Gets the patch version number."""
        return self._patch


@always_inline("nodebug")
fn _get_mpfr_version(mpfr_handle: DLHandle) -> String:
    return String(
        StringSlice[StaticConstantOrigin](
            unsafe_from_utf8_ptr=mpfr_handle.call[
                "mpfr_get_version", UnsafePointer[c_char]
            ]()
        )
    )


@value
@register_passable("trivial")
struct MpfrLibrary:
    var _handle: DLHandle

    fn __init__(out self):
        var file_name: String

        @parameter
        if os_is_linux():
            file_name = "libmpfr.so"
        else:
            constrained[os_is_macos(), "unsupported operating system"]()
            file_name = "libmpfr.dylib"

        var file_dir = getenv("MPFR_LIBRARY_DIR")

        if len(file_dir) == 0:
            self = MpfrLibrary(file_name)
        else:
            self = MpfrLibrary(String(Path(file_dir) / file_name))

    fn __init__(out self, path: String):
        try:
            self._handle = DLHandle(path)
        except e:
            self._handle = abort[DLHandle](
                "Failed to load GNU MPFR library from ", path, ":\n", e
            )

    @always_inline("nodebug")
    fn call[
        name: StringLiteral,
        return_type: AnyTrivialRegType = NoneType,
        *T: AnyType,
    ](self, *args: *T) -> return_type:
        return self._handle.call[name, return_type](args)

    fn close(mut self):
        self._handle.close()

    fn free_cache(self):
        self.call["mpfr_free_cache"]()

    fn get_version(self) -> MpfrVersion:
        return MpfrVersion(_get_mpfr_version(self._handle))


alias MpfrLibraryProvider = _Global[
    "MpfrLibrary", MpfrLibraryGuard, _init_mpfr_library
]
"""Provides access to a per-thread, global MPFR library instance."""


fn _init_mpfr_library() -> MpfrLibraryGuard:
    return MpfrLibraryGuard()


struct MpfrLibraryGuard:
    """Guards a per-thread, global MPFR library instance."""

    var _lib: MpfrLibrary

    fn __init__(out self):
        self._lib = MpfrLibrary()

    fn __moveinit__(out self, owned other: Self):
        self._lib = other._lib

    fn __del__(owned self):
        self._lib.free_cache()
        self._lib.close()

    @always_inline("nodebug")
    fn library(self) -> MpfrLibrary:
        return self._lib
