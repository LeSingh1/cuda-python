# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_cython_abi import EXT_SUFFIX, iter_public_extension_modules


def _make_package(root, relative_paths):
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_private_modules_are_skipped(tmp_path):
    build_dir = tmp_path / "cuda" / "bindings"
    _make_package(
        build_dir,
        [
            f"driver{EXT_SUFFIX}",
            f"_driver{EXT_SUFFIX}",
            f"_internal/utils{EXT_SUFFIX}",
            f"utils/version{EXT_SUFFIX}",
        ],
    )

    found = {p.relative_to(build_dir).as_posix() for p in iter_public_extension_modules(build_dir)}
    assert found == {f"driver{EXT_SUFFIX}", f"utils/version{EXT_SUFFIX}"}


def test_underscore_above_the_package_does_not_hide_modules(tmp_path):
    # manylinux installs Python under /opt/_internal, and GitHub Actions
    # containers check the repo out under /__w. Only the path inside the
    # package may mark a module private.
    build_dir = tmp_path / "_internal" / "site-packages" / "cuda" / "bindings"
    _make_package(build_dir, [f"driver{EXT_SUFFIX}", f"_internal/utils{EXT_SUFFIX}"])

    found = {p.relative_to(build_dir).as_posix() for p in iter_public_extension_modules(build_dir)}
    assert found == {f"driver{EXT_SUFFIX}"}
