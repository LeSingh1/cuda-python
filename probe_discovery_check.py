# Fork-only probe helper (never merged upstream).
#
# The build backends discover Cython extensions by string-slicing glob results.
# The pathlib migration rewrites that discovery, so this script re-implements
# BOTH the pre-migration and post-migration versions and asserts they produce
# identical results against the real source tree.

import glob
import os
import sys
from pathlib import Path

FAILURES = []


def check(label, old, new):
    status = "OK  " if old == new else "FAIL"
    print(f"[{status}] {label}: {len(old)} item(s)")
    if old != new:
        FAILURES.append(label)
        print(f"  old: {old}")
        print(f"  new: {new}")


# --------------------------------------------------------------------------
# cuda_core/build_hooks.py :: _build_cuda_core.module_names / get_sources
# --------------------------------------------------------------------------
os.chdir("cuda_core")

_posix_only_modules = frozenset({"_utils/_wsl_locale"})


def core_old():
    root_path = os.path.sep.join(["cuda", "core", ""])
    for filename in glob.glob(f"{root_path}/**/*.pyx", recursive=True):
        mod = filename[len(root_path) : -4]
        if sys.platform == "win32" and mod.replace(os.path.sep, "/") in _posix_only_modules:
            continue
        yield mod


def core_new():
    root_path = Path("cuda", "core")
    for filename in glob.glob(str(root_path / "**" / "*.pyx"), recursive=True):
        mod = Path(filename).relative_to(root_path).with_suffix("").as_posix()
        if sys.platform == "win32" and mod in _posix_only_modules:
            continue
        yield mod


def core_sources_old(mod_name):
    sources = [f"cuda/core/{mod_name}.pyx"]
    cpp_file = f"cuda/core/_cpp/{mod_name.lstrip('_')}.cpp"
    if os.path.exists(cpp_file):
        sources.append(cpp_file)
    return sources


def core_sources_new(mod_name):
    sources = [f"cuda/core/{mod_name}.pyx"]
    cpp_file = f"cuda/core/_cpp/{mod_name.lstrip('_')}.cpp"
    if Path(cpp_file).exists():
        sources.append(cpp_file)
    return sources


old_mods = sorted(core_old())
new_mods = sorted(core_new())
check("cuda_core module_names()", old_mods, new_mods)
check(
    "cuda_core Extension names",
    sorted(f"cuda.core.{m.replace(os.path.sep, '.')}" for m in old_mods),
    sorted(f"cuda.core.{m.replace('/', '.')}" for m in new_mods),
)
check(
    "cuda_core Extension sources",
    sorted(tuple(core_sources_old(m)) for m in old_mods),
    sorted(tuple(core_sources_new(m)) for m in new_mods),
)
check(
    "cuda_core include dir",
    [os.path.join("/opt/cuda", "include")],
    [str(Path("/opt/cuda") / "include")],
)

# --------------------------------------------------------------------------
# cuda_bindings/build_hooks.py :: _rename_architecture_specific_files / _prep_extensions
# --------------------------------------------------------------------------
os.chdir("../cuda_bindings")

suffix = "_windows" if sys.platform == "win32" else "_linux"

old_path = os.path.join("cuda", "bindings", "_internal")
new_path = Path("cuda", "bindings", "_internal")
check(
    "cuda_bindings arch-specific sources",
    sorted(glob.glob(os.path.join(old_path, f"*{suffix}.pyx"))),
    sorted(glob.glob(str(new_path / f"*{suffix}.pyx"))),
)

all_pyx = sorted(
    glob.glob("cuda/bindings/utils/*.pyx")
    + glob.glob("cuda/bindings/*.pyx")
    + glob.glob("cuda/bindings/_v2/*.pyx")
    + glob.glob("cuda/bindings/_internal/*.pyx")
)
check(
    "cuda_bindings Extension module names",
    [p.replace(".pyx", "").replace(os.sep, ".").replace("/", ".") for p in all_pyx],
    [".".join(Path(p).with_suffix("").parts) for p in all_pyx],
)
check(
    "cuda_bindings static-library selection (basename)",
    [os.path.basename(p) for p in all_pyx],
    [Path(p).name for p in all_pyx],
)
check(
    "cuda_bindings include/library dirs",
    [
        os.path.join("/opt/cuda", "include"),
        os.path.dirname("/usr/include/python3.12"),
        os.path.join("/opt/cuda", "lib64"),
        os.path.join("/opt/cuda", "lib"),
    ],
    [
        str(Path("/opt/cuda") / "include"),
        str(Path("/usr/include/python3.12").parent),
        str(Path("/opt/cuda") / "lib64"),
        str(Path("/opt/cuda") / "lib"),
    ],
)

print()
if FAILURES:
    print(f"DISCOVERY EQUIVALENCE FAILED: {FAILURES}")
    sys.exit(1)
print("DISCOVERY EQUIVALENCE OK")
