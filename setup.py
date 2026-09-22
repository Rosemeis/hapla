"""Build portable CPU kernels and the native HTSlib genotype reader."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


### Locate HTSlib through its prefix, pkg-config, or standard paths
def htsFlags():
    paths = [os.environ.get("HTSLIB_PREFIX")]
    if not paths[0]:
        try:
            flags = subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "htslib"], text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        else:
            flags = shlex.split(flags)
            return [v for v in flags if v.startswith(("-I", "-D"))], [
                v for v in flags if not v.startswith(("-I", "-D"))
            ]
        paths += [os.environ.get("CONDA_PREFIX"), sys.prefix, "/opt/homebrew", "/usr/local", "/usr"]
    for prefix in paths:
        if prefix and (Path(prefix) / "include/htslib/vcf.h").is_file():
            lib = Path(prefix) / "lib"
            return [f"-I{prefix}/include"], [f"-L{lib}", f"-Wl,-rpath,{lib}", "-lhts"]
    raise RuntimeError(
        "HTSlib headers and library are required. Install htslib (Conda/Homebrew) or "
        "libhts-dev (Linux), or set HTSLIB_PREFIX to its installation prefix."
    )


### Set portable compiler flags and select each kernel's native libraries
def extensions():
    cc = ["-O3", "-fno-math-errno"]
    ld = ["-lm"]

    # Portable by default for heterogeneous HPC nodes. Opt in to local tuning.
    if os.environ.get("HAPLA_NATIVE") == "1":
        cc.append("-march=native")
    if sys.platform == "darwin":
        omp_cc, omp_ld = ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
        for prefix in (
            os.environ.get("LIBOMP_PREFIX"),
            os.environ.get("CONDA_PREFIX"),
            sys.prefix,
            "/opt/homebrew/opt/libomp",
            "/usr/local/opt/libomp",
        ):
            if prefix and (Path(prefix) / "include/omp.h").is_file():
                omp_cc.append(f"-I{prefix}/include")
                omp_ld += [f"-L{prefix}/lib", f"-Wl,-rpath,{prefix}/lib"]
                break
    else:
        omp_cc, omp_ld = ["-fopenmp"], ["-fopenmp"]
    hts_cc, hts_ld = htsFlags()
    out = []
    for name in (
        "shared_cy",
        "admix_cy",
        "fatash_cy",
        "eval_cy",
        "vcf_cy",
        "packed_cy",
        "struct_cy",
    ):
        cargs, largs = cc.copy(), ld.copy()
        if name == "vcf_cy":
            cargs += hts_cc
            largs += hts_ld
        elif name != "packed_cy":
            cargs += omp_cc
            largs += omp_ld
        out.append(
            Extension(
                f"hapla.{name}",
                [f"hapla/{name}.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=cargs,
                extra_link_args=largs,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        )
    return out


setup(ext_modules=cythonize(extensions(), language_level=3))
