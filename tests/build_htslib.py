"""Build the local VCF/BCF dependencies bundled in release wheels."""

__author__ = "Jonas Meisner"

import os
import subprocess
import sys
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve


### Verify upstream archives before extracting or building them
def unpack(tmp, url, digest):
    path = tmp / url.rsplit("/", 1)[-1]
    urlretrieve(url, path)
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Source checksum differs: {path.name}")
    with tarfile.open(path) as src:
        src.extractall(tmp, filter="data")
    return tmp / path.name.removesuffix(".tar.bz2").removesuffix(".tar.gz")


### Build portable HTSlib with libdeflate for local VCF/BCF
def main():
    if len(sys.argv) != 2:
        raise SystemExit("Provide the HTSlib installation prefix")
    prefix = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="hapla-htslib-") as tmp:
        tmp = Path(tmp)
        lib = unpack(
            tmp,
            "https://github.com/ebiggers/libdeflate/releases/download/v1.26/libdeflate-1.26.tar.gz",
            "125856d4656e0feab660f94842f835923410c9281fedbcee64598c918da42b5a",
        )
        hts = unpack(
            tmp,
            "https://github.com/samtools/htslib/releases/download/1.25/htslib-1.25.tar.bz2",
            "158c271e4e2986bb2dd11283b640edc10c56695b293df77a109ce99480303730",
        )
        build = tmp / "build"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(lib),
                "-B",
                str(build),
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLIBDEFLATE_BUILD_STATIC_LIB=OFF",
                "-DLIBDEFLATE_BUILD_GZIP=OFF",
                "-DLIBDEFLATE_BUILD_TESTS=OFF",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
        subprocess.run(["cmake", "--install", str(build)], check=True)
        env = dict(
            os.environ,
            CPPFLAGS=f"-I{prefix}/include",
            LDFLAGS=f"-L{prefix}/lib -Wl,-rpath,{prefix}/lib",
        )
        subprocess.run(
            [
                "./configure",
                f"--prefix={prefix}",
                "--with-libdeflate",
                "--disable-libcurl",
                "--disable-plugins",
                "--disable-bz2",
                "--disable-lzma",
            ],
            cwd=hts,
            env=env,
            check=True,
        )
        subprocess.run(["make", "-j2"], cwd=hts, check=True)
        subprocess.run(["make", "install"], cwd=hts, check=True)


if __name__ == "__main__":
    main()
