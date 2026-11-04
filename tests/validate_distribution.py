"""Check that wheels and source archives contain only package and development files."""

__author__ = "Jonas Meisner"

import argparse
import ast
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = {
    p.relative_to(ROOT).as_posix()
    for p in (ROOT / "hapla").iterdir()
    if p.suffix in (".py", ".pyx", ".pxd")
}
NATIVE = {p.stem for p in (ROOT / "hapla").glob("*.pyx")}
LICENSES = {"LICENSE"} | {p.relative_to(ROOT).as_posix() for p in (ROOT / "licenses").glob("*.txt")}
SOURCE = {
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "environment-cpu.yml",
    "pyproject.toml",
    "setup.py",
}
SOURCE |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("*.py")}
SOURCE |= LICENSES


### Read archive members without extracting executable files
def validate(path, bundled=False):
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as src:
            files = {n: src.read(n) for n in src.namelist() if not n.endswith("/")}
        info = {n.split("/", 1)[0] for n in files if ".dist-info/" in n}
        if len(info) != 1:
            raise ValueError("Wheel requires one metadata directory")
        info = info.pop()
        metadata = f"{info}/METADATA"
        native = {
            n for n in files if Path(n).parent == Path("hapla") and n.endswith((".so", ".pyd"))
        }
        if sorted(Path(n).name.split(".")[0] for n in native) != sorted(NATIVE):
            raise ValueError("Wheel native extensions do not match the package")
        libs = {
            n
            for n in files
            if Path(n).parent.as_posix() in ("hapla.libs", "hapla/.dylibs")
            and (".so" in Path(n).suffixes or n.endswith(".dylib"))
        }
        if bundled:
            for names in (("libhts",), ("libdeflate",), ("libgomp", "libomp")):
                if not any(Path(n).name.startswith(names) for n in libs):
                    raise ValueError(f"Missing bundled shared library: {'/'.join(names)}")
        allowed = PACKAGE | native | libs | {n for n in files if n.startswith(info + "/")}
        required = PACKAGE | {metadata, f"{info}/entry_points.txt", f"{info}/licenses/LICENSE"}
        required |= {f"{info}/licenses/{n}" for n in LICENSES}
        if (
            files[f"{info}/entry_points.txt"].decode().strip()
            != "[console_scripts]\nhapla = hapla.main:main"
        ):
            raise ValueError("Unexpected command entry point")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r|gz") as src:
            files = {m.name: src.extractfile(m).read() for m in src if m.isfile()}
        roots = {n.split("/", 1)[0] for n in files}
        if len(roots) != 1:
            raise ValueError("Source archive requires one root directory")
        root = roots.pop() + "/"
        files = {n.removeprefix(root): value for n, value in files.items()}
        metadata = "PKG-INFO"
        required = PACKAGE | SOURCE | {metadata}
        allowed = required | {"setup.cfg"} | {n for n in files if n.startswith("hapla.egg-info/")}
    else:
        raise ValueError(f"Unsupported distribution: {path}")
    missing, extra = required - files.keys(), files.keys() - allowed
    if missing or extra:
        raise ValueError(f"Missing files: {sorted(missing)}. Unexpected files: {sorted(extra)}")
    version = ast.literal_eval(ast.parse(files["hapla/__init__.py"]).body[0].value)
    meta = Parser().parsestr(files[metadata].decode())
    if meta["Name"] != "hapla" or meta["Version"] != version:
        raise ValueError("Distribution name/version does not match the package")
    if meta.get_all("Requires-Dist") != ["numpy>2.0.0"]:
        raise ValueError("Unexpected runtime dependencies")
    for name in PACKAGE | (set() if wheel else SOURCE):
        if files[name] != (ROOT / name).read_bytes():
            raise ValueError(f"Distribution source differs from checkout: {name}")
    for name in LICENSES:
        key = f"{info}/licenses/{name}" if wheel else name
        if files[key] != (ROOT / name).read_bytes():
            raise ValueError(f"Distribution license differs from checkout: {name}")
    print(f"Validated {path.name}")


### Validate each wheel or source archive supplied on the command line
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundled", action="store_true", help="Require wheel runtime libraries")
    parser.add_argument("files", nargs="+", type=Path, help="Wheels or source archives")
    args = parser.parse_args()
    for path in args.files:
        validate(path, args.bundled)


if __name__ == "__main__":
    main()
