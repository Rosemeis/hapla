"""Run regression tests with native thread limits set before numerical imports."""

__author__ = "Jonas Meisner"

import argparse
import os
import tempfile
import unittest
from pathlib import Path


### Run source or installed tests without relying on the working directory
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="Number of native threads (1)")
    parser.add_argument("--installed", action="store_true", help="Require an installed package")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")

    import hapla
    from hapla.runtime import configureThreads

    configureThreads(args.threads)
    tests = Path(__file__).resolve().parent
    package = Path(hapla.__file__).resolve()
    if args.installed and package.is_relative_to(tests.parent / "hapla"):
        parser.error("Install the built wheel and run without the checkout on PYTHONPATH")
    print(f"Package: {package}\nNative threads: {args.threads}", flush=True)
    with tempfile.TemporaryDirectory(prefix="hapla-suite-") as tmp:
        cwd = Path.cwd()
        try:
            os.chdir(tmp)
            suite = unittest.defaultTestLoader.discover(str(tests))
            result = unittest.TextTestRunner(verbosity=1, buffer=True).run(suite)
        finally:
            os.chdir(cwd)
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
