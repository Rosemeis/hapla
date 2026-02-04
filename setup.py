from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="hapla.reader_cy",
        sources=["hapla/reader_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="hapla.cluster_cy",
        sources=["hapla/cluster_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="hapla.shared_cy",
        sources=["hapla/shared_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="hapla.memory_cy",
        sources=["hapla/memory_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="hapla.admix_cy",
        sources=["hapla/admix_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="hapla.fatash_cy",
        sources=["hapla/fatash_cy.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)