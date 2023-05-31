from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"src.cluster_cy",
		["src/cluster_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.shared_cy",
		["src/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.assoc_cy",
		["src/assoc_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.reader_cy",
		["src/reader_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		language="c++",
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="hapla",
	version="0.1",
	description="Framework for haplotype clustering",
	author="Jonas Meisner",
	packages=["src"],
	entry_points={
		"console_scripts": ["hapla=src.run:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"numpy",
		"scipy",
		"cyvcf2"
	],
	ext_modules=cythonize(extensions, compiler_directives={'language_level':'3'}),
	include_dirs=[numpy.get_include()]
)
