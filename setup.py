from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"hapla.reader_cy",
		["hapla/reader_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.cluster_cy",
		["hapla/cluster_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.shared_cy",
		["hapla/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.memory_cy",
		["hapla/memory_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), 
	Extension(
		"hapla.admix_cy",
		["hapla/admix_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.fatash_cy",
		["hapla/fatash_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="hapla",
	version="0.11",
	description="Framework for haplotype clustering",
	author="Jonas Meisner",
	packages=["hapla"],
	entry_points={
		"console_scripts": ["hapla=hapla.main:main"]
	},
	python_requires=">=3.7",
	install_requires=[
		"cython",
		"cyvcf2",
		"numpy",
		"scipy"
	],
	ext_modules=cythonize(extensions, compiler_directives={'language_level':'3'}),
	include_dirs=[numpy.get_include()]
)
