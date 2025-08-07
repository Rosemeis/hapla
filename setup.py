from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"hapla.reader_cy",
		["hapla/reader_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.cluster_cy",
		["hapla/cluster_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.shared_cy",
		["hapla/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.memory_cy",
		["hapla/memory_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.admix_cy",
		["hapla/admix_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"hapla.fatash_cy",
		["hapla/fatash_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-fno-signed-zeros', '-march=native'],
		extra_link_args=['-fopenmp', '-lm'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="hapla",
	version="0.32.1",
	author="Jonas Meisner",
	author_email="meisnerucph@gmail.com",
	description="Framework for haplotype clustering in phased genotype data",
	long_description_content_type="text/markdown",
	long_description=open("README.md").read(),
	url="https://github.com/Rosemeis/hapla",
	classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
	ext_modules=cythonize(extensions, compiler_directives={'language_level':'3'}),
	python_requires=">=3.10",
	install_requires=[
		"cython>3.0.0",
		"cyvcf2>=0.31.0",
		"numpy>2.0.0"
	],
	packages=["hapla"],
	entry_points={
		"console_scripts": ["hapla=hapla.main:main"]
	},
)
