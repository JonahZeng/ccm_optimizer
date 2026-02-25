'''build demosaic c extension'''
from setuptools._distutils.ccompiler import new_compiler
from setuptools import setup, Extension
import numpy as np

# Detect compiler type
compiler = new_compiler()
compiler_type = compiler.compiler_type # 'msvc', 'mingw32', 'unix', etc.
if compiler_type == 'msvc':
    extra_compile_args = ['/O2', '/W4']
elif compiler_type == 'mingw32': # MinGW64 使用 GCC 参数
    extra_compile_args = ['-O3', '-Wall', '-Wextra']
else: # Linux / macOS
    extra_compile_args = ['-O3', '-Wall']

# Define the C extension
demosaic_extension = Extension(
    'demosaic',
    sources=['./src/c_src/demosaic.c'],
    include_dirs=[np.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=extra_compile_args,
)

ccm_extension = Extension(
    'ccm',
    sources=['./src/c_src/ccm.c'],
    include_dirs=[np.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=extra_compile_args,
)

setup(
    name='dmccm',
    version='0.1.0',
    description='Minimal Bayer demosaicing C extension',
    ext_modules=[demosaic_extension, ccm_extension],
    install_requires=['numpy>=1.19.0'],
    python_requires='>=3.7',
)
