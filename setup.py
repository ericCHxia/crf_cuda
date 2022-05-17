from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='crf_cuda',
    version='1.0',
    description='',
    author='xia chenghao',
    author_email='xch@raxch.cn',
    ext_modules=[
        CUDAExtension('crf_cuda.crf_cuda', [
            'src/crf_cuda/alg_cuda_kernel.cu',
            'src/crf_cuda/score_cuda_kernal.cu',
            'src/crf_cuda/crf_cuda.cpp',
        ]),
    ],
    keywords=[
        'pytorch',
        'crf',
        'cuda'
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['crf_cuda'],
    package_dir={'crf_cuda': 'src'},
    package_data={'crf_cuda': ['*.py','*.pyi']},
    install_requires=[
        "torch~="+torch.__version__
    ],
)
