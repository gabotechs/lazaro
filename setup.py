from setuptools import setup, find_packages

setup(
    name='lazaro',
    version=open("version.txt").read(),
    packages=find_packages(),
    url='https://github.com/GabrielMusat/lazaro',
    python_requires=">=3.7.*",
    license='Apache 2.0',
    author='Gabriel Musat',
    install_requires=[req for req in open("requirements.txt").read().split("\n") if len(req) > 0],
    author_email='gabimtme@gmail.com',
    description='Reinforcement learning framework for implementing custom models on custom environments using state '
                'of the art RL algorithms',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed"
    ]
)
