from setuptools import setup, find_packages

setup(
    name='lazaro',
    version='0.1.3',
    packages=find_packages(),
    url='https://github.com/GabrielMusat/lazaro',
    python_requires=">=3.7.*",
    license='Apache 2.0',
    author='Gabriel Musat',
    install_requires=open("requirements.txt").read().split("\n"),
    author_email='gabimtme@gmail.com',
    description='Reinforcement learning framework for implementing custom models on custom environments using state '
                'of the art RL algorithms'
    
)
