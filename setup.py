from setuptools import setup

setup(
    name='lazaro',
    version='0.1.0',
    packages=['agents', 'agents.base', 'agents.tests', 'agents.explorers', 'agents.explorers.base',
              'agents.explorers.tests', 'agents.replay_buffers', 'agents.replay_buffers.base',
              'agents.replay_buffers.base.segment_trees', 'agents.replay_buffers.tests', 'logger', 'plotter',
              'environments', 'evolutioners'],
    url='https://github.com/GabrielMusat/lazaro',
    python_requires=">=3.7.*",
    license='Apache 2.0',
    author='Gabriel Musat',
    author_email='gabimtme@gmail.com',
    description='Reinforcement learning framework for implementing custom models on custom environments using state '
                'of the art RL algorithms'
    
)
