import os
from setuptools import setup, find_packages


packages = find_packages()
packages_to_install = []
# Ensure that we don't pollute the global namespace.
for p in packages:
    if 'drl_implementation' in p:
        packages_to_install.append(p)

setup(name='drl-implementation',
      version='1.0.0',
      description='A collection of deep reinforcement learning algorithms for fast implementation',
      url='https://github.com/IanYangChina/DRL_Implementation',
      author='XintongYang',
      author_email='YangX66@cardiff.ac.uk',
      packages=packages,
      package_dir={'drl_implementation': 'drl_implementation'},
      package_data={'drl_implementation': [
          'examples/*.md',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
