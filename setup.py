from setuptools import setup, find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name='yann',
  version='0.0.25',
  description='yet another neural network library',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url='https://github.com/michalwols/yann',
  author='Michal Wolski',
  author_email='michal@bite.ai',
  license='MIT',
  packages=find_packages(),
  entry_points={
    'console_scripts': ['yann=yann.cli:main'],
  },
  extras_require={
    'cli': ['click>=6.7'],
    'pretrainedmodels': ['pretrainedmodels']
  },
  install_requires=[
    'numpy',
    'scipy',
    'scikit-learn',
    'torch>=0.4.0',
    'matplotlib',
    'torchvision'
  ],
  python_requires='>=3.6',
  zip_safe=False)
