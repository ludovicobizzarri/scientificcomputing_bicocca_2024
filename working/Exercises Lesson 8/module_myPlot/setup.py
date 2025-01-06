from setuptools import setup, find_packages

setup(name='module_myPlot',
      description='plot data with the desired rc parameters',
      url='https://github.com/ludovicobizzarri',
      author='Ludovico Bizzarri',
      author_email='l.bizzarri1@campus.unimib.it',
      license='MIT',
      version='0.0.3',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])