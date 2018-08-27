import os

from setuptools import setup, find_packages

README = None
with open(os.path.abspath('README.md')) as fh:
    README = fh.read()

install_requires = [
    ]

tests_require = [
    'nose',
    ]

setup(
    name='music-ml',
    version='0.0.1',
    description='A framework for predicting genres of music',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Troy Holsapple',
    author_email='troy.holsapple@gmail.com',
    url='https://github.com/tholsapp/MusicML',
    packages=find_packages(),
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='nose.collector',
)

