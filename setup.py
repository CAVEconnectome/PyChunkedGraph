from setuptools import setup, find_packages
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name="PyChunkedGraph",
    version=find_version('pychunkedgraph', '__init__.py'),
    author="Sven Dorkenwald",
    author_email="svenmd@princeton.edu",
    description="Proofreading backend for Neuroglancer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seung-lab/PyChunkedGraph",
    packages=find_packages(),
    install_requires=required,
    dependency_links=[
        "git+https://github.com/seung-lab/igneous.git#egg=igneous-dev"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ]
)
