from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name="PyChunkedGraph",
    version="0.1",
    author="Sven Dorkenwald",
    author_email="svenmd@princeton.edu",
    description="Proofreading backend for Neuroglancer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seung-lab/PyChunkedGraph",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    dependency_links=['git+https://github.com/sdorkenw/MultiWrapper']
)
