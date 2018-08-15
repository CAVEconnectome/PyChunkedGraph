from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyChunkedGraph",
    version="0.1",
    author="Sven Dorkenwald",
    author_email="",
    description="Proofreading backend for Neuroglancer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seung-lab/PyChunkedGraph",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigtable",
        "networkx",
        "cloud-volume",
        "bitstring",
        "flask",
        "flask_cors"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
