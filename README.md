# PyChunkedGraph


[![Tests](https://github.com/seung-lab/PyChunkedGraph/actions/workflows/main.yml/badge.svg)](https://github.com/seung-lab/PyChunkedGraph/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/seung-lab/PyChunkedGraph/branch/main/graph/badge.svg)](https://codecov.io/gh/seung-lab/PyChunkedGraph)

The PyChunkedGraph is a proofreading and segmentation data management backend powering FlyWire and other proofreading platforms. It builds on an initial agglomeration of supervoxels and facilitates fast and parallel editing of connected components in the agglomeration graph by many users.

## PyChunkedGraph versions

The main branch represents the second version (v2) of the PyChunkedGraph implementation. The first version (v1) is still maintained and can be found under `pcgv1`. The v2 implementation resolved data storage concerns and removed some scaling bottlenecks in the implementation. Any new dataset should use the v2. 

## Using the PyChunkedGraph

The ChunkedGraph is built on Google Cloud BigTable. A BigTable instance is required to use this ChunkedGraph implementation. 

### Environmental Variables
There are three environmental variables that need to be set
to connect to a chunkgraph:

- `GOOGLE_APPLICATION_CREDENTIALS`: Location of the google-secret.json file.
- `BIGTABLE_PROJECT`: Name of the Google Cloud project name.
- `BIGTABLE_INSTANCE`: Name of the Bigtable Instance ID. (Default is 'pychunkedgraph')

### Ingest 

`/ingest` provides examples for ingest scripts. The ingestion pipeline designed to use the output of the seunglab's agglomeration pipeline but can be adjusted to use alternative data sources. 

### Deployment / Import

The PyChunkedGraph can be locally deployed (`run_dev.py`), imported in a python script (`from pychunkedgraph.backend import chunkedgraph`) or deployed on a kubernetes server. Deployment code for a kubernetes server on Google Cloud is not included in this repository. Please feel free to reach out if you are interested in that. 

## System Design

As a backend the PyChunkedGraph can be combined with any frontend that adheres to its API. We use an adapted version of [neuroglancer](https://github.com/seung-lab/neuroglancer/tree/nkem-multicut) which is publicly available.


## Publication 

When using or referencing the PyChunkedGraph, please use the citation below. The FlyWire paper described and published the PyChunkedGraph v1.

[FlyWire: Online community for whole-brain connectomics](https://www.nature.com/articles/s41592-021-01330-0)
```
@article{FlyWire2021,
  doi = {10.1038/s41592-021-01330-0},
  url = {https://doi.org/10.1038/s41592-021-01330-0},
  year = {2021},
  month = dec,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {19},
  number = {1},
  pages = {119--128},
  author = {Sven Dorkenwald and Claire E. McKellar and Thomas Macrina and Nico Kemnitz and Kisuk Lee and Ran Lu and Jingpeng Wu and Sergiy Popovych and Eric Mitchell and Barak Nehoran and Zhen Jia and J. Alexander Bae and Shang Mu and Dodam Ih and Manuel Castro and Oluwaseun Ogedengbe and Akhilesh Halageri and Kai Kuehner and Amy R. Sterling and Zoe Ashwood and Jonathan Zung and Derrick Brittain and Forrest Collman and Casey Schneider-Mizell and Chris Jordan and William Silversmith and Christa Baker and David Deutsch and Lucas Encarnacion-Rivera and Sandeep Kumar and Austin Burke and Doug Bland and Jay Gager and James Hebditch and Selden Koolman and Merlin Moore and Sarah Morejohn and Ben Silverman and Kyle Willie and Ryan Willie and Szi-chieh Yu and Mala Murthy and H. Sebastian Seung},
  title = {{FlyWire}: online community for whole-brain connectomics},
  journal = {Nature Methods}
}
```
