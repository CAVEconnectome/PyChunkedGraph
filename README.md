# PyChunkedGraph


[![Build Status](https://travis-ci.org/seung-lab/PyChunkedGraph.svg?branch=master)](https://travis-ci.org/seung-lab/PyChunkedGraph)
[![codecov](https://codecov.io/gh/seung-lab/PyChunkedGraph/branch/master/graph/badge.svg)](https://codecov.io/gh/seung-lab/PyChunkedGraph)

The PyChunkedGraph is a proofreading and segmentation data management backend powering FlyWire and other proofreading platforms. It builds on an initial agglomeration of supervoxels and facilitates fast and parallel editing of connected components in the agglomeration graph by many users.

## Using the PyChunkedGraph

The ChunkedGraph is built on Google Cloud BigTable. A BigTable instance is required to use this ChunkedGraph implementation. 

### Environmental Variables
There are three environmental variables that need to be set
to connect to a chunkgraph:

> GOOGLE_APPLICATION_CREDENTIALS

Location of the google-secret.json file.

> BIGTABLE_PROJECT

Name of the Google Cloud project name.

> BIGTABLE_INSTANCE

Name of the Bigtable Instance ID. (Default is 'pychunkedgraph')
### Ingest 

`/ingest` provides examples for ingest scripts. The ingestion pipeline designed to use the output of the seunglab's agglomeration pipeline but can be adjusted to use alternative data sources. 

### Deployment / Import

The PyChunkedGraph can be locally deployed (`run_dev.py`), imported in a python script (`from pychunkedgraph.backend import chunkedgraph`) or deployed on a kubernetes server. Deployment code for a kubernetes server on Google Cloud is not included in this repository. Please feel free to reach out if you are interested in that. 

## Publication 

[FlyWire: Online community for whole-brain connectomics](https://www.biorxiv.org/content/10.1101/2020.08.30.274225v1)
```
@article {FlyWire2020,
  	  author = {Dorkenwald, Sven and McKellar, Claire and Macrina, Thomas and Kemnitz, Nico and Lee, Kisuk and Lu, Ran and Wu, Jingpeng and Popovych, Sergiy and Mitchell, Eric and Nehoran, Barak and Jia, Zhen and Bae, J. Alexander and Mu, Shang and Ih, Dodam and Castro, Manuel and Ogedengbe, Oluwaseun and Halageri, Akhilesh and Ashwood, Zoe and Zung, Jonathan and Brittain, Derrick and Collman, Forrest and Schneider-Mizell, Casey and Jordan, Chris and Silversmith, William and Baker, Christa and Deutsch, David M and Encarnacion-Rivera, Lucas and Kumar, Sandeep and Burke, Austin and Gager, Jay and Hebditch, James and Koolman, Selden and Moore, Merlin and Morejohn, Sarah and Silverman, Ben and Willie, Kyle and Willie, Ryan and Yu, Szi-chieh and Murthy, Mala and Seung, Hyunjune Sebastian},
	  title = {FlyWire: Online community for whole-brain connectomics},
	  elocation-id = {2020.08.30.274225},
	  year = {2020},
	  doi = {10.1101/2020.08.30.274225},
	  publisher = {Cold Spring Harbor Laboratory},
	  URL = {https://www.biorxiv.org/content/early/2020/08/30/2020.08.30.274225},
	  eprint = {https://www.biorxiv.org/content/early/2020/08/30/2020.08.30.274225.full.pdf},
	  journal = {bioRxiv}
}
```

