# PyChunkedGraph


[![Build Status](https://travis-ci.org/seung-lab/PyChunkedGraph.svg?branch=master)](https://travis-ci.org/seung-lab/PyChunkedGraph)
[![codecov](https://codecov.io/gh/seung-lab/PyChunkedGraph/branch/master/graph/badge.svg)](https://codecov.io/gh/seung-lab/PyChunkedGraph)

The PyChunkedGraph is a proofreading and segmentation data management backend with (not limited to) the following features:
- Concurrent proofreading by multiple users without restrictions on the workflow
- Continuous versioning of proofreading edits
- Making changes visible to all users immediately
- Local mincut computations

## Scaling to large datasets


## Deployment 

While the PyChunkedGraph can be deployed as a stand-alone entity, we deploy it within our [annotation infrastructure](https://github.com/seung-lab/AnnotationPipelineOverview) which uses a CI Kubernetes deployment. We will make instructions and scripts for a mostly automated deployment to Google Cloud available soon.

## Building your own PyChunkedGraph



## System Design

As a backend the PyChunkedGraph can be combined with any frontend that adheres to its API. We use an adapted version of [neuroglancer](https://github.com/seung-lab/neuroglancer/tree/nkem-multicut) which is publicly available.


[system_design]: https://github.com/seung-lab/PyChunkedGraph/blob/master/ProofreadingDiagram.png "System Design"
![alt text][system_design]
