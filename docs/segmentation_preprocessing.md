# Preprocessing 

The goal of preprocessing a over-segmentation is to store the voxel and edge data in such a way that a ChunkedGraph can be created from it. 

## Tools
The ChunkedGraph makes heavy use of [CloudVolume](https://github.com/seung-lab/cloud-volume/) for interfacing with volumetric data and [CloudFiles](https://github.com/seung-lab/cloud-files) for all other data on Google Cloud. It is recommended to use these tools during preprocessing to ensure compatibility. 

## Data formats

[`precomputed`](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) introduced by [neuroglancer](https://github.com/google/neuroglancer) is a commonly used data format for volumetric data and is supported by CloudVolume. The supervoxel segmentation for the ChunkedGraph should ideally be stored using precomputed. Further, we recommend storing the supervoxel segmentation on Google Cloud in the same zone the ChunkedGraph server will be deployed in to reduce latency and avoid egress costs (Google Cloud does not charge for within-zone egress). 

The ChunkedGraph's format is called [`graphene`](https://github.com/seung-lab/cloud-volume/wiki/Graphene) which builds on precomputed. It combines the supervoxel segmentation with an agglomeration graph to provide a dynamic segmentation that can be edited.

## Segmentation IDs

Graphene follows a strict ID nomenclature for all supervoxels and other nodes in the ChunkedGraph hierarchy. In practice that means that most segmentations need to be rewritten to follow this nomenclature. Each ID stores data about its location in space (Chunk coord) and in the ChunkedGraph hierarchy (Layer id):

![](https://user-images.githubusercontent.com/2517065/77118406-7dbd5a00-6a0a-11ea-96bb-003b83beb866.png)

The number of bits for the layer id is fixed to 8 bits. Supervoxels are on layer 1. The number of bits for the chunk coordinates decreases with each layer as there are fewer chunks in higher layers but more segments per chunk. 

Before preprocessing one's segmentation one must determine the bounding box of the segmentation and then the number of bits needed to represent the chunk coordinates. CloudVolume supports bounding boxes that start at arbitrary locations in space (see also `cv.bounds` for a cloudvolume instance of a precomputed segmentation). Once ingested, the bounding box cannot be changed anymore. It is advantageous to keep the bounding box as small as possible. 

After determining the bounding box one can extract the necessary number of bits for the chunk coords by counting the number of chunks in each dimension and calculating how many bits are required to represent the ids in the largest dimension. The number of bits for all dimensions are identical.

Lastly, each supervoxel within a chunk is assigned an ID that is unique among all supervoxels _within_ the same chunk. This ID can be assigned at random but we recommend assigning IDs from a sequential ID space starting at 0. 

## Storing supervoxel edges and components

There are three types of edges:
1. `in_chunk`: edges between supervoxels within a chunk
2. `cross_chunk`: edges between parts of "the same" supervoxel in the unchunked segmentation that has been split across chunk boundary 
3. `between_chunk`: edges between supervoxels across chunks

Every pair of touching supervoxels has an edge between them. All edges are stored using [protobuf](https://github.com/CAVEconnectome/PyChunkedGraph/blob/pcgv2/pychunkedgraph/io/protobuf/chunkEdges.proto). During ingest only edges of type 2. and 3. are copied into BigTable, whereas edges of type 1. are always read from storage to reduce cost. Similar to the supervoxel segmentation, we recommed storing these on GCloud in the same zone the ChunkedGraph server will be deployed in to reduce latency. 

To denote which edges form a connected component within a chunk, a component mapping needs to be created. This mapping is only used during ingest. 

More details on how to create these protobuf files can be found [here](https://github.com/CAVEconnectome/PyChunkedGraph/blob/pcgv2/docs/storage.md).


