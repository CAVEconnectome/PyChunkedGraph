## Meshing Overview

This file is intended to explain and record how our meshing procedure works and why we designed it this way.
Meshing is done by starting at layer 2 in the chunk graph, and meshing each chunk within it. For each higher layer in the 
graph, adjacent chunks in the layer below it are stitched together.

### How meshes are created

Meshes are created by giving 3D labeled voxel segmentation data, and passing that to ZMesh, a python library that runs marching cubes on the 
data and produces meshes for the different objects in the data. ZMesh will produce a list of floats and one of unsigned integers. 
The floats represent coordinates of the vertices of the mesh; the unsigned ints represent triangular faces of the mesh -- 
indices in the list of floats. After running marching cubes, ZMesh will run a quadratic simplification algorithm on the mesh to simplify
the output. ZMesh is a python library written by Will Silversmith that wraps a C++ meshing library originally written by Alexander Zlateski.

When we mesh each chunk, we download the segmentation data from CloudVolume for the chunk, as well as a 1 voxel overlap in the positive x, y, and z
directions (so that meshes across successive chunks that represent the same object will "stitch" together). This 3D subvolume is passed into ZMesh.

[Two Adjacent Chunks][meshing_diagrams/TwoAdjacentChunks.png]

Above: Two adjacent chunks (ABFE and EFGH) that will be passed into the MeshTask. EFCD is the 1 voxel overlap region when meshing ABFE, so the actual
region that will be meshed is ABCD.

## The issue that using Draco introduces

Draco quantizes the coordinates of the vertices in its mesh, meaning it creates a grid with equally spaced points over the mesh, 
and changes the coordinates of each vertex to be the gridpoint that vertex is closest to. We can choose the size of the grid and its origin, 
but it has to be a cube, and we have a limited amount of control over the amount of points in the grid. This is an issue because our chunks are
not necessarily cubes, and because we need our meshes that cross chunk boundaries to have identical vertices on each side of the boundary to
be able to stitch them together at the layer above. Because our chunks are not cubes, if we specify the bounding box of the grid to be the
smallest cube that contains the chunk we are meshing, some of this chunk's boundaries will not lie on the grid and because 
the vertices on those boundaries will be snapped to the grid, there will be no way to stitch the affected meshes to their sides on the other
side of the chunk boundary.

[Unstable Chunk Boundary][meshing_diagrams/UnstableChunkBoundary.png]

Above is a diagram to visualize the problem. ABFE and EFGH are our PCG chunks, ABIJ and EFKL our respective draco bounding cubes. The problem with the
above setup is that if EF does not lie on the draco grid when meshing ABIJ, it will move, and it will end up in a different location than when meshing
EFKL.

## Our resolution to this issue

In order to resolve the above issue we need a way of choosing the draco bounding boxes such that any chunk boundary will be snapped 
to the same location when meshing from either chunk on the sides of the boundary. We cannot avoid the boundaries moving but we can ensure 
that given any pair of adjacent chunks, their shared boundary moves to the same location after meshing. 

To do this we create a global grid over the entire worldspace at each layer, where the lines in this grid will be exactly where the quantization
lines in the draco bounding box grids will be. We set the size of each draco cube to be a specific multiple of the length between each point in the
grid, and we make sure the origin of each draco cube is at some point in the grid. We do this because if we mesh two adjacent chunks, 
where the chunks' draco cubes overlap and sit in this global grid, then the boundary these two chunks share will be moved to the same plane in the
grid when meshed from either chunk.

[Stable Chunk Boundary][meshing_diagrams/StableChunkBoundary.png]

The above diagram seems similar to the previous setup, but the key difference is that the length of a side of the new cube is 7 instead of 6.
Draco forces us to have 2^n grid points but that includes the beginning and end of the grid, meaning that if we want to have our grid points be at
integer values our draco cube side length needs to be a multiple of 2^n - 1, in this case n = 3 so 7. This n represents the amount of bits Draco uses
internally to encode a coordinate of a vertex.

## Details of how the draco parameters are chosen

The details for how the draco parameters are chosen are in the function get_draco_encoding_settings_for_chunk_exact_boundary in meshgen.py.
Here's a brief summary: given a layer in the chunk graph and a mip level of the segmentation data, we find the longest length of a chunk in that layer
in nm and the shortest component of a voxel in that mip level in nm. Preliminarly select that longest length to be the side length of the draco cube,
then select the smallest n such that the draco bin size is less than that shortest component over the square root of 2 (because of how marching cubes works).
Then expand the draco cube size until the bin size is an integer. Now the bins and origin are all at integers, making stitching possible.
