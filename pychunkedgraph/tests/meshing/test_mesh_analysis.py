"""Tests for pychunkedgraph.meshing.mesh_analysis"""

import numpy as np

from pychunkedgraph.meshing.mesh_analysis import compute_centroid_by_range


class TestComputeCentroidByRange:
    def test_single_point(self):
        vertices = np.array([[5.0, 10.0, 15.0]])
        centroid = compute_centroid_by_range(vertices)
        np.testing.assert_array_equal(centroid, [5.0, 10.0, 15.0])

    def test_two_points(self):
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        centroid = compute_centroid_by_range(vertices)
        np.testing.assert_array_equal(centroid, [5.0, 10.0, 15.0])

    def test_symmetric_cube(self):
        """Centroid of a unit cube centered at origin."""
        vertices = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
            ]
        )
        centroid = compute_centroid_by_range(vertices)
        np.testing.assert_array_equal(centroid, [0.0, 0.0, 0.0])

    def test_asymmetric_distribution(self):
        """Many points clustered but centroid is bbox midpoint, not mean."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [10.0, 10.0, 10.0],
            ]
        )
        centroid = compute_centroid_by_range(vertices)
        # bbox: [0,0,0] to [10,10,10], midpoint = [5,5,5]
        np.testing.assert_array_equal(centroid, [5.0, 5.0, 5.0])

    def test_negative_coordinates(self):
        vertices = np.array([[-10.0, -20.0, -30.0], [-2.0, -4.0, -6.0]])
        centroid = compute_centroid_by_range(vertices)
        np.testing.assert_array_equal(centroid, [-6.0, -12.0, -18.0])

    def test_mixed_coordinates(self):
        vertices = np.array([[-5.0, 0.0, 10.0], [5.0, 20.0, 30.0]])
        centroid = compute_centroid_by_range(vertices)
        np.testing.assert_array_equal(centroid, [0.0, 10.0, 20.0])
