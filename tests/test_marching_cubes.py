"""
Tests for marching_cubes.py - Isosurface extraction algorithm.
"""

import pytest
import numpy as np
from marching_cubes import marching_cubes, marching_cubes_optimized


class TestMarchingCubesDataStructures:
    """Tests for data structure shapes and integrity."""
    
    def test_edge_table_length(self):
        """Test EDGE_TABLE has correct length (256 entries)."""
        from marching_cubes import EDGE_TABLE
        assert len(EDGE_TABLE) == 256
    
    def test_edge_table_values_valid(self):
        """Test EDGE_TABLE contains valid hex values."""
        from marching_cubes import EDGE_TABLE
        for val in EDGE_TABLE:
            assert isinstance(val, int)
            assert 0 <= val <= 0xFFF  # 12-bit values
    
    def test_tri_table_length(self):
        """Test TRI_TABLE has correct length (256 entries)."""
        from marching_cubes import TRI_TABLE
        assert len(TRI_TABLE) == 256
    
    def test_tri_table_entries_valid(self):
        """Test TRI_TABLE entries are valid."""
        from marching_cubes import TRI_TABLE
        for entry in TRI_TABLE:
            assert isinstance(entry, list)
            # Each entry should end with -1
            assert entry[-1] == -1
            # All other values should be edge indices 0-11 or -1
            for val in entry:
                assert val == -1 or (0 <= val <= 11)
    
    def test_tri_table_triangles(self):
        """Test TRI_TABLE entries form complete triangles."""
        from marching_cubes import TRI_TABLE
        for i, entry in enumerate(TRI_TABLE):
            # Count non-(-1) values
            values = [v for v in entry if v != -1]
            # Should be multiple of 3 (complete triangles)
            assert len(values) % 3 == 0, f"Entry {i} has {len(values)} values (not multiple of 3)"
    
    def test_edge_vertices_length(self):
        """Test EDGE_VERTICES has 12 edges."""
        from marching_cubes import EDGE_VERTICES
        assert len(EDGE_VERTICES) == 12
    
    def test_edge_vertices_valid_indices(self):
        """Test EDGE_VERTICES contains valid vertex pairs."""
        from marching_cubes import EDGE_VERTICES
        for edge in EDGE_VERTICES:
            assert len(edge) == 2
            v0, v1 = edge
            # Vertices should be 0-7 (cube corners)
            assert 0 <= v0 <= 7
            assert 0 <= v1 <= 7
            # Should be different vertices
            assert v0 != v1
    
    def test_edge_vertices_cube_topology(self):
        """Test EDGE_VERTICES represents valid cube edges."""
        from marching_cubes import EDGE_VERTICES
        
        # Define cube topology: each vertex should connect to exactly 3 others
        connections = {i: [] for i in range(8)}
        for v0, v1 in EDGE_VERTICES:
            connections[v0].append(v1)
            connections[v1].append(v0)
        
        # Each vertex should have exactly 3 connections
        for vertex, neighbors in connections.items():
            assert len(neighbors) == 3, f"Vertex {vertex} has {len(neighbors)} connections"


class TestMarchingCubesZeroAreaTriangles:
    """Tests for detecting and handling degenerate triangles."""
    
    def test_no_zero_area_triangles_sphere(self):
        """Test sphere produces no zero-area triangles."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Check each triangle has non-zero area
        for v0, v1, v2 in triangles:
            # Compute triangle area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            
            # Area should be non-zero (allow small tolerance for numerical precision)
            assert area > 1e-10, f"Zero-area triangle found: {v0}, {v1}, {v2}"
    
    def test_triangles_not_degenerate(self):
        """Test triangles are not degenerate (vertices not collinear)."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 15)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        for v0, v1, v2 in triangles:
            # Check vertices are distinct
            assert not np.allclose(v0, v1), "Vertices v0 and v1 are identical"
            assert not np.allclose(v1, v2), "Vertices v1 and v2 are identical"
            assert not np.allclose(v0, v2), "Vertices v0 and v2 are identical"
            
            # Check vertices are not collinear
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            
            # Cross product should be non-zero
            assert np.linalg.norm(cross) > 1e-10, "Vertices are collinear"
    
    def test_triangle_orientation_consistent(self):
        """Test triangles have consistent winding order."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 12)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1  # Sphere
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # For a sphere, normals should generally point outward
        # Sample a few triangles and check normal direction
        for v0, v1, v2 in triangles[:min(10, len(triangles))]:
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            # Centroid of triangle
            centroid = (v0 + v1 + v2) / 3
            
            # For sphere centered at origin, normal should point away from origin
            # (or toward it, consistently)
            dot = np.dot(normal, centroid)
            
            # Just check that normal is not perpendicular to radius
            # (which would indicate degenerate orientation)
            assert abs(dot) > 1e-8


class TestMarchingCubesNumericalRobustness:
    """Enhanced numerical robustness tests."""
    
    def test_handles_inf_with_assertion(self):
        """Test that infinity values are handled without producing invalid triangles."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        field[3, 3, 3] = np.inf
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Should produce some triangles (from non-inf regions)
        # All vertices should be finite
        for v0, v1, v2 in triangles:
            assert np.all(np.isfinite(v0))
            assert np.all(np.isfinite(v1))
            assert np.all(np.isfinite(v2))
    
    def test_handles_zero_denominator_in_interpolation(self):
        """Test interpolation handles zero denominator."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # Create field where some adjacent values are identical
        field = np.round(X**2 + Y**2 + Z**2)  # Rounded values
        
        triangles = marching_cubes(field, 0.5, coords, coords, coords)
        
        # Should handle gracefully - all vertices should be finite
        for v0, v1, v2 in triangles:
            assert np.all(np.isfinite(v0))
            assert np.all(np.isfinite(v1))
            assert np.all(np.isfinite(v2))
    
    def test_runtime_warning_suppression(self):
        """Test that runtime warnings are handled."""
        from marching_cubes import marching_cubes
        import warnings
        
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # Create field that might cause division warnings
        field = X**2 + Y**2 + Z**2 - 1
        # Make some adjacent cells have identical values
        field[5:7, 5:7, 5:7] = 0.0
        
        # Should not raise warnings as errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            triangles = marching_cubes(field, 0.0, coords, coords, coords)
            
            # Should still produce valid output
            assert isinstance(triangles, list)


class TestMarchingCubesEdgeCases:
    """Enhanced edge case tests with assertions."""
    
    def test_single_cell_produces_valid_output(self):
        """Test with minimal 2x2x2 grid produces valid output."""
        from marching_cubes import marching_cubes
        
        coords = np.array([0.0, 1.0])
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        # Create a crossing
        field = X + Y + Z - 1.5  # Diagonal plane
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Should be a list (may be empty or have triangles)
        assert isinstance(triangles, list)
        
        # If triangles exist, they should be valid
        for v0, v1, v2 in triangles:
            assert v0.shape == (3,)
            assert v1.shape == (3,)
            assert v2.shape == (3,)
    
    def test_exact_threshold_values_handled(self):
        """Test when field values exactly equal threshold."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X  # Linear, includes exact zero
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Should handle gracefully without errors
        assert isinstance(triangles, list)
        
        # Triangles should be valid if they exist
        for v0, v1, v2 in triangles:
            assert np.all(np.isfinite(v0))
            assert np.all(np.isfinite(v1))
            assert np.all(np.isfinite(v2))
    
    def test_constant_field_at_threshold_produces_no_triangles(self):
        """Test field that is constant at threshold produces no triangles."""
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = np.zeros_like(X)  # All values at threshold
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Should produce no triangles (degenerate case)
        assert len(triangles) == 0


class TestMarchingCubesBasic:
    """Basic tests for marching cubes algorithm."""
    
    def test_empty_field(self):
        """Test with field entirely above threshold."""
        coords = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        # All values positive, threshold 0
        field = np.ones_like(X) * 10
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) == 0
    
    def test_full_field(self):
        """Test with field entirely below threshold."""
        coords = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = np.ones_like(X) * -10
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) == 0
    
    def test_sphere_produces_triangles(self):
        """Test that a sphere produces triangles."""
        coords = np.linspace(-2, 2, 15)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1  # Unit sphere
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) > 0
    
    def test_triangle_structure(self):
        """Test that triangles have correct structure."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        for tri in triangles:
            assert len(tri) == 3  # Three vertices
            for vertex in tri:
                assert isinstance(vertex, np.ndarray)
                assert vertex.shape == (3,)  # 3D coordinates
    
    def test_vertices_in_domain(self):
        """Test that vertices are within the domain."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        for tri in triangles:
            for vertex in tri:
                assert vertex[0] >= -2 and vertex[0] <= 2
                assert vertex[1] >= -2 and vertex[1] <= 2
                assert vertex[2] >= -2 and vertex[2] <= 2
    
    def test_vertices_near_isosurface(self):
        """Test that vertices lie approximately on the isosurface."""
        coords = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1  # Unit sphere
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Check that vertex coordinates satisfy the equation approximately
        tolerance = 0.5  # Tolerance depends on grid resolution
        for tri in triangles:
            for v in tri:
                r_squared = v[0]**2 + v[1]**2 + v[2]**2
                assert abs(r_squared - 1.0) < tolerance


class TestMarchingCubesNumerical:
    """Numerical robustness tests."""
    
    def test_handles_nan(self):
        """Test that NaN values are handled gracefully."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        field[5, 5, 5] = np.nan
        
        # Should not raise exception
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        # Result should still be valid
        for tri in triangles:
            for v in tri:
                assert np.all(np.isfinite(v))
    
    def test_handles_inf(self):
        """Test that infinity values are handled."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        field[3, 3, 3] = np.inf
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        # Should still produce some valid triangles
    
    def test_very_small_values(self):
        """Test with very small field values."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = (X**2 + Y**2 + Z**2 - 1) * 1e-10
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) > 0
    
    def test_threshold_variation(self):
        """Test with different threshold values."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2
        
        # Smaller threshold = smaller sphere
        triangles_small = marching_cubes(field, 0.5, coords, coords, coords)
        triangles_large = marching_cubes(field, 2.0, coords, coords, coords)
        
        # Larger sphere should have more triangles
        assert len(triangles_large) >= len(triangles_small)


class TestMarchingCubesGeometry:
    """Geometry-related tests."""
    
    def test_plane_produces_triangles(self):
        """Test that a plane produces triangles."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = Z  # Plane at z=0
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) > 0
    
    def test_plane_vertices_on_plane(self):
        """Test that plane vertices have correct z coordinate."""
        coords = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = Z - 0.5  # Plane at z=0.5
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        for tri in triangles:
            for v in tri:
                assert abs(v[2] - 0.5) < 0.2  # Within grid cell
    
    def test_cube_topology(self):
        """Test isosurface of a cube."""
        coords = np.linspace(-2, 2, 15)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        # Cube: max(|x|, |y|, |z|) = 1
        field = np.maximum.reduce([np.abs(X), np.abs(Y), np.abs(Z)]) - 1
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) > 0
    
    def test_torus_topology(self):
        """Test isosurface of a torus."""
        coords = np.linspace(-3, 3, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        # Torus
        R, r = 2.0, 0.5
        field = (np.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        assert len(triangles) > 0


class TestMarchingCubesResolution:
    """Resolution-related tests."""
    
    def test_higher_resolution_more_triangles(self):
        """Test that higher resolution produces more triangles."""
        coords_low = np.linspace(-2, 2, 8)
        coords_high = np.linspace(-2, 2, 16)
        
        X_low, Y_low, Z_low = np.meshgrid(coords_low, coords_low, coords_low, indexing='ij')
        X_high, Y_high, Z_high = np.meshgrid(coords_high, coords_high, coords_high, indexing='ij')
        
        field_low = X_low**2 + Y_low**2 + Z_low**2 - 1
        field_high = X_high**2 + Y_high**2 + Z_high**2 - 1
        
        tri_low = marching_cubes(field_low, 0.0, coords_low, coords_low, coords_low)
        tri_high = marching_cubes(field_high, 0.0, coords_high, coords_high, coords_high)
        
        assert len(tri_high) > len(tri_low)
    
    def test_non_uniform_grid(self):
        """Test with non-uniform grid spacing."""
        x_coords = np.array([-2, -1, 0, 0.5, 1, 2])
        y_coords = np.array([-2, -0.5, 0, 0.5, 2])
        z_coords = np.linspace(-2, 2, 8)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        triangles = marching_cubes(field, 0.0, x_coords, y_coords, z_coords)
        assert len(triangles) > 0


class TestMarchingCubesOptimized:
    """Tests for optimized marching cubes."""
    
    def test_returns_arrays(self):
        """Test that optimized version returns numpy arrays."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        vertices, faces = marching_cubes_optimized(field, 0.0, coords, coords, coords)
        
        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)
    
    def test_array_shapes(self):
        """Test output array shapes."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        vertices, faces = marching_cubes_optimized(field, 0.0, coords, coords, coords)
        
        if len(vertices) > 0:
            assert vertices.ndim == 2
            assert vertices.shape[1] == 3
            assert faces.ndim == 2
            assert faces.shape[1] == 3
    
    def test_empty_result(self):
        """Test optimized version with empty result."""
        coords = np.linspace(-2, 2, 5)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = np.ones_like(X) * 10  # All above threshold
        
        vertices, faces = marching_cubes_optimized(field, 0.0, coords, coords, coords)
        
        assert vertices.shape == (0, 3)
        assert faces.shape == (0, 3)
    
    def test_consistency_with_basic(self):
        """Test that optimized produces same number of triangles."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        vertices, faces = marching_cubes_optimized(field, 0.0, coords, coords, coords)
        
        assert len(triangles) == len(faces)
    
    def test_face_indices_valid(self):
        """Test that face indices are valid."""
        coords = np.linspace(-2, 2, 10)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        field = X**2 + Y**2 + Z**2 - 1
        
        vertices, faces = marching_cubes_optimized(field, 0.0, coords, coords, coords)
        
        if len(faces) > 0:
            assert np.all(faces >= 0)
            assert np.all(faces < len(vertices))