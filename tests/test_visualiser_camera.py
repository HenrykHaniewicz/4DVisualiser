"""
Tests for visualiser.py - Camera class.
"""

import pytest
import numpy as np
import sys
from unittest.mock import MagicMock

sys.modules['pygame'] = MagicMock()


class TestCamera:
    """Tests for Camera class."""
    
    @pytest.fixture
    def camera(self):
        """Create Camera instance."""
        from visualiser import Camera, Config
        config = Config()
        return Camera(config)
    
    @pytest.fixture
    def config(self):
        """Create Config instance."""
        from visualiser import Config
        return Config()
    
    def test_initial_values(self, camera, config):
        """Test camera initializes with config values."""
        assert camera.elevation == config.INIT_ELEV
        assert camera.azimuth == config.INIT_AZIM
        assert camera.zoom == config.INIT_ZOOM
    
    def test_center_calculation(self, camera, config):
        """Test center is calculated correctly."""
        expected_x = config.VIEW_X + config.VIEW_WIDTH // 2
        expected_y = config.VIEW_Y + config.VIEW_HEIGHT // 2
        assert camera.center_x == expected_x
        assert camera.center_y == expected_y
    
    def test_light_direction_normalized(self, camera):
        """Test light direction is normalized."""
        length = np.linalg.norm(camera.light_dir)
        assert np.isclose(length, 1.0)
    
    def test_rotation_matrix_shape(self, camera):
        """Test rotation matrix has correct shape."""
        R = camera.get_rotation_matrix()
        assert R.shape == (3, 3)
    
    def test_rotation_matrix_orthogonal(self, camera):
        """Test rotation matrix is orthogonal."""
        R = camera.get_rotation_matrix()
        identity = R @ R.T
        np.testing.assert_array_almost_equal(identity, np.eye(3))
    
    def test_rotation_matrix_determinant(self, camera):
        """Test rotation matrix has determinant 1."""
        R = camera.get_rotation_matrix()
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0)
    
    def test_rotate_azimuth(self, camera):
        """Test azimuth rotation."""
        initial_azim = camera.azimuth
        camera.rotate(10, 0)
        assert camera.azimuth == initial_azim + 10
    
    def test_rotate_elevation(self, camera):
        """Test elevation rotation."""
        initial_elev = camera.elevation
        camera.rotate(0, 5)
        assert camera.elevation == initial_elev + 5
    
    def test_elevation_clamping_upper(self, camera):
        """Test elevation is clamped at upper limit."""
        camera.rotate(0, 1000)
        assert camera.elevation <= 89
    
    def test_elevation_clamping_lower(self, camera):
        """Test elevation is clamped at lower limit."""
        camera.rotate(0, -1000)
        assert camera.elevation >= -89
    
    def test_zoom_in(self, camera):
        """Test zoom in increases zoom."""
        initial_zoom = camera.zoom
        camera.zoom_in(1.5)
        assert camera.zoom > initial_zoom
    
    def test_zoom_out(self, camera):
        """Test zoom out decreases zoom."""
        initial_zoom = camera.zoom
        camera.zoom_in(0.5)
        assert camera.zoom < initial_zoom
    
    def test_zoom_clamping_upper(self, camera):
        """Test zoom is clamped at upper limit."""
        camera.zoom_in(1000)
        assert camera.zoom <= 500
    
    def test_zoom_clamping_lower(self, camera):
        """Test zoom is clamped at lower limit."""
        camera.zoom_in(0.001)
        assert camera.zoom >= 20
    
    def test_reset(self, camera, config):
        """Test reset restores initial values."""
        camera.rotate(45, 30)
        camera.zoom_in(2)
        camera.reset()
        
        assert camera.elevation == config.INIT_ELEV
        assert camera.azimuth == config.INIT_AZIM
        assert camera.zoom == config.INIT_ZOOM
    
    def test_rotation_updates_matrix(self, camera):
        """Test that rotation updates the matrix."""
        R1 = camera.get_rotation_matrix().copy()
        camera.rotate(45, 0)
        R2 = camera.get_rotation_matrix()
        
        assert not np.allclose(R1, R2)
    
    def test_zero_rotation(self, camera):
        """Test rotation matrix at zero angles."""
        camera.elevation = 0
        camera.azimuth = 0
        camera._update_rotation_matrix()
        R = camera.get_rotation_matrix()
        
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_90_degree_azimuth(self, camera):
        """Test 90 degree azimuth rotation."""
        camera.elevation = 0
        camera.azimuth = 90
        camera._update_rotation_matrix()
        R = camera.get_rotation_matrix()
        
        # X axis should map to Y axis
        result = R @ np.array([1, 0, 0])
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)