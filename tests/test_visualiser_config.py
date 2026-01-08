"""
Tests for visualiser.py - Configuration classes.
"""

import pytest
from dataclasses import fields


class TestConfig:
    """Tests for Config dataclass."""
    
    @pytest.fixture
    def config(self):
        """Create Config instance."""
        # Import here to avoid pygame initialization issues
        import sys
        from unittest.mock import MagicMock
        sys.modules['pygame'] = MagicMock()
        from visualiser import Config
        return Config()
    
    def test_default_window_dimensions(self, config):
        """Test default window dimensions."""
        assert config.WINDOW_WIDTH == 1280
        assert config.WINDOW_HEIGHT == 820
    
    def test_default_fps(self, config):
        """Test default FPS."""
        assert config.FPS == 60
    
    def test_color_tuples(self, config):
        """Test color values are valid tuples."""
        color_attrs = ['BG_COLOR', 'UI_BG_COLOR', 'UI_BORDER_COLOR', 
                       'TEXT_COLOR', 'HIGHLIGHT_COLOR', 'ACCENT_COLOR']
        
        for attr in color_attrs:
            color = getattr(config, attr)
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_view_dimensions(self, config):
        """Test view area dimensions."""
        assert config.VIEW_WIDTH > 0
        assert config.VIEW_HEIGHT > 0
        assert config.VIEW_X >= 0
        assert config.VIEW_Y >= 0
    
    def test_view_fits_in_window(self, config):
        """Test view fits within window."""
        assert config.VIEW_X + config.VIEW_WIDTH <= config.WINDOW_WIDTH
        assert config.VIEW_Y + config.VIEW_HEIGHT <= config.WINDOW_HEIGHT
    
    def test_initial_camera_values(self, config):
        """Test initial camera values are sensible."""
        assert -90 <= config.INIT_ELEV <= 90
        assert config.INIT_ZOOM > 0
    
    def test_hypersolid_resolution(self, config):
        """Test hypersolid resolution."""
        assert config.HYPERSOLID_RESOLUTION > 0
        assert config.HYPERSOLID_RESOLUTION <= 50  # Reasonable upper limit


class TestVisualizationType:
    """Tests for VisualizationType enum."""
    
    @pytest.fixture
    def vis_type(self):
        """Get VisualizationType enum."""
        import sys
        from unittest.mock import MagicMock
        sys.modules['pygame'] = MagicMock()
        from visualiser import VisualizationType
        return VisualizationType
    
    def test_enum_values_exist(self, vis_type):
        """Test all expected enum values exist."""
        assert hasattr(vis_type, 'SURFACE_3D')
        assert hasattr(vis_type, 'HYPERSURFACE_4D')
        assert hasattr(vis_type, 'HYPERSOLID_4D')
    
    def test_enum_uniqueness(self, vis_type):
        """Test enum values are unique."""
        values = [vis_type.SURFACE_3D, vis_type.HYPERSURFACE_4D, vis_type.HYPERSOLID_4D]
        assert len(values) == len(set(v.value for v in values))
    
    def test_enum_iteration(self, vis_type):
        """Test enum can be iterated."""
        types = list(vis_type)
        assert len(types) == 3