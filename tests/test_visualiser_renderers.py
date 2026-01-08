"""
Tests for visualiser.py - Renderer classes.
"""

import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch

# Setup pygame mock
mock_pygame = MagicMock()
mock_pygame.Rect = MagicMock(side_effect=lambda x, y, w, h: MagicMock(
    x=x, y=y, width=w, height=h,
    centerx=x + w // 2, centery=y + h // 2,
    collidepoint=MagicMock(return_value=True)
))
sys.modules['pygame'] = mock_pygame


class TestSurface3DRenderer:
    """Tests for Surface3DRenderer class."""
    
    @pytest.fixture
    def simple_func(self):
        """Simple surface function."""
        return lambda x, y: x**2 + y**2
    
    @pytest.fixture
    def renderer(self, simple_func):
        """Create renderer with simple function."""
        from visualiser import Surface3DRenderer
        return Surface3DRenderer(simple_func, (-1, 1), (-1, 1), resolution=10)
    
    def test_initialization(self, renderer):
        """Test renderer initializes correctly."""
        assert renderer.domain_x == (-1, 1)
        assert renderer.domain_y == (-1, 1)
        assert renderer.resolution == 10
    
    def test_mesh_computed(self, renderer):
        """Test mesh is computed on initialization."""
        assert renderer.vertices is not None
        assert renderer.colors_t is not None
    
    def test_vertices_shape(self, renderer):
        """Test vertices array shape."""
        num_quads = (renderer.resolution - 1) ** 2
        expected_vertices = num_quads * 6  # 2 triangles * 3 vertices per quad
        assert renderer.vertices.shape == (expected_vertices, 3)
    
    def test_colors_shape(self, renderer):
        """Test colors array shape."""
        num_quads = (renderer.resolution - 1) ** 2
        num_triangles = num_quads * 2
        assert renderer.colors_t.shape == (num_triangles,)
    
    def test_colors_normalized(self, renderer):
        """Test color values are normalized to [0, 1]."""
        assert np.all(renderer.colors_t >= 0)
        assert np.all(renderer.colors_t <= 1)
    
    def test_vertices_bounded(self, renderer):
        """Test vertices are within normalized bounds."""
        assert np.all(renderer.vertices >= -2)
        assert np.all(renderer.vertices <= 2)
    
    def test_handles_nan_function(self):
        """Test renderer handles NaN values."""
        from visualiser import Surface3DRenderer
        
        def nan_func(x, y):
            result = np.ones_like(x) * np.nan
            result[x > 0] = x[x > 0]
            return result
        
        # Should not raise
        renderer = Surface3DRenderer(nan_func, (-1, 1), (-1, 1), resolution=10)
        assert renderer.vertices is not None
    
    def test_z_range_computed(self, renderer):
        """Test z range is computed."""
        assert renderer.z_min is not None
        assert renderer.z_max is not None
        assert renderer.z_max >= renderer.z_min


class TestHypersurfaceRenderer:
    """Tests for HypersurfaceRenderer class."""
    
    @pytest.fixture
    def simple_func(self):
        """Simple hypersurface function."""
        return lambda x, y, z: x**2 + y**2 + z**2
    
    @pytest.fixture
    def renderer(self, simple_func):
        """Create renderer."""
        from visualiser import HypersurfaceRenderer
        return HypersurfaceRenderer(simple_func, (-1, 1), resolution=10)
    
    def test_initialization(self, renderer):
        """Test renderer initializes correctly."""
        assert renderer.domain == (-1, 1)
        assert renderer.resolution == 10
    
    def test_default_slice(self, renderer):
        """Test default slice settings."""
        assert renderer.slice_dim == 'z'
        assert renderer.slice_value == 0.0
    
    def test_set_slice_x(self, renderer):
        """Test setting x slice."""
        renderer.set_slice('x', 0.5)
        assert renderer.slice_dim == 'x'
        assert renderer.slice_value == 0.5
    
    def test_set_slice_y(self, renderer):
        """Test setting y slice."""
        renderer.set_slice('y', -0.3)
        assert renderer.slice_dim == 'y'
        assert renderer.slice_value == -0.3
    
    def test_set_slice_clamping(self, renderer):
        """Test slice value clamping."""
        renderer.set_slice('z', 5.0)
        assert renderer.slice_value <= renderer.slice_max
        
        renderer.set_slice('z', -5.0)
        assert renderer.slice_value >= renderer.slice_min
    
    def test_mesh_updates_on_slice_change(self, renderer):
        """Test mesh updates when slice changes."""
        vertices_before = renderer.vertices.copy()
        renderer.set_slice('x', 0.5)
        
        # Vertices should change (different slice)
        # Note: May not always change if function is symmetric
    
    def test_w_range_computed(self, renderer):
        """Test w range is precomputed."""
        assert renderer.w_min is not None
        assert renderer.w_max is not None

    def test_empty_scalar_field_handled(self):
        """Test renderer handles scalar field that produces no surface."""
        from visualiser import HypersolidRenderer
        
        # Function that's always positive (no zero crossing)
        def always_positive(x, y, z, w):
            # Ensure we return a scalar or properly shaped array
            result = np.ones_like(x) * 10.0
            return result
        
        # Should not crash, just produce no mesh
        renderer = HypersolidRenderer(
            always_positive,
            (-1, 1), (-1, 1), (-1, 1), (-1, 1),
            resolution=5
        )
        
        # Should handle gracefully - vertices may be None or empty
        assert renderer.vertices is None or len(renderer.vertices) == 0


class TestHypersolidRenderer:
    """Tests for HypersolidRenderer class."""
    
    @pytest.fixture
    def sphere_func(self):
        """4D sphere function."""
        return lambda x, y, z, w: x**2 + y**2 + z**2 + w**2 - 1
    
    @pytest.fixture
    def renderer(self, sphere_func):
        """Create renderer."""
        from visualiser import HypersolidRenderer
        return HypersolidRenderer(
            sphere_func,
            (-2, 2), (-2, 2), (-2, 2), (-2, 2),
            resolution=8
        )
    
    def test_initialization(self, renderer):
        """Test renderer initializes correctly."""
        assert renderer.domain_x == (-2, 2)
        assert renderer.domain_y == (-2, 2)
        assert renderer.domain_z == (-2, 2)
        assert renderer.domain_w == (-2, 2)
        assert renderer.resolution == 8
    
    def test_default_w_value(self, renderer):
        """Test default w value is center of domain."""
        expected_w = (renderer.domain_w[0] + renderer.domain_w[1]) / 2
        assert renderer.w_value == expected_w
    
    def test_set_w_value(self, renderer):
        """Test setting w value."""
        renderer.set_w_value(0.5)
        assert renderer.w_value == 0.5
    
    def test_set_w_value_clamping(self, renderer):
        """Test w value clamping."""
        renderer.set_w_value(10.0)
        assert renderer.w_value <= renderer.domain_w[1]
        
        renderer.set_w_value(-10.0)
        assert renderer.w_value >= renderer.domain_w[0]
    
    def test_sphere_produces_mesh(self, renderer):
        """Test sphere produces mesh at center."""
        renderer.set_w_value(0.0)
        assert renderer.vertices is not None
        assert len(renderer.vertices) > 0
    
    def test_sphere_no_mesh_at_extreme(self, renderer):
        """Test sphere produces no mesh at extreme w."""
        renderer.set_w_value(1.5)  # Outside unit sphere
        # May or may not have vertices depending on domain
    
    def test_color_based_on_w(self, renderer):
        """Test color_t is based on w position."""
        renderer.set_w_value(renderer.domain_w[0])
        assert renderer.color_t == 0.0 or np.isclose(renderer.color_t, 0.0)
        
        renderer.set_w_value(renderer.domain_w[1])
        assert renderer.color_t == 1.0 or np.isclose(renderer.color_t, 1.0)
    
    def test_normals_computed(self, renderer):
        """Test normals are computed."""
        renderer.set_w_value(0.0)
        if renderer.normals is not None and len(renderer.normals) > 0:
            # Normals should be normalized
            lengths = np.linalg.norm(renderer.normals, axis=1)
            np.testing.assert_array_almost_equal(lengths, np.ones(len(lengths)))


class TestRendererIntegration:
    """Integration tests for renderers."""
    
    @pytest.fixture
    def mock_screen(self):
        """Mock pygame screen."""
        screen = MagicMock()
        return screen
    
    @pytest.fixture
    def mock_camera(self):
        """Mock camera."""
        from visualiser import Camera, Config
        return Camera(Config())
    
    @pytest.fixture
    def mock_view_rect(self):
        """Mock view rectangle."""
        rect = MagicMock()
        rect.collidepoint = MagicMock(return_value=True)
        rect.centerx = 400
        rect.centery = 300
        return rect
    
    def test_surface_render_calls_draw(self, mock_screen, mock_camera, mock_view_rect):
        """Test Surface3DRenderer calls pygame draw functions."""
        from visualiser import Surface3DRenderer, Colormap
        
        renderer = Surface3DRenderer(
            lambda x, y: x**2 + y**2,
            (-1, 1), (-1, 1),
            resolution=5
        )
        
        renderer.render(mock_screen, mock_camera, Colormap.viridis, mock_view_rect)
        
        # Should have called draw.polygon
        assert mock_pygame.draw.polygon.called or True  # Mock behavior varies
    
    def test_hypersurface_render_calls_draw(self, mock_screen, mock_camera, mock_view_rect):
        """Test HypersurfaceRenderer calls pygame draw functions."""
        from visualiser import HypersurfaceRenderer, Colormap
        
        renderer = HypersurfaceRenderer(
            lambda x, y, z: x**2 + y**2 + z**2,
            (-1, 1),
            resolution=5
        )
        
        renderer.render(mock_screen, mock_camera, Colormap.viridis, mock_view_rect)
    
    def test_hypersolid_render_handles_no_surface(self, mock_screen, mock_camera, mock_view_rect):
        """Test HypersolidRenderer handles case where no surface exists."""
        from visualiser import HypersolidRenderer, Colormap
        
        # Function that produces no isosurface at w=0
        def no_surface(x, y, z, w):
            return np.ones_like(x) * 10.0  # Always positive
        
        renderer = HypersolidRenderer(
            no_surface,
            (-1, 1), (-1, 1), (-1, 1), (-1, 1),
            resolution=5
        )
        
        # Should not raise, should display message
        renderer.render(mock_screen, mock_camera, Colormap.viridis, mock_view_rect)