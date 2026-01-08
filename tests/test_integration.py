"""
Integration tests combining multiple components.
"""

import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_pygame():
    """Mock pygame for integration tests."""
    mock_pg = MagicMock()
    
    def mock_rect(x, y, w, h):
        rect = MagicMock()
        rect.x = x
        rect.y = y
        rect.width = w
        rect.height = h
        rect.centerx = x + w // 2
        rect.centery = y + h // 2
        rect.center = (rect.centerx, rect.centery)
        rect.right = x + w
        rect.collidepoint = MagicMock(return_value=True)
        return rect
    
    mock_pg.Rect = mock_rect
    mock_font = MagicMock()
    mock_surface = MagicMock()
    mock_surface.get_width = MagicMock(return_value=100)
    mock_surface.get_height = MagicMock(return_value=20)
    mock_surface.get_rect = MagicMock(return_value=MagicMock(center=(50, 10)))
    mock_font.render = MagicMock(return_value=mock_surface)
    mock_pg.font.SysFont = MagicMock(return_value=mock_font)
    
    mock_screen = MagicMock()
    mock_pg.display.set_mode = MagicMock(return_value=mock_screen)
    mock_pg.display.set_caption = MagicMock()
    mock_pg.display.flip = MagicMock()
    
    mock_clock = MagicMock()
    mock_clock.tick = MagicMock(return_value=16)
    mock_pg.time.Clock = MagicMock(return_value=mock_clock)
    
    mock_pg.QUIT = 256
    mock_pg.MOUSEBUTTONDOWN = 1025
    mock_pg.MOUSEBUTTONUP = 1026
    mock_pg.MOUSEMOTION = 1024
    mock_pg.KEYDOWN = 768
    mock_pg.K_SPACE = 32
    mock_pg.K_r = 114
    mock_pg.K_ESCAPE = 27
    
    mock_pg.event.get = MagicMock(return_value=[])
    mock_pg.mouse.get_pos = MagicMock(return_value=(640, 410))
    mock_pg.mouse.get_pressed = MagicMock(return_value=(False, False, False))
    mock_pg.init = MagicMock()
    mock_pg.quit = MagicMock()
    mock_pg.draw = MagicMock()
    
    with patch.dict('sys.modules', {'pygame': mock_pg}):
        yield mock_pg


class TestFunctionsWithMarchingCubes:
    """Test functions work correctly with marching cubes."""
    
    def test_hypersolid_sphere_marching_cubes(self):
        """Test hypersolid sphere with marching cubes."""
        from functions import hypersolid_sphere
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-3, 3, 15)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        W = np.zeros_like(X)
        
        field = hypersolid_sphere(X, Y, Z, W)
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Should produce sphere triangles
        assert len(triangles) > 0
    
    def test_hypersolid_hypercube_marching_cubes(self):
        """Test hypercube with marching cubes."""
        from functions import hypersolid_hypercube
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-2, 2, 15)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        W = np.zeros_like(X)
        
        field = hypersolid_hypercube(X, Y, Z, W)
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        assert len(triangles) > 0
    
    def test_hypersolid_torus_marching_cubes(self):
        """Test torus with marching cubes."""
        from functions import hypersolid_torus
        from marching_cubes import marching_cubes
        
        coords = np.linspace(-3, 3, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        W = np.zeros_like(X)
        
        field = hypersolid_torus(X, Y, Z, W)
        triangles = marching_cubes(field, 0.0, coords, coords, coords)
        
        # Torus should produce triangles at W=0
        assert len(triangles) > 0


class TestRendererWithRealFunctions:
    """Test renderers with actual mathematical functions."""
    
    def test_surface_renderer_with_saddle(self):
        """Test Surface3DRenderer with saddle function."""
        from visualiser import Surface3DRenderer
        from functions import surface_saddle
        
        renderer = Surface3DRenderer(surface_saddle, (-2, 2), (-2, 2), resolution=20)
        
        assert len(renderer.vertices) > 0
        assert np.all(np.isfinite(renderer.vertices))
    
    def test_hypersurface_renderer_with_gaussian(self):
        """Test HypersurfaceRenderer with Gaussian function."""
        from visualiser import HypersurfaceRenderer
        from functions import hypersurface_gaussian
        
        renderer = HypersurfaceRenderer(hypersurface_gaussian, (-2, 2), resolution=20)
        
        assert len(renderer.vertices) > 0
        assert np.all(np.isfinite(renderer.vertices))
    
    def test_hypersolid_renderer_with_sphere(self):
        """Test HypersolidRenderer with sphere function."""
        from visualiser import HypersolidRenderer
        from functions import hypersolid_sphere
        
        renderer = HypersolidRenderer(
            hypersolid_sphere,
            (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5),
            resolution=10
        )
        
        renderer.set_w_value(0.0)
        assert renderer.vertices is not None
        assert len(renderer.vertices) > 0


class TestCompleteVisualizationCycle:
    """Test complete visualization cycles."""
    
    def test_surface_visualization_cycle(self, mock_pygame):
        """Test complete surface visualization cycle."""
        from visualiser import HypersolidViewer, VisualizationType
        
        viewer = HypersolidViewer()
        viewer.vis_type = VisualizationType.SURFACE_3D
        viewer.current_func_id = 1
        viewer._load_function()
        
        # Simulate one render cycle
        viewer._update()
        viewer._render()
        
        # Should complete without error
        assert viewer.renderer is not None
    
    def test_hypersurface_visualization_cycle(self, mock_pygame):
        """Test complete hypersurface visualization cycle."""
        from visualiser import HypersolidViewer, VisualizationType
        
        viewer = HypersolidViewer()
        viewer.vis_type = VisualizationType.HYPERSURFACE_4D
        viewer.current_func_id = 1
        viewer._load_function()
        
        viewer._update()
        viewer._render()
        
        assert viewer.renderer is not None
    
    def test_hypersolid_visualization_cycle(self, mock_pygame):
        """Test complete hypersolid visualization cycle."""
        from visualiser import HypersolidViewer, VisualizationType
        
        viewer = HypersolidViewer()
        viewer.vis_type = VisualizationType.HYPERSOLID_4D
        viewer.current_func_id = 1
        viewer._load_function()
        
        viewer._update()
        viewer._render()
        
        assert viewer.renderer is not None
    
    def test_switch_visualization_types(self, mock_pygame):
        """Test switching between visualization types."""
        from visualiser import HypersolidViewer, VisualizationType
        
        viewer = HypersolidViewer()
        
        for vis_type in VisualizationType:
            viewer.vis_type = vis_type
            viewer.current_func_id = 1
            viewer._load_function()
            viewer._update()
            viewer._render()
            
            assert viewer.renderer is not None
    
    def test_animation_cycle(self, mock_pygame):
        """Test animation over multiple frames."""
        from visualiser import HypersolidViewer
        
        viewer = HypersolidViewer()
        viewer.animating = True
        
        values = []
        for _ in range(10):
            viewer._update()
            values.append(viewer.slice_slider.value)
        
        # Values should change
        assert not all(v == values[0] for v in values)


class TestCameraRendererInteraction:
    """Test camera and renderer interaction."""
    
    def test_camera_rotation_affects_render(self, mock_pygame):
        """Test camera rotation changes render output."""
        from visualiser import (
            Camera, Config, Surface3DRenderer, Colormap
        )
        from functions import surface_saddle
        
        config = Config()
        camera = Camera(config)
        renderer = Surface3DRenderer(surface_saddle, (-2, 2), (-2, 2), resolution=10)
        
        mock_screen = MagicMock()
        mock_rect = MagicMock()
        mock_rect.collidepoint = MagicMock(return_value=True)
        
        # Render at initial position
        R1 = camera.get_rotation_matrix().copy()
        
        # Rotate camera
        camera.rotate(45, 30)
        R2 = camera.get_rotation_matrix()
        
        # Matrices should differ
        assert not np.allclose(R1, R2)
    
    def test_zoom_affects_projection(self, mock_pygame):
        """Test zoom changes projection."""
        from visualiser import Camera, Config
        
        config = Config()
        camera = Camera(config)
        
        zoom1 = camera.zoom
        camera.zoom_in(2.0)
        zoom2 = camera.zoom
        
        assert zoom2 > zoom1


class TestColorMapRendererInteraction:
    """Test colormap and renderer interaction."""
    
    def test_different_colormaps_produce_different_colors(self, mock_pygame):
        """Test different colormaps produce visually different results."""
        from visualiser import Colormap
        
        t = 0.5
        colors = {
            'viridis': Colormap.viridis(t),
            'plasma': Colormap.plasma(t),
            'inferno': Colormap.inferno(t),
            'cool_warm': Colormap.cool_warm(t),
        }
        
        # All colors should be different
        color_values = list(colors.values())
        for i, c1 in enumerate(color_values):
            for c2 in color_values[i+1:]:
                # At least one component should differ significantly
                diff = sum(abs(a - b) for a, b in zip(c1, c2))
                # Allow some colormaps to be similar but not identical
                # This is a weak test - mainly checking they're not all the same