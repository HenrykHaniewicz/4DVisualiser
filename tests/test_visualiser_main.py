"""
Tests for visualiser.py - Main application class.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch, PropertyMock, call
import numpy as np


@pytest.fixture(autouse=True)
def mock_pygame_module():
    """Mock pygame module for all tests in this file."""
    mock_pg = MagicMock()
    
    # Mock Rect properly - return a proper mock with all needed attributes
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
        rect.top = y
        rect.bottom = y + h
        # Always return False for collidepoint to avoid triggering click handlers
        rect.collidepoint = MagicMock(return_value=False)
        return rect
    
    mock_pg.Rect = mock_rect
    
    # Mock font
    mock_font = MagicMock()
    mock_surface = MagicMock()
    mock_surface.get_width = MagicMock(return_value=100)
    mock_surface.get_height = MagicMock(return_value=20)
    mock_surface.get_rect = MagicMock(return_value=MagicMock(center=(50, 10)))
    mock_font.render = MagicMock(return_value=mock_surface)
    mock_pg.font.SysFont = MagicMock(return_value=mock_font)
    
    # Mock display
    mock_screen = MagicMock()
    mock_screen.fill = MagicMock()
    mock_screen.blit = MagicMock()
    mock_pg.display.set_mode = MagicMock(return_value=mock_screen)
    mock_pg.display.set_caption = MagicMock()
    mock_pg.display.flip = MagicMock()
    
    # Mock clock
    mock_clock = MagicMock()
    mock_clock.tick = MagicMock(return_value=16)
    mock_pg.time.Clock = MagicMock(return_value=mock_clock)
    
    # Mock events - use actual integer values
    mock_pg.QUIT = 256
    mock_pg.MOUSEBUTTONDOWN = 1025
    mock_pg.MOUSEBUTTONUP = 1026
    mock_pg.MOUSEMOTION = 1024
    mock_pg.KEYDOWN = 768
    mock_pg.K_SPACE = 32
    mock_pg.K_r = 114
    mock_pg.K_ESCAPE = 27
    
    mock_pg.event.get = MagicMock(return_value=[])
    # Return actual tuple, not MagicMock - this is critical
    mock_pg.mouse.get_pos = MagicMock(return_value=(640, 410))
    mock_pg.mouse.get_pressed = MagicMock(return_value=(False, False, False))
    
    mock_pg.init = MagicMock()
    mock_pg.quit = MagicMock()
    mock_pg.draw = MagicMock()
    mock_pg.draw.rect = MagicMock()
    mock_pg.draw.polygon = MagicMock()
    mock_pg.draw.circle = MagicMock()
    mock_pg.draw.line = MagicMock()
    
    with patch.dict('sys.modules', {'pygame': mock_pg}):
        yield mock_pg


class TestHypersolidViewerInitialization:
    """Tests for HypersolidViewer initialization."""
    
    def test_initialization_succeeds(self, mock_pygame_module):
        """Test viewer initializes without errors."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        assert viewer is not None
    
    def test_initial_state(self, mock_pygame_module):
        """Test initial state is set correctly."""
        from visualiser import HypersolidViewer, VisualizationType
        viewer = HypersolidViewer()
        
        assert viewer.running == True
        assert viewer.vis_type == VisualizationType.HYPERSOLID_4D
        assert viewer.current_func_id == 1
        assert viewer.current_colormap == 'viridis'
    
    def test_camera_created(self, mock_pygame_module):
        """Test camera is created."""
        from visualiser import HypersolidViewer, Camera
        viewer = HypersolidViewer()
        assert viewer.camera is not None
    
    def test_renderer_created(self, mock_pygame_module):
        """Test initial renderer is created."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        assert viewer.renderer is not None
    
    def test_ui_components_created(self, mock_pygame_module):
        """Test UI components are created."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        assert len(viewer.type_buttons) == 3
        assert len(viewer.cmap_buttons) == 4
        assert len(viewer.slice_dim_buttons) == 3
        assert viewer.animate_button is not None
        assert viewer.reset_button is not None


class TestHypersolidViewerFunctionLoading: 
    """Tests for function loading."""
    
    def test_load_surface_function(self, mock_pygame_module):
        """Test loading 3D surface function."""
        from visualiser import HypersolidViewer, VisualizationType, Surface3DRenderer
        viewer = HypersolidViewer()
        
        viewer.vis_type = VisualizationType.SURFACE_3D
        viewer.current_func_id = 1
        viewer._load_function()
        
        assert isinstance(viewer.renderer, Surface3DRenderer)
    
    def test_load_hypersurface_function(self, mock_pygame_module):
        """Test loading 4D hypersurface function."""
        from visualiser import HypersolidViewer, VisualizationType, HypersurfaceRenderer
        viewer = HypersolidViewer()
        
        viewer.vis_type = VisualizationType.HYPERSURFACE_4D
        viewer.current_func_id = 1
        viewer._load_function()
        
        assert isinstance(viewer.renderer, HypersurfaceRenderer)
    
    def test_load_hypersolid_function(self, mock_pygame_module):
        """Test loading 4D hypersolid function."""
        from visualiser import HypersolidViewer, VisualizationType, HypersolidRenderer
        viewer = HypersolidViewer()
        
        viewer.vis_type = VisualizationType.HYPERSOLID_4D
        viewer.current_func_id = 1
        viewer._load_function()
        
        assert isinstance(viewer.renderer, HypersolidRenderer)
    
    def test_function_name_updated(self, mock_pygame_module):
        """Test function name is updated."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        assert viewer.current_name != ""
        assert viewer.current_formula != ""


class TestHypersolidViewerAnimation:
    """Tests for animation functionality."""
    
    def test_animation_toggle(self, mock_pygame_module):
        """Test animation can be toggled."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        assert viewer.animating == False
        viewer.animating = True
        assert viewer.animating == True
    
    def test_animation_updates_slice(self, mock_pygame_module):
        """Test animation updates slice value."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        viewer.animating = True
        initial_value = viewer.slice_slider.value
        viewer._update()
        
        assert viewer.slice_slider.value != initial_value
    
    def test_animation_direction_reversal(self, mock_pygame_module):
        """Test animation reverses at bounds."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        viewer.animating = True
        viewer.slice_slider.value = viewer.slice_slider.max_val
        viewer.animation_direction = 1
        viewer._update()
        
        assert viewer.animation_direction == -1
    
    def test_animation_speed(self, mock_pygame_module):
        """Test animation speed is applied."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        viewer.animating = True
        viewer.slice_slider.value = 0
        viewer.animation_speed = 0.1
        viewer.animation_direction = 1
        viewer._update()
        
        assert viewer.slice_slider.value == pytest.approx(0.1, abs=0.01)


class TestHypersolidViewerEventHandling:
    """Tests for event handling."""
    
    def test_quit_event(self, mock_pygame_module):
        """Test quit event stops viewer."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        # Create quit event
        quit_event = MagicMock()
        quit_event.type = mock_pygame_module.QUIT
        
        # Setup mocks to return consistent values during the entire event handling
        mock_pygame_module.event.get = MagicMock(return_value=[quit_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(640, 410))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        # Patch the module-level pygame to ensure consistency
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.running == False
    
    def test_escape_key_quits(self, mock_pygame_module):
        """Test escape key stops viewer."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        # Create escape key event
        key_event = MagicMock()
        key_event.type = mock_pygame_module.KEYDOWN
        key_event.key = mock_pygame_module.K_ESCAPE
        
        mock_pygame_module.event.get = MagicMock(return_value=[key_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(640, 410))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.running == False
    
    def test_space_toggles_animation(self, mock_pygame_module):
        """Test space key toggles animation."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        # Create space key event
        key_event = MagicMock()
        key_event.type = mock_pygame_module.KEYDOWN
        key_event.key = mock_pygame_module.K_SPACE
        
        mock_pygame_module.event.get = MagicMock(return_value=[key_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(640, 410))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        initial_state = viewer.animating
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.animating == (not initial_state)
    
    def test_r_key_resets_view(self, mock_pygame_module):
        """Test R key resets camera."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        # Change camera
        viewer.camera.rotate(45, 30)
        
        # Create R key event
        key_event = MagicMock()
        key_event.type = mock_pygame_module.KEYDOWN
        key_event.key = mock_pygame_module.K_r
        
        mock_pygame_module.event.get = MagicMock(return_value=[key_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(640, 410))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        # Camera should be reset
        assert viewer.camera.elevation == viewer.config.INIT_ELEV
        assert viewer.camera.azimuth == viewer.config.INIT_AZIM
    
    def test_mouse_drag_rotates(self, mock_pygame_module):
        """Test mouse drag rotates camera."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        # Setup view_rect to detect mouse in view area
        viewer.view_rect = MagicMock()
        viewer.view_rect.collidepoint = MagicMock(return_value=True)
        
        initial_azim = viewer.camera.azimuth
        
        # Mouse down event
        down_event = MagicMock()
        down_event.type = mock_pygame_module.MOUSEBUTTONDOWN
        down_event.button = 1
        
        mock_pygame_module.event.get = MagicMock(return_value=[down_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(400, 300))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(True, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.dragging == True
        
        # Mouse motion event
        motion_event = MagicMock()
        motion_event.type = mock_pygame_module.MOUSEMOTION
        
        mock_pygame_module.event.get = MagicMock(return_value=[motion_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(450, 300))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(True, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        # Camera should have rotated
        assert viewer.camera.azimuth != initial_azim
    
    def test_scroll_wheel_zoom_in(self, mock_pygame_module):
        """Test scroll wheel zooms in."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        viewer.view_rect = MagicMock()
        viewer.view_rect.collidepoint = MagicMock(return_value=True)
        
        initial_zoom = viewer.camera.zoom
        
        # Scroll up event
        scroll_event = MagicMock()
        scroll_event.type = mock_pygame_module.MOUSEBUTTONDOWN
        scroll_event.button = 4  # Scroll up
        
        mock_pygame_module.event.get = MagicMock(return_value=[scroll_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(400, 300))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.camera.zoom > initial_zoom
    
    def test_scroll_wheel_zoom_out(self, mock_pygame_module):
        """Test scroll wheel zooms out."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        viewer.view_rect = MagicMock()
        viewer.view_rect.collidepoint = MagicMock(return_value=True)
        
        initial_zoom = viewer.camera.zoom
        
        # Scroll down event
        scroll_event = MagicMock()
        scroll_event.type = mock_pygame_module.MOUSEBUTTONDOWN
        scroll_event.button = 5  # Scroll down
        
        mock_pygame_module.event.get = MagicMock(return_value=[scroll_event])
        mock_pygame_module.mouse.get_pos = MagicMock(return_value=(400, 300))
        mock_pygame_module.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._handle_events()
        
        assert viewer.camera.zoom < initial_zoom


class TestHypersolidViewerRendering:
    """Tests for rendering functionality."""
    
    def test_render_does_not_crash(self, mock_pygame_module):
        """Test render method completes without error."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._render()
    
    def test_draw_axes_does_not_crash(self, mock_pygame_module):
        """Test axis drawing completes without error."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._draw_axes()
    
    def test_draw_ui_panel_does_not_crash(self, mock_pygame_module):
        """Test UI panel drawing completes without error."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._draw_ui_panel()
    
    def test_draw_instructions_does_not_crash(self, mock_pygame_module):
        """Test instructions drawing completes without error."""
        from visualiser import HypersolidViewer
        viewer = HypersolidViewer()
        
        with patch('visualiser.pygame', mock_pygame_module):
            viewer._draw_instructions()