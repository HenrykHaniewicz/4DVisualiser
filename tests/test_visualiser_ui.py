"""
Tests for visualiser.py - UI component classes.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
import numpy as np

# Mock pygame
mock_pygame = MagicMock()

# Proper Rect mock that returns actual values
def make_rect(x, y, w, h):
    rect = MagicMock()
    # Store as actual integers, not MagicMocks
    rect.x = int(x) if not isinstance(x, MagicMock) else 0
    rect.y = int(y) if not isinstance(y, MagicMock) else 0
    rect.width = int(w) if not isinstance(w, MagicMock) else 0
    rect.height = int(h) if not isinstance(h, MagicMock) else 0
    rect.center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
    rect.centerx = rect.center[0]
    rect.centery = rect.center[1]
    rect.right = rect.x + rect.width
    
    # Proper collidepoint that handles actual coordinates
    def collidepoint(point):
        if isinstance(point, (tuple, list)) and len(point) == 2:
            px, py = point
            # Extract actual values if they're MagicMocks
            if isinstance(px, MagicMock):
                return True  # Default to True for mocked positions
            if isinstance(py, MagicMock):
                return True
            return (rect.x <= px <= rect.x + rect.width and 
                    rect.y <= py <= rect.y + rect.height)
        return True
    
    rect.collidepoint = collidepoint
    return rect

mock_pygame.Rect = make_rect
sys.modules['pygame'] = mock_pygame


class TestButton:
    """Tests for Button class."""
    
    @pytest.fixture
    def button(self):
        """Create button instance."""
        from visualiser import Button
        return Button(100, 200, 150, 40, "Test Button")
    
    def test_initialization(self, button):
        """Test button initializes correctly."""
        assert button.text == "Test Button"
        assert button.hovered == False
        assert button.selected == False
    
    def test_set_position(self, button):
        """Test setting button position."""
        button.set_position(50, 100)
        assert button.rect.x == 50
        assert button.rect.y == 100
    
    def test_update_hover_inside(self, button):
        """Test hover detection when mouse inside."""
        # Position within button bounds (100, 200, 150, 40)
        button.update((125, 220))
        assert button.hovered == True
    
    def test_update_hover_outside(self, button):
        """Test hover detection when mouse outside."""
        button.update((0, 0))
        assert button.hovered == False
    
    def test_is_clicked_true(self, button):
        """Test click detection when clicked."""
        assert button.is_clicked((125, 220), True) == True
    
    def test_is_clicked_false_position(self, button):
        """Test click detection when not on button."""
        assert button.is_clicked((0, 0), True) == False
    
    def test_is_clicked_false_not_pressed(self, button):
        """Test click detection when not pressed."""
        assert button.is_clicked((125, 220), False) == False
    
    def test_selected_state(self, button):
        """Test selected state."""
        button.selected = True
        assert button.selected == True
    
    def test_custom_colors(self):
        """Test button with custom colors."""
        from visualiser import Button
        btn = Button(0, 0, 100, 30, "Custom",
                    color=(255, 0, 0),
                    hover_color=(0, 255, 0),
                    text_color=(0, 0, 255))
        assert btn.color == (255, 0, 0)
        assert btn.hover_color == (0, 255, 0)
        assert btn.text_color == (0, 0, 255)


class TestSlider:
    """Tests for Slider class."""
    
    @pytest.fixture
    def slider(self):
        """Create slider instance."""
        from visualiser import Slider
        return Slider(100, 200, 200, 30, 0.0, 10.0, 5.0, "Test Slider")
    
    def test_initialization(self, slider):
        """Test slider initializes correctly."""
        assert slider.min_val == 0.0
        assert slider.max_val == 10.0
        assert slider.value == 5.0
        assert slider.label == "Test Slider"
    
    def test_set_position(self, slider):
        """Test setting slider position."""
        slider.set_position(50, 100)
        assert slider.rect.x == 50
        assert slider.rect.y == 100
    
    def test_value_from_x_minimum(self, slider):
        """Test value calculation at minimum x."""
        # At left edge (x + 10)
        val = slider._value_from_x(slider.rect.x + 10)
        assert val == slider.min_val
    
    def test_value_from_x_maximum(self, slider):
        """Test value calculation at maximum x."""
        # At right edge (x + width - 10)
        val = slider._value_from_x(slider.rect.x + slider.rect.width - 10)
        assert val == slider.max_val
    
    def test_value_from_x_middle(self, slider):
        """Test value calculation at middle."""
        middle_x = slider.rect.x + slider.rect.width / 2
        val = slider._value_from_x(middle_x)
        expected = (slider.min_val + slider.max_val) / 2
        assert np.isclose(val, expected, atol=0.5)
    
    def test_get_handle_x(self, slider):
        """Test handle position calculation."""
        slider.value = slider.min_val
        x_min = slider._get_handle_x()
        
        slider.value = slider.max_val
        x_max = slider._get_handle_x()
        
        assert x_max > x_min
    
    def test_dragging_state(self, slider):
        """Test dragging state changes."""
        slider.dragging = True
        assert slider.dragging == True
        
        slider.dragging = False
        assert slider.dragging == False
    
    def test_format_string(self, slider):
        """Test custom format string."""
        slider.format_str = "{:.1f}"
        # Format string should be stored
        assert slider.format_str == "{:.1f}"
    
    def test_equal_min_max(self):
        """Test slider with equal min and max."""
        from visualiser import Slider
        slider = Slider(0, 0, 200, 30, 5.0, 5.0, 5.0)
        # Should handle gracefully
        handle_x = slider._get_handle_x()
        assert handle_x is not None


class TestUILayout:
    """Tests for UILayout class."""
    
    @pytest.fixture
    def layout(self):
        """Create layout instance."""
        from visualiser import UILayout, Config
        return UILayout(Config())
    
    def test_initialization(self, layout):
        """Test layout initializes correctly."""
        assert layout.current_y > 0
        assert layout.spacing > 0
        assert layout.section_spacing > 0
    
    def test_reset(self, layout):
        """Test reset returns to initial position."""
        initial_y = layout.current_y
        layout.add_header()
        layout.add_button()
        layout.reset()
        assert layout.current_y == initial_y
    
    def test_add_header(self, layout):
        """Test adding header advances position."""
        y_before = layout.current_y
        y_header = layout.add_header()
        assert y_header == y_before
        assert layout.current_y > y_before
    
    def test_add_button(self, layout):
        """Test adding button advances position."""
        y_before = layout.current_y
        y_button = layout.add_button()
        assert y_button == y_before
        assert layout.current_y > y_before
    
    def test_add_button_row(self, layout):
        """Test adding button row advances position."""
        y_before = layout.current_y
        y_row = layout.add_button_row(3)
        assert y_row == y_before
        assert layout.current_y > y_before
    
    def test_add_slider(self, layout):
        """Test adding slider advances position."""
        y_before = layout.current_y
        y_slider = layout.add_slider()
        # Slider y includes offset for label
        assert y_slider > y_before
        assert layout.current_y > y_slider
    
    def test_add_section_break(self, layout):
        """Test section break adds extra spacing."""
        y_before = layout.current_y
        layout.add_section_break()
        assert layout.current_y == y_before + layout.section_spacing
    
    def test_fits_in_panel_true(self, layout):
        """Test fits_in_panel when content fits."""
        layout.reset()
        assert layout.fits_in_panel() == True
    
    def test_fits_in_panel_false(self, layout):
        """Test fits_in_panel when content overflows."""
        layout.current_y = layout.panel_height + 100
        assert layout.fits_in_panel() == False