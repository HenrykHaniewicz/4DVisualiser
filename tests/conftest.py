"""
Shared pytest fixtures for all test modules.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# PYGAME MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_pygame():
    """Complete pygame mock for testing without display."""
    with patch.dict('sys.modules', {'pygame': MagicMock()}):
        import pygame
        
        # Mock display
        pygame.display.set_mode = MagicMock(return_value=MagicMock())
        pygame.display.set_caption = MagicMock()
        pygame.display.flip = MagicMock()
        
        # Mock font
        mock_font = MagicMock()
        mock_font.render = MagicMock(return_value=MagicMock(
            get_width=MagicMock(return_value=100),
            get_height=MagicMock(return_value=20),
            get_rect=MagicMock(return_value=MagicMock(center=(50, 10)))
        ))
        pygame.font.SysFont = MagicMock(return_value=mock_font)
        pygame.font.Font = MagicMock(return_value=mock_font)
        
        # Mock drawing functions
        pygame.draw.rect = MagicMock()
        pygame.draw.polygon = MagicMock()
        pygame.draw.circle = MagicMock()
        pygame.draw.line = MagicMock()
        
        # Mock Rect
        pygame.Rect = MagicMock(side_effect=lambda x, y, w, h: MagicMock(
            x=x, y=y, width=w, height=h,
            centerx=x + w // 2, centery=y + h // 2,
            center=(x + w // 2, y + h // 2),
            right=x + w,
            collidepoint=MagicMock(return_value=True)
        ))
        
        # Mock Surface
        mock_surface = MagicMock()
        mock_surface.fill = MagicMock()
        mock_surface.blit = MagicMock()
        mock_surface.get_width = MagicMock(return_value=1280)
        mock_surface.get_height = MagicMock(return_value=820)
        pygame.Surface = MagicMock(return_value=mock_surface)
        
        # Mock events
        pygame.QUIT = 256
        pygame.MOUSEBUTTONDOWN = 1025
        pygame.MOUSEBUTTONUP = 1026
        pygame.MOUSEMOTION = 1024
        pygame.KEYDOWN = 768
        pygame.K_SPACE = 32
        pygame.K_r = 114
        pygame.K_ESCAPE = 27
        
        # Mock clock
        mock_clock = MagicMock()
        mock_clock.tick = MagicMock(return_value=16)
        pygame.time.Clock = MagicMock(return_value=mock_clock)
        
        # Mock init/quit
        pygame.init = MagicMock()
        pygame.quit = MagicMock()
        
        # Mock mouse
        pygame.mouse.get_pos = MagicMock(return_value=(640, 410))
        pygame.mouse.get_pressed = MagicMock(return_value=(False, False, False))
        
        # Mock event.get
        pygame.event.get = MagicMock(return_value=[])
        
        yield pygame


@pytest.fixture
def mock_pygame_initialized(mock_pygame):
    """Pygame mock with initialization called."""
    mock_pygame.init()
    yield mock_pygame


@pytest.fixture
def mock_screen(mock_pygame):
    """Mock pygame screen surface."""
    screen = MagicMock()
    screen.fill = MagicMock()
    screen.blit = MagicMock()
    screen.get_width = MagicMock(return_value=1280)
    screen.get_height = MagicMock(return_value=820)
    return screen


@pytest.fixture
def mock_font(mock_pygame):
    """Mock pygame font."""
    font = MagicMock()
    mock_surface = MagicMock()
    mock_surface.get_width = MagicMock(return_value=100)
    mock_surface.get_height = MagicMock(return_value=20)
    mock_surface.get_rect = MagicMock(return_value=MagicMock(center=(50, 10)))
    font.render = MagicMock(return_value=mock_surface)
    return font


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default configuration for testing."""
    @dataclass
    class TestConfig:
        WINDOW_WIDTH: int = 1280
        WINDOW_HEIGHT: int = 820
        FPS: int = 60
        BG_COLOR: Tuple[int, int, int] = (15, 15, 25)
        UI_BG_COLOR: Tuple[int, int, int] = (25, 25, 40)
        UI_BORDER_COLOR: Tuple[int, int, int] = (60, 60, 80)
        TEXT_COLOR: Tuple[int, int, int] = (220, 220, 230)
        HIGHLIGHT_COLOR: Tuple[int, int, int] = (100, 150, 255)
        ACCENT_COLOR: Tuple[int, int, int] = (255, 100, 150)
        VIEW_WIDTH: int = 850
        VIEW_HEIGHT: int = 710
        VIEW_X: int = 20
        VIEW_Y: int = 90
        INIT_ELEV: float = 25.0
        INIT_AZIM: float = 45.0
        INIT_ZOOM: float = 120.0
        HYPERSOLID_RESOLUTION: int = 15
    
    return TestConfig()


# =============================================================================
# MATHEMATICAL FIXTURES
# =============================================================================

@pytest.fixture
def sample_scalar_field():
    """Create a simple scalar field for testing marching cubes."""
    # Sphere: x² + y² + z² - 1 = 0
    coords = np.linspace(-2, 2, 10)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return X**2 + Y**2 + Z**2 - 1, coords


@pytest.fixture
def simple_surface_func():
    """Simple surface function for testing."""
    def func(x, y):
        return x**2 + y**2
    return func


@pytest.fixture
def simple_hypersurface_func():
    """Simple hypersurface function for testing."""
    def func(x, y, z):
        return x**2 + y**2 + z**2
    return func


@pytest.fixture
def simple_hypersolid_func():
    """Simple hypersolid function for testing."""
    def func(x, y, z, w):
        return x**2 + y**2 + z**2 + w**2 - 1
    return func


# =============================================================================
# VIEW RECT FIXTURES
# =============================================================================

@pytest.fixture
def view_rect():
    """Create a view rectangle for testing."""
    class MockRect:
        def __init__(self):
            self.x = 20
            self.y = 90
            self.width = 850
            self.height = 710
            self.centerx = 20 + 850 // 2
            self.centery = 90 + 710 // 2
        
        def collidepoint(self, point):
            x, y = point
            return (self.x <= x <= self.x + self.width and 
                    self.y <= y <= self.y + self.height)
    
    return MockRect()