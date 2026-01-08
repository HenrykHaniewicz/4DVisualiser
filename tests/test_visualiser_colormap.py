"""
Tests for visualiser.py - Colormap classes.
"""

import pytest
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock pygame before importing visualiser
sys.modules['pygame'] = MagicMock()


class TestColormap:
    """Tests for Colormap class."""
    
    @pytest.fixture
    def colormap_class(self):
        """Get Colormap class."""
        from visualiser import Colormap
        return Colormap
    
    def test_viridis_range(self, colormap_class):
        """Test viridis returns valid RGB values."""
        for t in np.linspace(0, 1, 11):
            color = colormap_class.viridis(t)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
            assert all(isinstance(c, (int, np.integer)) for c in color)
    
    def test_plasma_range(self, colormap_class):
        """Test plasma returns valid RGB values."""
        for t in np.linspace(0, 1, 11):
            color = colormap_class.plasma(t)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_inferno_range(self, colormap_class):
        """Test inferno returns valid RGB values."""
        for t in np.linspace(0, 1, 11):
            color = colormap_class.inferno(t)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_cool_warm_range(self, colormap_class):
        """Test cool_warm returns valid RGB values."""
        for t in np.linspace(0, 1, 11):
            color = colormap_class.cool_warm(t)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_viridis_clipping_below(self, colormap_class):
        """Test viridis clips values below 0."""
        color_neg = colormap_class.viridis(-0.5)
        color_zero = colormap_class.viridis(0.0)
        assert color_neg == color_zero
    
    def test_viridis_clipping_above(self, colormap_class):
        """Test viridis clips values above 1."""
        color_above = colormap_class.viridis(1.5)
        color_one = colormap_class.viridis(1.0)
        assert color_above == color_one
    
    def test_cool_warm_transition(self, colormap_class):
        """Test cool_warm has cool and warm ends."""
        cold = colormap_class.cool_warm(0.0)
        warm = colormap_class.cool_warm(1.0)
        
        # Cold end should have more blue
        assert cold[2] > cold[0]
        # Warm end should have more red
        assert warm[0] > warm[2]
    
    def test_endpoints_distinct(self, colormap_class):
        """Test colormap endpoints are visually distinct."""
        for colormap in [colormap_class.viridis, colormap_class.plasma, 
                         colormap_class.inferno, colormap_class.cool_warm]:
            start = colormap(0.0)
            end = colormap(1.0)
            # Colors should differ significantly
            diff = sum(abs(s - e) for s, e in zip(start, end))
            assert diff > 100  # Significant color difference


class TestColormapRegistry:
    """Tests for COLORMAPS dictionary."""
    
    @pytest.fixture
    def colormaps(self):
        """Get COLORMAPS dictionary."""
        from visualiser import COLORMAPS
        return COLORMAPS
    
    def test_registry_contains_expected(self, colormaps):
        """Test registry contains expected colormaps."""
        expected = ['viridis', 'plasma', 'inferno', 'cool_warm']
        for name in expected:
            assert name in colormaps
    
    def test_all_callable(self, colormaps):
        """Test all registered colormaps are callable."""
        for name, cmap in colormaps.items():
            assert callable(cmap)
            result = cmap(0.5)
            assert len(result) == 3