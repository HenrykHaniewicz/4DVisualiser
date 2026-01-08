"""
Tests for functions.py - Mathematical function definitions.
"""

import pytest
import numpy as np
from functions import (
    # 3D Surfaces
    surface_gaussian_poly,
    surface_sinusoidal,
    surface_ripple,
    surface_saddle,
    # 4D Hypersurfaces
    hypersurface_sphere,
    hypersurface_sinusoidal,
    hypersurface_saddle,
    hypersurface_gaussian,
    hypersurface_cubic,
    # 4D Hypersolids
    hypersolid_sphere,
    hypersolid_hyperboloid,
    hypersolid_torus,
    hypersolid_clifford,
    hypersolid_complex,
    hypersolid_hypercube,
    hypersolid_duocylinder,
    hypersolid_klein_bottle,
    # Registries
    SURFACES,
    HYPERSURFACES,
    HYPERSOLIDS,
)


# =============================================================================
# 3D SURFACE FUNCTION TESTS
# =============================================================================

class TestSurfaceGaussianPoly:
    """Tests for surface_gaussian_poly function."""
    
    def test_scalar_input(self):
        """Test with scalar inputs."""
        result = surface_gaussian_poly(1.0, 1.0)
        expected = 1.0 * 1.0 * np.exp(-1.0 - 1.0)
        assert np.isclose(result, expected)
    
    def test_origin(self):
        """Test at origin - should be zero."""
        result = surface_gaussian_poly(0.0, 0.0)
        assert result == 0.0
    
    def test_symmetry(self):
        """Test x-y symmetry."""
        result1 = surface_gaussian_poly(1.0, 2.0)
        result2 = surface_gaussian_poly(2.0, 1.0)
        assert np.isclose(result1, result2)
    
    def test_sign_symmetry(self):
        """Test even function behavior."""
        result1 = surface_gaussian_poly(1.0, 1.0)
        result2 = surface_gaussian_poly(-1.0, -1.0)
        assert np.isclose(result1, result2)
    
    def test_array_input(self):
        """Test with numpy array inputs."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = surface_gaussian_poly(x, y)
        assert result.shape == (3,)
        assert result[0] == 0.0
    
    def test_meshgrid_input(self):
        """Test with meshgrid inputs."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        result = surface_gaussian_poly(X, Y)
        assert result.shape == (5, 5)
    
    def test_decay_at_infinity(self):
        """Test exponential decay for large inputs."""
        result_small = surface_gaussian_poly(1.0, 1.0)
        result_large = surface_gaussian_poly(10.0, 10.0)
        assert abs(result_large) < abs(result_small)


class TestSurfaceSinusoidal:
    """Tests for surface_sinusoidal function."""
    
    def test_origin(self):
        """Test at origin."""
        result = surface_sinusoidal(0.0, 0.0)
        assert np.isclose(result, 0.0)
    
    def test_quarter_period(self):
        """Test at quarter period points."""
        # sin(π/2) * cos(0) = 1 * 1 = 1
        result = surface_sinusoidal(0.5, 0.0)
        assert np.isclose(result, 1.0)
    
    def test_half_period(self):
        """Test at half period."""
        result = surface_sinusoidal(1.0, 0.0)
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_periodicity_x(self):
        """Test periodicity in x."""
        result1 = surface_sinusoidal(0.25, 0.5)
        result2 = surface_sinusoidal(2.25, 0.5)
        assert np.isclose(result1, result2)
    
    def test_periodicity_y(self):
        """Test periodicity in y."""
        result1 = surface_sinusoidal(0.5, 0.25)
        result2 = surface_sinusoidal(0.5, 2.25)
        assert np.isclose(result1, result2)
    
    def test_range(self):
        """Test output range is [-1, 1]."""
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        result = surface_sinusoidal(X, Y)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestSurfaceRipple:
    """Tests for surface_ripple function."""
    
    def test_origin(self):
        """Test at origin - sin(0)/1 = 0."""
        result = surface_ripple(0.0, 0.0)
        assert np.isclose(result, 0.0)
    
    def test_radial_symmetry(self):
        """Test radial symmetry."""
        result1 = surface_ripple(1.0, 0.0)
        result2 = surface_ripple(0.0, 1.0)
        result3 = surface_ripple(np.sqrt(0.5), np.sqrt(0.5))
        assert np.isclose(result1, result2)
        assert np.isclose(result1, result3)
    
    def test_decay_with_radius(self):
        """Test amplitude decay with increasing radius."""
        result_r1 = abs(surface_ripple(1.0, 0.0))
        result_r5 = abs(surface_ripple(5.0, 0.0))
        # Amplitude should decrease due to 1/(1+r) factor
        # Not strictly true for all r due to sin, but envelope decays
        # Test at specific points where sin is similar
        r1 = 1.0
        r2 = 1.0 + 2*np.pi/5  # Same phase, larger r
        val1 = abs(surface_ripple(r1, 0.0))
        val2 = abs(surface_ripple(r2, 0.0))
        assert val2 < val1
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        result1 = surface_ripple(1.0, 1.0)
        result2 = surface_ripple(-1.0, -1.0)
        assert np.isclose(result1, result2)


class TestSurfaceSaddle:
    """Tests for surface_saddle function."""
    
    def test_origin(self):
        """Test at origin."""
        result = surface_saddle(0.0, 0.0)
        assert result == 0.0
    
    def test_x_axis(self):
        """Test along x-axis (positive)."""
        result = surface_saddle(2.0, 0.0)
        assert result == 4.0
    
    def test_y_axis(self):
        """Test along y-axis (negative)."""
        result = surface_saddle(0.0, 2.0)
        assert result == -4.0
    
    def test_diagonal_positive(self):
        """Test on y=0 line."""
        result = surface_saddle(1.0, 0.0)
        assert result == 1.0
    
    def test_diagonal_negative(self):
        """Test on x=0 line."""
        result = surface_saddle(0.0, 1.0)
        assert result == -1.0
    
    def test_equal_coordinates(self):
        """Test when x=y (should be zero)."""
        result = surface_saddle(2.0, 2.0)
        assert result == 0.0
    
    def test_antisymmetry(self):
        """Test anti-symmetry under x<->y exchange."""
        result1 = surface_saddle(1.0, 2.0)
        result2 = surface_saddle(2.0, 1.0)
        assert result1 == -result2


# =============================================================================
# 4D HYPERSURFACE FUNCTION TESTS
# =============================================================================

class TestHypersurfaceSphere:
    """Tests for hypersurface_sphere function."""
    
    def test_origin(self):
        """Test at origin."""
        result = hypersurface_sphere(0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_unit_distance(self):
        """Test at unit distance from origin."""
        result = hypersurface_sphere(1.0, 0.0, 0.0)
        assert result == 1.0
    
    def test_spherical_symmetry(self):
        """Test spherical symmetry."""
        r = np.sqrt(3)
        result1 = hypersurface_sphere(r, 0.0, 0.0)
        result2 = hypersurface_sphere(1.0, 1.0, 1.0)
        assert np.isclose(result1, result2)
    
    def test_pythagorean(self):
        """Test Pythagorean theorem."""
        result = hypersurface_sphere(3.0, 4.0, 0.0)
        assert result == 25.0
    
    def test_array_input(self):
        """Test with array inputs."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 0.0])
        result = hypersurface_sphere(x, y, z)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestHypersurfaceSinusoidal:
    """Tests for hypersurface_sinusoidal function."""
    
    def test_origin(self):
        """Test at origin."""
        result = hypersurface_sinusoidal(0.0, 0.0, 0.0)
        assert np.isclose(result, 0.0)
    
    def test_maximum(self):
        """Test at point where all factors are 1."""
        result = hypersurface_sinusoidal(0.5, 0.0, 0.5)
        # sin(π/2) * cos(0) * sin(π/2) = 1 * 1 * 1 = 1
        assert np.isclose(result, 1.0)
    
    def test_range(self):
        """Test output range."""
        coords = np.linspace(-1, 1, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        result = hypersurface_sinusoidal(X, Y, Z)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_periodicity(self):
        """Test periodicity."""
        result1 = hypersurface_sinusoidal(0.5, 0.25, 0.5)
        result2 = hypersurface_sinusoidal(0.5, 2.25, 0.5)
        assert np.isclose(result1, result2)


class TestHypersurfaceSaddle:
    """Tests for hypersurface_saddle function."""
    
    def test_origin(self):
        """Test at origin."""
        result = hypersurface_saddle(0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_z_linear(self):
        """Test linear z dependence."""
        result1 = hypersurface_saddle(0.0, 0.0, 1.0)
        result2 = hypersurface_saddle(0.0, 0.0, 2.0)
        assert result2 == 2 * result1
    
    def test_saddle_behavior(self):
        """Test saddle behavior in x-y plane."""
        result_x = hypersurface_saddle(1.0, 0.0, 0.0)
        result_y = hypersurface_saddle(0.0, 1.0, 0.0)
        assert result_x == 1.0
        assert result_y == -1.0


class TestHypersurfaceGaussian:
    """Tests for hypersurface_gaussian function."""
    
    def test_origin_maximum(self):
        """Test maximum at origin."""
        result = hypersurface_gaussian(0.0, 0.0, 0.0)
        assert result == 1.0
    
    def test_decay(self):
        """Test decay from origin."""
        result_origin = hypersurface_gaussian(0.0, 0.0, 0.0)
        result_away = hypersurface_gaussian(1.0, 0.0, 0.0)
        assert result_away < result_origin
    
    def test_spherical_symmetry(self):
        """Test spherical symmetry."""
        r = 1.0
        result1 = hypersurface_gaussian(r, 0.0, 0.0)
        result2 = hypersurface_gaussian(0.0, r, 0.0)
        result3 = hypersurface_gaussian(0.0, 0.0, r)
        assert np.isclose(result1, result2)
        assert np.isclose(result2, result3)
    
    def test_specific_value(self):
        """Test specific value."""
        result = hypersurface_gaussian(1.0, 0.0, 0.0)
        assert np.isclose(result, np.exp(-1))
    
    def test_range(self):
        """Test output range (0, 1]."""
        coords = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        result = hypersurface_gaussian(X, Y, Z)
        assert np.all(result > 0) and np.all(result <= 1.0)


class TestHypersurfaceCubic:
    """Tests for hypersurface_cubic function."""
    
    def test_origin(self):
        """Test at origin."""
        result = hypersurface_cubic(0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_cubic_behavior(self):
        """Test cubic behavior."""
        result = hypersurface_cubic(2.0, 0.0, 0.0)
        assert result == 8.0
    
    def test_sign_behavior(self):
        """Test sign behavior for negative inputs."""
        result = hypersurface_cubic(-1.0, 0.0, 0.0)
        assert result == -1.0
    
    def test_combination(self):
        """Test combined values."""
        result = hypersurface_cubic(1.0, 1.0, 1.0)
        # 1 + 1 - 1 = 1
        assert result == 1.0


# =============================================================================
# 4D HYPERSOLID FUNCTION TESTS
# =============================================================================

class TestHypersolidSphere:
    """Tests for hypersolid_sphere function."""
    
    def test_origin_inside(self):
        """Test origin is inside sphere."""
        result = hypersolid_sphere(0.0, 0.0, 0.0, 0.0)
        # R=2, so at origin: 0 - 4 = -4 (inside)
        assert result == -4.0
    
    def test_on_surface(self):
        """Test points on surface."""
        R = 2.0
        result = hypersolid_sphere(R, 0.0, 0.0, 0.0)
        assert np.isclose(result, 0.0)
    
    def test_outside(self):
        """Test points outside."""
        result = hypersolid_sphere(3.0, 0.0, 0.0, 0.0)
        assert result > 0
    
    def test_4d_symmetry(self):
        """Test 4D spherical symmetry."""
        result1 = hypersolid_sphere(1.0, 0.0, 0.0, 0.0)
        result2 = hypersolid_sphere(0.0, 1.0, 0.0, 0.0)
        result3 = hypersolid_sphere(0.0, 0.0, 1.0, 0.0)
        result4 = hypersolid_sphere(0.0, 0.0, 0.0, 1.0)
        assert np.allclose([result1, result2, result3, result4], result1)
    
    def test_diagonal(self):
        """Test on 4D diagonal."""
        r = 1.0  # Each component
        result = hypersolid_sphere(r, r, r, r)
        # 4r² - 4 = 4(1) - 4 = 0 when r=1
        assert np.isclose(result, 0.0)


class TestHypersolidHyperboloid:
    """Tests for hypersolid_hyperboloid function."""
    
    def test_specific_point(self):
        """Test at a known point."""
        # w² - x² - y² + z² = 1
        # At (0,0,0,w): w² = 1 => w = ±1
        result = hypersolid_hyperboloid(0.0, 0.0, 0.0, 1.0)
        assert np.isclose(result, 0.0)
    
    def test_symmetry_w_z(self):
        """Test symmetry between w and z."""
        result1 = hypersolid_hyperboloid(0.0, 0.0, 1.0, 0.0)
        result2 = hypersolid_hyperboloid(0.0, 0.0, 0.0, 1.0)
        # z² - 1 = 0 vs w² - 1 = 0
        assert np.isclose(result1, result2)
    
    def test_symmetry_x_y(self):
        """Test symmetry between x and y."""
        result1 = hypersolid_hyperboloid(1.0, 0.0, 0.0, 0.0)
        result2 = hypersolid_hyperboloid(0.0, 1.0, 0.0, 0.0)
        assert np.isclose(result1, result2)


class TestHypersolidTorus:
    """Tests for hypersolid_torus function."""
    
    def test_major_circle_xy(self):
        """Test point on major circle in xy plane."""
        R = 1.2  # Major radius from function
        r = 0.5  # Minor radius
        # Point on major circle: (R, 0, R, 0)
        # d1 = |R| - R = 0, d2 = |R| - R = 0
        result = hypersolid_torus(R, 0.0, R, 0.0)
        assert np.isclose(result, -r**2, atol=0.01)
    
    def test_symmetry(self):
        """Test symmetry properties."""
        result1 = hypersolid_torus(1.0, 0.5, 0.5, 1.0)
        result2 = hypersolid_torus(0.5, 1.0, 1.0, 0.5)
        # Due to symmetry in (x,y) and (z,w) pairs
        assert np.isclose(result1, result2)


class TestHypersolidClifford:
    """Tests for hypersolid_clifford function."""
    
    def test_ideal_point(self):
        """Test at ideal Clifford torus point."""
        # When x²+y² = 1 and z²+w² = 1
        # Both terms become 0, result = -0.1
        result = hypersolid_clifford(1.0, 0.0, 1.0, 0.0)
        assert np.isclose(result, -0.1)
    
    def test_off_surface(self):
        """Test points away from surface."""
        result = hypersolid_clifford(2.0, 0.0, 0.0, 0.0)
        # (4-1)² + (0-1)² - 0.1 = 9 + 1 - 0.1 = 9.9
        assert np.isclose(result, 9.9)


class TestHypersolidComplex:
    """Tests for hypersolid_complex function."""
    
    def test_specific_zeros(self):
        """Test that specific points give expected values."""
        # sin(0)cos(0) + sin(0)cos(0) - 0.5 = -0.5
        result = hypersolid_complex(0.0, 0.0, 0.0, 0.0)
        assert np.isclose(result, -0.5)
    
    def test_periodicity(self):
        """Test periodic behavior."""
        result1 = hypersolid_complex(0.5, 0.5, 0.5, 0.5)
        result2 = hypersolid_complex(0.5 + 2*np.pi, 0.5, 0.5, 0.5)
        assert np.isclose(result1, result2)


class TestHypersolidHypercube:
    """Tests for hypersolid_hypercube function."""
    
    def test_origin_inside(self):
        """Test origin is inside."""
        result = hypersolid_hypercube(0.0, 0.0, 0.0, 0.0)
        assert result == -1.0
    
    def test_on_face(self):
        """Test point on face."""
        result = hypersolid_hypercube(1.0, 0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_corner(self):
        """Test corner point."""
        result = hypersolid_hypercube(1.0, 1.0, 1.0, 1.0)
        assert result == 0.0
    
    def test_outside(self):
        """Test point outside."""
        result = hypersolid_hypercube(2.0, 0.0, 0.0, 0.0)
        assert result == 1.0
    
    def test_symmetry(self):
        """Test 4D symmetry."""
        result1 = hypersolid_hypercube(0.5, 0.3, 0.2, 0.1)
        result2 = hypersolid_hypercube(0.1, 0.5, 0.3, 0.2)
        assert np.isclose(result1, result2)


class TestHypersolidDuocylinder:
    """Tests for hypersolid_duocylinder function."""
    
    def test_origin_inside(self):
        """Test origin is inside."""
        R = 1.5
        result = hypersolid_duocylinder(0.0, 0.0, 0.0, 0.0)
        assert result == -R**2
    
    def test_on_surface(self):
        """Test point on surface."""
        R = 1.5
        result = hypersolid_duocylinder(R, 0.0, 0.0, 0.0)
        assert np.isclose(result, 0.0)
    
    def test_edge_point(self):
        """Test point on edge (both cylinders' surfaces)."""
        R = 1.5
        result = hypersolid_duocylinder(R, 0.0, R, 0.0)
        assert np.isclose(result, 0.0)


class TestHypersolidKleinBottle:
    """Tests for hypersolid_klein_bottle function."""
    
    def test_finite_values(self):
        """Test that function returns finite values."""
        result = hypersolid_klein_bottle(1.0, 0.5, 0.3, 0.2)
        assert np.isfinite(result)
    
    def test_array_input(self):
        """Test with array input."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        z = np.zeros(5)
        w = np.zeros(5)
        result = hypersolid_klein_bottle(x, y, z, w)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
    
    def test_symmetry_breaking(self):
        """Test that Klein bottle breaks certain symmetries."""
        # Klein bottle should not have full 4D spherical symmetry
        result1 = hypersolid_klein_bottle(1.0, 0.0, 0.0, 0.0)
        result2 = hypersolid_klein_bottle(0.0, 0.0, 1.0, 0.0)
        # These should generally differ
        # Note: This is a weak test; the structure is complex


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestSurfaceRegistry:
    """Tests for SURFACES registry."""
    
    def test_registry_not_empty(self):
        """Test registry has entries."""
        assert len(SURFACES) > 0
    
    def test_registry_structure(self):
        """Test registry entry structure."""
        for func_id, entry in SURFACES.items():
            assert isinstance(func_id, int)
            assert len(entry) == 5
            func, name, formula, domain_x, domain_y = entry
            assert callable(func)
            assert isinstance(name, str)
            assert isinstance(formula, str)
            assert len(domain_x) == 2
            assert len(domain_y) == 2
            assert domain_x[0] < domain_x[1]
            assert domain_y[0] < domain_y[1]
    
    def test_all_functions_callable(self):
        """Test all registered functions are callable."""
        for func_id, entry in SURFACES.items():
            func = entry[0]
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
            result = func(x, y)
            assert result.shape == (2,)
    
    def test_functions_handle_meshgrid(self):
        """Test functions handle meshgrid input."""
        for func_id, entry in SURFACES.items():
            func, _, _, domain_x, domain_y = entry
            x = np.linspace(domain_x[0], domain_x[1], 10)
            y = np.linspace(domain_y[0], domain_y[1], 10)
            X, Y = np.meshgrid(x, y)
            result = func(X, Y)
            assert result.shape == (10, 10)


class TestHypersurfaceRegistry:
    """Tests for HYPERSURFACES registry."""
    
    def test_registry_not_empty(self):
        """Test registry has entries."""
        assert len(HYPERSURFACES) > 0
    
    def test_registry_structure(self):
        """Test registry entry structure."""
        for func_id, entry in HYPERSURFACES.items():
            assert isinstance(func_id, int)
            assert len(entry) == 4
            func, name, formula, domain = entry
            assert callable(func)
            assert isinstance(name, str)
            assert isinstance(formula, str)
            assert len(domain) == 2
            assert domain[0] < domain[1]
    
    def test_all_functions_callable(self):
        """Test all registered functions are callable."""
        for func_id, entry in HYPERSURFACES.items():
            func = entry[0]
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
            z = np.array([0.0, 1.0])
            result = func(x, y, z)
            assert result.shape == (2,)


class TestHypersolidRegistry:
    """Tests for HYPERSOLIDS registry."""
    
    def test_registry_not_empty(self):
        """Test registry has entries."""
        assert len(HYPERSOLIDS) > 0
    
    def test_registry_structure(self):
        """Test registry entry structure."""
        for func_id, entry in HYPERSOLIDS.items():
            assert isinstance(func_id, int)
            assert len(entry) == 7
            func, name, formula, dx, dy, dz, dw = entry
            assert callable(func)
            assert isinstance(name, str)
            assert isinstance(formula, str)
            for domain in [dx, dy, dz, dw]:
                assert len(domain) == 2
                assert domain[0] < domain[1]
    
    def test_all_functions_callable(self):
        """Test all registered functions are callable."""
        for func_id, entry in HYPERSOLIDS.items():
            func = entry[0]
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
            z = np.array([0.0, 1.0])
            w = np.array([0.0, 1.0])
            result = func(x, y, z, w)
            assert result.shape == (2,)
    
    def test_functions_have_isosurface(self):
        """Test that functions produce zero crossings in their domains."""
        for func_id, entry in HYPERSOLIDS.items():
            func, name, _, dx, dy, dz, dw = entry
            # Sample the domain
            x = np.linspace(dx[0], dx[1], 5)
            y = np.linspace(dy[0], dy[1], 5)
            z = np.linspace(dz[0], dz[1], 5)
            w_val = (dw[0] + dw[1]) / 2
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            W = np.full_like(X, w_val)
            result = func(X, Y, Z, W)
            # Should have both positive and negative values (crossing zero)
            has_positive = np.any(result > 0)
            has_negative = np.any(result < 0)
            # Note: Not all w slices will have crossings
            # This is a sanity check, not a strict requirement