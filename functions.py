"""
Example functions for 4D visualization.
Contains 3D surfaces, 4D hypersurfaces, and 4D solid definitions.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any


# =============================================================================
# 3D SURFACE FUNCTIONS (z = f(x, y))
# =============================================================================

def surface_gaussian_poly(x, y):
    """Gaussian-modulated polynomial"""
    return x**2 * y**2 * np.exp(-x**2 - y**2)


def surface_sinusoidal(x, y):
    """Sinusoidal surface"""
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def surface_ripple(x, y):
    """Ripple pattern"""
    r = np.sqrt(x**2 + y**2)
    return np.sin(5 * r) / (1 + r)


def surface_saddle(x, y):
    """Saddle surface"""
    return x**2 - y**2


# =============================================================================
# 4D HYPERSURFACE FUNCTIONS (w = f(x, y, z))
# =============================================================================

def hypersurface_sphere(x, y, z):
    """4D hypersphere (squared radius)"""
    return x**2 + y**2 + z**2


def hypersurface_sinusoidal(x, y, z):
    """3D sinusoidal product"""
    return np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)


def hypersurface_saddle(x, y, z):
    """Saddle with linear term"""
    return x**2 - y**2 + z


def hypersurface_gaussian(x, y, z):
    """3D Gaussian"""
    return np.exp(-(x**2 + y**2 + z**2))


def hypersurface_cubic(x, y, z):
    """Cubic combination"""
    return x**3 + y**3 - z**3


# =============================================================================
# 4D SOLID FUNCTIONS (Implicit: f(x, y, z, w) = 0)
# =============================================================================

def hypersolid_sphere(x, y, z, w):
    """4D sphere: x² + y² + z² + w² = R²
    Returns signed distance (negative inside, positive outside)
    """
    R = 2.0
    return x**2 + y**2 + z**2 + w**2 - R**2


def hypersolid_hyperboloid(x, y, z, w):
    """4D hyperboloid: w² - x² - y² + z² = 1"""
    return w**2 - x**2 - y**2 + z**2 - 1


def hypersolid_torus(x, y, z, w):
    """4D torus (Clifford torus variant)
    (sqrt(x² + y²) - R)² + (sqrt(z² + w²) - R)² = r²
    """
    R = 1.2  # Major radius
    r = 0.5  # Minor radius
    d1 = np.sqrt(x**2 + y**2) - R
    d2 = np.sqrt(z**2 + w**2) - R
    return d1**2 + d2**2 - r**2


def hypersolid_clifford(x, y, z, w):
    """Clifford torus: x² + y² = z² + w² = 1 (on 3-sphere)
    Parameterized on the 3-sphere x² + y² + z² + w² = 2
    """
    return (x**2 + y**2 - 1)**2 + (z**2 + w**2 - 1)**2 - 0.1


def hypersolid_complex(x, y, z, w):
    """Complex 4D shape"""
    return np.sin(x) * np.cos(y) + np.sin(z) * np.cos(w) - 0.5


def hypersolid_hypercube(x, y, z, w):
    """4D hypercube (tesseract) - signed distance approximation"""
    return np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z), np.abs(w)]) - 1.0


def hypersolid_duocylinder(x, y, z, w):
    """Duocylinder: intersection of two solid cylinders in 4D
    x² + y² <= R² and z² + w² <= R²
    """
    R = 1.5
    d1 = x**2 + y**2 - R**2
    d2 = z**2 + w**2 - R**2
    return np.maximum(d1, d2)


def hypersolid_klein_bottle(x, y, z, w):
    """Klein bottle embedded in 4D
    
    This uses the "figure-8" parameterisation lifted to 4D:
    The implicit form is derived from the parametric equations:
        x = (a + b*cos(v)) * cos(u)
        y = (a + b*cos(v)) * sin(u)
        z = b * sin(v) * cos(u/2)
        w = b * sin(v) * sin(u/2)
    
    We use an implicit approximation based on the distance to this surface.
    """
    a = 1.0  # Major radius
    b = 0.4  # Tube radius
    
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    half_theta = theta / 2
    
    z_rot = z * np.cos(half_theta) + w * np.sin(half_theta)
    w_rot = -z * np.sin(half_theta) + w * np.cos(half_theta)
    
    dist_to_circle = np.sqrt((r_xy - a)**2 + z_rot**2) - b

    return dist_to_circle**2 + w_rot**2 - (b * 0.3)**2

def hypersolid_klein_bottlev2(x, y, z, w):
    """Klein bottle embedded in 4D - alternate
    """
    a = 2.0  # Scale of the bottle
    b = 1.0  # Width of the tube
    
    u = np.arctan2(z, x)
    
    rxz = np.sqrt(x**2 + z**2)
    
    R = a * (1.0 + 0.5 * np.cos(u))
    
    twist_angle = u / 2.0
    
    dx = rxz - R
    
    cos_t = np.cos(twist_angle)
    sin_t = np.sin(twist_angle)
    y_twisted = y * cos_t - w * sin_t
    w_twisted = y * sin_t + w * cos_t
    
    cross_section = dx**2 + y_twisted**2 - b**2
    
    return cross_section + w_twisted**2 * 0.5


# =============================================================================
# FUNCTION REGISTRIES
# =============================================================================

# Format: {id: (function, name, formula, domain_x, domain_y)}
SURFACES: Dict[int, Tuple[Callable, str, str, Tuple[float, float], Tuple[float, float]]] = {
    1: (surface_gaussian_poly, "Gaussian Polynomial", "z = x²y²e^(-x²-y²)", (-2, 2), (-2, 2)),
    2: (surface_sinusoidal, "Sinusoidal", "z = sin(πx)cos(πy)", (-2, 2), (-2, 2)),
    3: (surface_ripple, "Ripple", "z = sin(5r)/(1+r)", (-3, 3), (-3, 3)),
    4: (surface_saddle, "Saddle", "z = x² - y²", (-2, 2), (-2, 2)),
}

# Format: {id: (function, name, formula, domain)}
HYPERSURFACES: Dict[int, Tuple[Callable, str, str, Tuple[float, float]]] = {
    1: (hypersurface_sphere, "4D Sphere", "w = x² + y² + z²", (-2, 2)),
    2: (hypersurface_sinusoidal, "Sinusoidal 4D", "w = sin(πx)cos(πy)sin(πz)", (-1, 1)),
    3: (hypersurface_saddle, "4D Saddle", "w = x² - y² + z", (-2, 2)),
    4: (hypersurface_gaussian, "4D Gaussian", "w = e^(-(x²+y²+z²))", (-2, 2)),
    5: (hypersurface_cubic, "Cubic 4D", "w = x³ + y³ - z³", (-1.5, 1.5)),
}

# Format: {id: (function, name, formula, domain_x, domain_y, domain_z, domain_w)}
HYPERSOLIDS: Dict[int, Tuple[Callable, str, str, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    1: (hypersolid_sphere, "4D Hypersphere", "x²+y²+z²+w²=4", (-2.2, 2.2), (-2.2, 2.2), (-2.2, 2.2), (-2.2, 2.2)),
    2: (hypersolid_hyperboloid, "4D Hyperboloid", "w²-x²-y²+z²=1", (-2, 2), (-2, 2), (-2, 2), (-2, 2)),
    3: (hypersolid_torus, "4D Torus", "(√(x²+y²)-R)²+(√(z²+w²)-R)²=r²", (-2, 2), (-2, 2), (-2, 2), (-2, 2)),
    4: (hypersolid_clifford, "Clifford Torus", "(x²+y²-1)²+(z²+w²-1)²=0.1", (-1.8, 1.8), (-1.8, 1.8), (-1.8, 1.8), (-1.8, 1.8)),
    5: (hypersolid_complex, "Complex 4D", "sin(x)cos(y)+sin(z)cos(w)=0.5", (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    6: (hypersolid_hypercube, "4D Hypercube", "max(|x|,|y|,|z|,|w|)=1", (-1.8, 1.8), (-1.8, 1.8), (-1.8, 1.8), (-1.8, 1.8)),
    7: (hypersolid_duocylinder, "Duocylinder", "max(x²+y²,z²+w²)=R²", (-2, 2), (-2, 2), (-2, 2), (-2, 2)),
    8: (hypersolid_klein_bottlev2, "Klein Bottle", "4D figure-8 Klein bottle", (-4, 4), (-2.5, 2.5), (-4, 4), (-2.5, 2.5)),
}