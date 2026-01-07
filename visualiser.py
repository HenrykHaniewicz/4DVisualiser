"""
Interactive 4D Visualization Viewer using Pygame
A visual application for exploring 3D surfaces, 4D hypersurfaces, and 4D solids.
Features solid mesh rendering with proper depth sorting and shading.
"""

import numpy as np
import pygame
import sys
import math
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional, Dict
from enum import Enum, auto

from marching_cubes import marching_cubes
from functions import SURFACES, HYPERSURFACES, HYPERSOLIDS


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Application configuration"""
    WINDOW_WIDTH: int = 1280
    WINDOW_HEIGHT: int = 820
    FPS: int = 60
    
    # Colors
    BG_COLOR: Tuple[int, int, int] = (15, 15, 25)
    UI_BG_COLOR: Tuple[int, int, int] = (25, 25, 40)
    UI_BORDER_COLOR: Tuple[int, int, int] = (60, 60, 80)
    TEXT_COLOR: Tuple[int, int, int] = (220, 220, 230)
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (100, 150, 255)
    ACCENT_COLOR: Tuple[int, int, int] = (255, 100, 150)
    
    # 3D View settings
    VIEW_WIDTH: int = 850
    VIEW_HEIGHT: int = 710
    VIEW_X: int = 20
    VIEW_Y: int = 90
    
    # Initial view angles
    INIT_ELEV: float = 25.0
    INIT_AZIM: float = 45.0
    INIT_ZOOM: float = 120.0
    
    # Hypersolid resolution (fixed for performance)
    HYPERSOLID_RESOLUTION: int = 15


class VisualizationType(Enum):
    SURFACE_3D = auto()
    HYPERSURFACE_4D = auto()
    HYPERSOLID_4D = auto()


# =============================================================================
# COLORMAPS
# =============================================================================

class Colormap:
    """Simple colormap implementation"""
    
    @staticmethod
    def viridis(t: float) -> Tuple[int, int, int]:
        t = np.clip(t, 0, 1)
        r = int(255 * (0.267 + 0.329 * t + 2.21 * t**2 - 4.04 * t**3 + 1.62 * t**4))
        g = int(255 * (0.004 + 1.42 * t - 1.60 * t**2 + 1.52 * t**3 - 0.48 * t**4))
        b = int(255 * (0.329 + 1.44 * t - 2.89 * t**2 + 2.06 * t**3 - 0.55 * t**4))
        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))
    
    @staticmethod
    def plasma(t: float) -> Tuple[int, int, int]:
        t = np.clip(t, 0, 1)
        r = int(255 * (0.05 + 0.91 * t + 0.54 * t**2 - 0.62 * t**3))
        g = int(255 * (0.02 + 0.10 * t + 1.88 * t**2 - 1.32 * t**3))
        b = int(255 * (0.53 + 0.69 * t - 2.02 * t**2 + 1.16 * t**3))
        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))
    
    @staticmethod
    def inferno(t: float) -> Tuple[int, int, int]:
        t = np.clip(t, 0, 1)
        r = int(255 * (0.0 + 1.13 * t + 1.89 * t**2 - 2.39 * t**3 + 0.54 * t**4))
        g = int(255 * (0.0 + 0.08 * t + 2.08 * t**2 - 2.14 * t**3 + 0.69 * t**4))
        b = int(255 * (0.01 + 1.68 * t - 3.00 * t**2 + 2.27 * t**3 - 0.67 * t**4))
        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))
    
    @staticmethod
    def cool_warm(t: float) -> Tuple[int, int, int]:
        t = np.clip(t, 0, 1)
        if t < 0.5:
            s = t * 2
            r = int(60 + 140 * s)
            g = int(100 + 100 * s)
            b = int(220 - 20 * s)
        else:
            s = (t - 0.5) * 2
            r = int(200 + 55 * s)
            g = int(200 - 120 * s)
            b = int(200 - 180 * s)
        return (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))


COLORMAPS = {
    'viridis': Colormap.viridis,
    'plasma': Colormap.plasma,
    'inferno': Colormap.inferno,
    'cool_warm': Colormap.cool_warm,
}


# =============================================================================
# 3D CAMERA
# =============================================================================

class Camera:
    """3D Camera for projection"""
    
    def __init__(self, config: Config):
        self.elevation = config.INIT_ELEV
        self.azimuth = config.INIT_AZIM
        self.zoom = config.INIT_ZOOM
        self.center_x = config.VIEW_X + config.VIEW_WIDTH // 2
        self.center_y = config.VIEW_Y + config.VIEW_HEIGHT // 2
        self.config = config
        
        # Light direction (normalized)
        self.light_dir = np.array([0.5, 0.3, 0.8])
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        
        # Cache rotation matrix
        self._update_rotation_matrix()
    
    def _update_rotation_matrix(self):
        """Update cached rotation matrix"""
        elev = math.radians(self.elevation)
        azim = math.radians(self.azimuth)
        
        # Rotation around Z axis (azimuth)
        Rz = np.array([
            [math.cos(azim), -math.sin(azim), 0],
            [math.sin(azim), math.cos(azim), 0],
            [0, 0, 1]
        ])
        
        # Rotation around X axis (elevation)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(elev), -math.sin(elev)],
            [0, math.sin(elev), math.cos(elev)]
        ])
        
        self._rotation_matrix = Rx @ Rz
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get the combined rotation matrix"""
        return self._rotation_matrix
    
    def rotate(self, d_azim: float, d_elev: float):
        self.azimuth += d_azim
        self.elevation = np.clip(self.elevation + d_elev, -89, 89)
        self._update_rotation_matrix()
    
    def zoom_in(self, factor: float):
        self.zoom = np.clip(self.zoom * factor, 20, 500)
    
    def reset(self):
        self.elevation = self.config.INIT_ELEV
        self.azimuth = self.config.INIT_AZIM
        self.zoom = self.config.INIT_ZOOM
        self._update_rotation_matrix()


# =============================================================================
# VISUALIZATION RENDERERS
# =============================================================================

class Surface3DRenderer:
    """Renderer for 3D surfaces using mesh triangles"""
    
    def __init__(self, func: Callable, domain_x: Tuple[float, float], 
                 domain_y: Tuple[float, float], resolution: int = 50):
        self.func = func
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.resolution = resolution
        self._compute_mesh()
    
    def _compute_mesh(self):
        """Compute surface mesh triangles"""
        x = np.linspace(self.domain_x[0], self.domain_x[1], self.resolution)
        y = np.linspace(self.domain_y[0], self.domain_y[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.z_min, self.z_max = np.min(Z), np.max(Z)
        if self.z_max == self.z_min:
            self.z_max = self.z_min + 1
        
        # Normalize coordinates
        x_range = self.domain_x[1] - self.domain_x[0]
        y_range = self.domain_y[1] - self.domain_y[0]
        z_range = self.z_max - self.z_min
        max_range = max(x_range, y_range, z_range)
        
        x_center = (self.domain_x[0] + self.domain_x[1]) / 2
        y_center = (self.domain_y[0] + self.domain_y[1]) / 2
        z_center = (self.z_min + self.z_max) / 2
        
        # Build triangle arrays
        num_quads = (self.resolution - 1) ** 2
        self.vertices = np.zeros((num_quads * 6, 3))
        self.colors_t = np.zeros(num_quads * 2)
        
        idx = 0
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                # Quad corners
                v00 = np.array([
                    (X[i, j] - x_center) / max_range * 2,
                    (Y[i, j] - y_center) / max_range * 2,
                    (Z[i, j] - z_center) / max_range * 2
                ])
                v10 = np.array([
                    (X[i+1, j] - x_center) / max_range * 2,
                    (Y[i+1, j] - y_center) / max_range * 2,
                    (Z[i+1, j] - z_center) / max_range * 2
                ])
                v01 = np.array([
                    (X[i, j+1] - x_center) / max_range * 2,
                    (Y[i, j+1] - y_center) / max_range * 2,
                    (Z[i, j+1] - z_center) / max_range * 2
                ])
                v11 = np.array([
                    (X[i+1, j+1] - x_center) / max_range * 2,
                    (Y[i+1, j+1] - y_center) / max_range * 2,
                    (Z[i+1, j+1] - z_center) / max_range * 2
                ])
                
                # Color values
                t00 = (Z[i, j] - self.z_min) / (self.z_max - self.z_min)
                t10 = (Z[i+1, j] - self.z_min) / (self.z_max - self.z_min)
                t01 = (Z[i, j+1] - self.z_min) / (self.z_max - self.z_min)
                t11 = (Z[i+1, j+1] - self.z_min) / (self.z_max - self.z_min)
                
                # Triangle 1
                self.vertices[idx*3] = v00
                self.vertices[idx*3 + 1] = v10
                self.vertices[idx*3 + 2] = v11
                self.colors_t[idx] = (t00 + t10 + t11) / 3
                idx += 1
                
                # Triangle 2
                self.vertices[idx*3] = v00
                self.vertices[idx*3 + 1] = v11
                self.vertices[idx*3 + 2] = v01
                self.colors_t[idx] = (t00 + t11 + t01) / 3
                idx += 1
    
    def render(self, screen: pygame.Surface, camera: Camera, colormap: Callable,
               view_rect: pygame.Rect):
        """Render the surface with proper shading"""
        R = camera.get_rotation_matrix()
        
        # Transform all vertices at once
        rotated = self.vertices @ R.T
        
        # Process triangles
        num_tris = len(self.colors_t)
        tri_data = []
        
        for i in range(num_tris):
            r0 = rotated[i*3]
            r1 = rotated[i*3 + 1]
            r2 = rotated[i*3 + 2]
            
            # Centroid depth
            depth = (r0[1] + r1[1] + r2[1]) / 3
            
            # Normal for lighting
            edge1 = r1 - r0
            edge2 = r2 - r0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
            
            # Screen coordinates
            s0 = (int(camera.center_x + r0[0] * camera.zoom),
                  int(camera.center_y - r0[2] * camera.zoom))
            s1 = (int(camera.center_x + r1[0] * camera.zoom),
                  int(camera.center_y - r1[2] * camera.zoom))
            s2 = (int(camera.center_x + r2[0] * camera.zoom),
                  int(camera.center_y - r2[2] * camera.zoom))
            
            tri_data.append((s0, s1, s2, depth, normal, self.colors_t[i]))
        
        # Sort by depth
        tri_data.sort(key=lambda x: x[3], reverse=True)
        
        # Render
        for s0, s1, s2, depth, normal, t in tri_data:
            if not (view_rect.collidepoint(s0) or view_rect.collidepoint(s1) or view_rect.collidepoint(s2)):
                continue
            
            base_color = colormap(t)
            light_intensity = max(0.3, min(1.0, 0.4 + 0.6 * abs(np.dot(normal, camera.light_dir))))
            color = tuple(int(c * light_intensity) for c in base_color)
            
            try:
                pygame.draw.polygon(screen, color, [s0, s1, s2])
            except:
                pass


class HypersurfaceRenderer:
    """Renderer for 4D hypersurfaces with slicing"""
    
    def __init__(self, func: Callable, domain: Tuple[float, float], resolution: int = 40):
        self.func = func
        self.domain = domain
        self.resolution = resolution
        self.slice_dim = 'z'
        self.slice_value = 0.0
        self.slice_min = domain[0]
        self.slice_max = domain[1]
        self._precompute()
        self._update_slice()
    
    def _precompute(self):
        """Precompute function range"""
        coords = np.linspace(self.domain[0], self.domain[1], 20)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        W = self.func(X, Y, Z)
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        self.w_min, self.w_max = np.min(W), np.max(W)
        if self.w_max == self.w_min:
            self.w_max = self.w_min + 1
    
    def set_slice(self, dim: str, value: float):
        self.slice_dim = dim
        self.slice_value = np.clip(value, self.slice_min, self.slice_max)
        self._update_slice()
    
    def _update_slice(self):
        """Update mesh based on current slice"""
        coords = np.linspace(self.domain[0], self.domain[1], self.resolution)
        
        if self.slice_dim == 'x':
            Y, Z = np.meshgrid(coords, coords)
            X = np.full_like(Y, self.slice_value)
        elif self.slice_dim == 'y':
            X, Z = np.meshgrid(coords, coords)
            Y = np.full_like(X, self.slice_value)
        else:
            X, Y = np.meshgrid(coords, coords)
            Z = np.full_like(X, self.slice_value)
        
        W = self.func(X, Y, Z)
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        range_val = self.domain[1] - self.domain[0]
        w_range = self.w_max - self.w_min
        max_range = max(range_val, w_range) if w_range > 0 else range_val
        
        center = self.domain[0] + range_val / 2
        w_center = (self.w_min + self.w_max) / 2
        
        # Build mesh
        num_quads = (self.resolution - 1) ** 2
        self.vertices = np.zeros((num_quads * 6, 3))
        self.colors_t = np.zeros(num_quads * 2)
        
        idx = 0
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                v00 = np.array([(coords[i] - center) / max_range * 2,
                               (coords[j] - center) / max_range * 2,
                               (W[i, j] - w_center) / max_range * 2])
                v10 = np.array([(coords[i+1] - center) / max_range * 2,
                               (coords[j] - center) / max_range * 2,
                               (W[i+1, j] - w_center) / max_range * 2])
                v01 = np.array([(coords[i] - center) / max_range * 2,
                               (coords[j+1] - center) / max_range * 2,
                               (W[i, j+1] - w_center) / max_range * 2])
                v11 = np.array([(coords[i+1] - center) / max_range * 2,
                               (coords[j+1] - center) / max_range * 2,
                               (W[i+1, j+1] - w_center) / max_range * 2])
                
                t_avg = ((W[i, j] + W[i+1, j] + W[i, j+1] + W[i+1, j+1]) / 4 - self.w_min) / (self.w_max - self.w_min)
                t_avg = np.clip(t_avg, 0, 1)
                
                self.vertices[idx*3] = v00
                self.vertices[idx*3 + 1] = v10
                self.vertices[idx*3 + 2] = v11
                self.colors_t[idx] = t_avg
                idx += 1
                
                self.vertices[idx*3] = v00
                self.vertices[idx*3 + 1] = v11
                self.vertices[idx*3 + 2] = v01
                self.colors_t[idx] = t_avg
                idx += 1
    
    def render(self, screen: pygame.Surface, camera: Camera, colormap: Callable,
               view_rect: pygame.Rect):
        """Render the hypersurface slice"""
        R = camera.get_rotation_matrix()
        rotated = self.vertices @ R.T
        
        num_tris = len(self.colors_t)
        tri_data = []
        
        for i in range(num_tris):
            r0 = rotated[i*3]
            r1 = rotated[i*3 + 1]
            r2 = rotated[i*3 + 2]
            
            depth = (r0[1] + r1[1] + r2[1]) / 3
            
            edge1 = r1 - r0
            edge2 = r2 - r0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
            
            s0 = (int(camera.center_x + r0[0] * camera.zoom),
                  int(camera.center_y - r0[2] * camera.zoom))
            s1 = (int(camera.center_x + r1[0] * camera.zoom),
                  int(camera.center_y - r1[2] * camera.zoom))
            s2 = (int(camera.center_x + r2[0] * camera.zoom),
                  int(camera.center_y - r2[2] * camera.zoom))
            
            tri_data.append((s0, s1, s2, depth, normal, self.colors_t[i]))
        
        tri_data.sort(key=lambda x: x[3], reverse=True)
        
        for s0, s1, s2, depth, normal, t in tri_data:
            if not (view_rect.collidepoint(s0) or view_rect.collidepoint(s1) or view_rect.collidepoint(s2)):
                continue
            
            base_color = colormap(t)
            light_intensity = max(0.3, min(1.0, 0.4 + 0.6 * abs(np.dot(normal, camera.light_dir))))
            color = tuple(int(c * light_intensity) for c in base_color)
            
            try:
                pygame.draw.polygon(screen, color, [s0, s1, s2])
            except:
                pass


class HypersolidRenderer:
    """Renderer for 4D solids showing 3D cross-sections"""
    
    def __init__(self, func: Callable, domain_x: Tuple[float, float],
                 domain_y: Tuple[float, float], domain_z: Tuple[float, float],
                 domain_w: Tuple[float, float], resolution: int = 15):
        self.func = func
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.domain_w = domain_w
        self.resolution = resolution
        self.w_value = (domain_w[0] + domain_w[1]) / 2
        
        # Precompute coordinate arrays
        self.x_coords = np.linspace(domain_x[0], domain_x[1], resolution)
        self.y_coords = np.linspace(domain_y[0], domain_y[1], resolution)
        self.z_coords = np.linspace(domain_z[0], domain_z[1], resolution)
        
        # Normalization factors
        x_range = domain_x[1] - domain_x[0]
        y_range = domain_y[1] - domain_y[0]
        z_range = domain_z[1] - domain_z[0]
        self.max_range = max(x_range, y_range, z_range)
        self.x_center = (domain_x[0] + domain_x[1]) / 2
        self.y_center = (domain_y[0] + domain_y[1]) / 2
        self.z_center = (domain_z[0] + domain_z[1]) / 2
        self.w_range = domain_w[1] - domain_w[0]
        
        # Cache for mesh data
        self.vertices = None
        self.normals = None
        self.color_t = 0.5
        
        self._update_slice()
    
    def set_w_value(self, w: float):
        self.w_value = np.clip(w, self.domain_w[0], self.domain_w[1])
        self._update_slice()
    
    def _update_slice(self):
        """Compute the 3D cross-section at current w value"""
        X, Y, Z = np.meshgrid(self.x_coords, self.y_coords, self.z_coords, indexing='ij')
        W = np.full_like(X, self.w_value)
        
        scalar_field = self.func(X, Y, Z, W)
        scalar_field = np.nan_to_num(scalar_field, nan=1e10, posinf=1e10, neginf=-1e10)
        
        # Extract isosurface
        raw_triangles = marching_cubes(scalar_field, 0.0, self.x_coords, self.y_coords, self.z_coords)
        
        # Color based on w position
        self.color_t = (self.w_value - self.domain_w[0]) / self.w_range if self.w_range > 0 else 0.5
        
        if not raw_triangles:
            self.vertices = None
            self.normals = None
            return
        
        # Convert to arrays and normalize
        num_tris = len(raw_triangles)
        self.vertices = np.zeros((num_tris * 3, 3))
        self.normals = np.zeros((num_tris, 3))
        
        for i, (v0, v1, v2) in enumerate(raw_triangles):
            # Normalize vertices
            self.vertices[i*3] = np.array([
                (v0[0] - self.x_center) / self.max_range * 2,
                (v0[1] - self.y_center) / self.max_range * 2,
                (v0[2] - self.z_center) / self.max_range * 2
            ])
            self.vertices[i*3 + 1] = np.array([
                (v1[0] - self.x_center) / self.max_range * 2,
                (v1[1] - self.y_center) / self.max_range * 2,
                (v1[2] - self.z_center) / self.max_range * 2
            ])
            self.vertices[i*3 + 2] = np.array([
                (v2[0] - self.x_center) / self.max_range * 2,
                (v2[1] - self.y_center) / self.max_range * 2,
                (v2[2] - self.z_center) / self.max_range * 2
            ])
            
            # Compute normal
            edge1 = self.vertices[i*3 + 1] - self.vertices[i*3]
            edge2 = self.vertices[i*3 + 2] - self.vertices[i*3]
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                self.normals[i] = normal / norm_len
            else:
                self.normals[i] = np.array([0, 0, 1])
    
    def render(self, screen: pygame.Surface, camera: Camera, colormap: Callable,
               view_rect: pygame.Rect):
        """Render the 3D cross-section"""
        if self.vertices is None or len(self.vertices) == 0:
            font = pygame.font.SysFont('Arial', 16)
            text = font.render("No intersection at this W value", True, (150, 150, 170))
            screen.blit(text, (view_rect.centerx - text.get_width() // 2, view_rect.centery))
            return
        
        R = camera.get_rotation_matrix()
        
        # Transform all vertices
        rotated = self.vertices @ R.T
        rotated_normals = self.normals @ R.T
        
        # Build triangle data
        num_tris = len(self.normals)
        tri_data = []
        
        base_color = colormap(self.color_t)
        
        for i in range(num_tris):
            r0 = rotated[i*3]
            r1 = rotated[i*3 + 1]
            r2 = rotated[i*3 + 2]
            normal = rotated_normals[i]
            
            depth = (r0[1] + r1[1] + r2[1]) / 3
            
            s0 = (int(camera.center_x + r0[0] * camera.zoom),
                  int(camera.center_y - r0[2] * camera.zoom))
            s1 = (int(camera.center_x + r1[0] * camera.zoom),
                  int(camera.center_y - r1[2] * camera.zoom))
            s2 = (int(camera.center_x + r2[0] * camera.zoom),
                  int(camera.center_y - r2[2] * camera.zoom))
            
            # Compute lighting
            light_intensity = max(0.3, min(1.0, 0.4 + 0.6 * abs(np.dot(normal, camera.light_dir))))
            color = tuple(int(c * light_intensity) for c in base_color)
            
            tri_data.append((s0, s1, s2, depth, color))
        
        # Sort by depth
        tri_data.sort(key=lambda x: x[3], reverse=True)
        
        # Render triangles
        for s0, s1, s2, depth, color in tri_data:
            if not (view_rect.collidepoint(s0) or view_rect.collidepoint(s1) or view_rect.collidepoint(s2)):
                continue
            try:
                pygame.draw.polygon(screen, color, [s0, s1, s2])
            except:
                pass


# =============================================================================
# UI COMPONENTS
# =============================================================================

class Button:
    """Simple button component"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: Tuple[int, int, int] = (60, 60, 80),
                 hover_color: Tuple[int, int, int] = (80, 80, 100),
                 text_color: Tuple[int, int, int] = (220, 220, 230)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.selected = False
    
    def set_position(self, x: int, y: int):
        self.rect.x = x
        self.rect.y = y
    
    def update(self, mouse_pos: Tuple[int, int]):
        self.hovered = self.rect.collidepoint(mouse_pos)
    
    def render(self, screen: pygame.Surface, font: pygame.font.Font):
        color = self.hover_color if self.hovered else self.color
        if self.selected:
            color = (100, 150, 255)
        
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 1, border_radius=5)
        
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_pressed: bool) -> bool:
        return self.rect.collidepoint(mouse_pos) and mouse_pressed


class Slider:
    """Slider component for continuous values"""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 min_val: float, max_val: float, initial: float,
                 label: str = "", format_str: str = "{:.2f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.format_str = format_str
        self.dragging = False
        self.handle_radius = 8
    
    def set_position(self, x: int, y: int):
        self.rect.x = x
        self.rect.y = y
    
    def _get_handle_x(self) -> int:
        if self.max_val == self.min_val:
            t = 0.5
        else:
            t = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + 10 + t * (self.rect.width - 20))
    
    def _value_from_x(self, x: int) -> float:
        t = (x - self.rect.x - 10) / (self.rect.width - 20)
        t = np.clip(t, 0, 1)
        return self.min_val + t * (self.max_val - self.min_val)
    
    def update(self, mouse_pos: Tuple[int, int], mouse_pressed: bool):
        handle_x = self._get_handle_x()
        handle_rect = pygame.Rect(handle_x - self.handle_radius, 
                                  self.rect.centery - self.handle_radius,
                                  self.handle_radius * 2, self.handle_radius * 2)
        
        if mouse_pressed and handle_rect.collidepoint(mouse_pos):
            self.dragging = True
        elif not mouse_pressed:
            self.dragging = False
        
        if self.dragging:
            self.value = self._value_from_x(mouse_pos[0])
    
    def render(self, screen: pygame.Surface, font: pygame.font.Font):
        if self.label:
            label_surf = font.render(self.label, True, (200, 200, 210))
            screen.blit(label_surf, (self.rect.x, self.rect.y - 18))
        
        track_rect = pygame.Rect(self.rect.x + 10, self.rect.centery - 2,
                                 self.rect.width - 20, 4)
        pygame.draw.rect(screen, (50, 50, 70), track_rect, border_radius=2)
        
        handle_x = self._get_handle_x()
        filled_rect = pygame.Rect(self.rect.x + 10, self.rect.centery - 2,
                                  handle_x - self.rect.x - 10, 4)
        pygame.draw.rect(screen, (100, 150, 255), filled_rect, border_radius=2)
        
        pygame.draw.circle(screen, (100, 150, 255), (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(screen, (150, 200, 255), (handle_x, self.rect.centery), self.handle_radius - 2)
        
        value_str = self.format_str.format(self.value)
        value_surf = font.render(value_str, True, (200, 200, 210))
        screen.blit(value_surf, (self.rect.right + 10, self.rect.centery - 8))


class UILayout:
    """Dynamic UI layout manager"""
    
    def __init__(self, config: Config):
        self.config = config
        self.panel_x = config.VIEW_X + config.VIEW_WIDTH + 20
        self.panel_width = config.WINDOW_WIDTH - self.panel_x - 20
        self.panel_height = config.WINDOW_HEIGHT - 20
        self.spacing = 6
        self.section_spacing = 12
        self.current_y = 25
    
    def reset(self):
        self.current_y = 25
    
    def add_header(self, height: int = 22) -> int:
        y = self.current_y
        self.current_y += height + self.spacing
        return y
    
    def add_button(self, height: int = 26) -> int:
        y = self.current_y
        self.current_y += height + self.spacing
        return y
    
    def add_button_row(self, count: int, height: int = 24) -> int:
        y = self.current_y
        self.current_y += height + self.spacing
        return y
    
    def add_slider(self, height: int = 18) -> int:
        y = self.current_y + 16
        self.current_y += height + 22 + self.spacing
        return y
    
    def add_section_break(self):
        self.current_y += self.section_spacing
    
    def fits_in_panel(self) -> bool:
        """Check if current content fits in panel"""
        return self.current_y < self.panel_height - 10


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class HypersolidViewer:
    """Main application class"""
    
    def __init__(self):
        pygame.init()
        self.config = Config()
        self.screen = pygame.display.set_mode((self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT))
        pygame.display.set_caption("4D Hypersolid & Surface Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 13)
        self.title_font = pygame.font.SysFont('Arial', 14, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 11)
        
        # State
        self.running = True
        self.vis_type = VisualizationType.HYPERSOLID_4D
        self.current_func_id = 1
        self.current_colormap = 'viridis'
        self.camera = Camera(self.config)
        self.renderer = None
        self.animating = False
        self.animation_speed = 0.02
        self.animation_direction = 1
        self.current_name = ""
        self.current_formula = ""
        
        # Mouse state
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        # View rect
        self.view_rect = pygame.Rect(self.config.VIEW_X, self.config.VIEW_Y,
                                     self.config.VIEW_WIDTH, self.config.VIEW_HEIGHT)
        
        # UI Layout
        self.layout = UILayout(self.config)
        
        self._create_ui_components()
        self._load_function()
    
    def _create_ui_components(self):
        """Create all UI components"""
        panel_x = self.layout.panel_x
        panel_width = self.layout.panel_width
        
        self.type_buttons = [
            Button(0, 0, panel_width, 26, "3D Surface"),
            Button(0, 0, panel_width, 26, "4D Hypersurface"),
            Button(0, 0, panel_width, 26, "4D Solid"),
        ]
        self.type_buttons[2].selected = True
        
        self.func_buttons = []
        
        btn_width = (panel_width - 8) // 2
        self.cmap_buttons = [
            Button(0, 0, btn_width, 24, "Viridis"),
            Button(0, 0, btn_width, 24, "Plasma"),
            Button(0, 0, btn_width, 24, "Inferno"),
            Button(0, 0, btn_width, 24, "CoolWarm"),
        ]
        self.cmap_buttons[0].selected = True
        
        btn3_width = (panel_width - 16) // 3
        self.slice_dim_buttons = [
            Button(0, 0, btn3_width, 24, "X"),
            Button(0, 0, btn3_width, 24, "Y"),
            Button(0, 0, btn3_width, 24, "Z"),
        ]
        self.slice_dim_buttons[2].selected = True
        
        self.slice_slider = Slider(0, 0, panel_width - 50, 18, -2, 2, 0, "W Value")
        
        self.animate_button = Button(0, 0, panel_width, 26, "▶ Animate")
        self.reset_button = Button(0, 0, panel_width, 26, "Reset View")
    
    def _update_layout(self):
        """Update UI component positions"""
        self.layout.reset()
        panel_x = self.layout.panel_x
        panel_width = self.layout.panel_width
        
        self.header_positions = {}
        
        # Type section
        self.header_positions['type'] = self.layout.add_header()
        for btn in self.type_buttons:
            y = self.layout.add_button()
            btn.set_position(panel_x, y)
        
        self.layout.add_section_break()
        
        # Function section
        self.header_positions['function'] = self.layout.add_header()
        self._update_func_buttons()
        for _, btn in self.func_buttons:
            y = self.layout.add_button(height=24)
            btn.set_position(panel_x, y)
        
        self.layout.add_section_break()
        
        # Colormap section
        self.header_positions['colormap'] = self.layout.add_header()
        btn_width = (panel_width - 8) // 2
        y = self.layout.add_button_row(2)
        self.cmap_buttons[0].set_position(panel_x, y)
        self.cmap_buttons[1].set_position(panel_x + btn_width + 8, y)
        y = self.layout.add_button_row(2)
        self.cmap_buttons[2].set_position(panel_x, y)
        self.cmap_buttons[3].set_position(panel_x + btn_width + 8, y)
        
        self.layout.add_section_break()
        
        # Type-specific controls
        if self.vis_type == VisualizationType.HYPERSURFACE_4D:
            self.header_positions['slice_dim'] = self.layout.add_header()
            btn3_width = (panel_width - 16) // 3
            y = self.layout.add_button_row(3)
            self.slice_dim_buttons[0].set_position(panel_x, y)
            self.slice_dim_buttons[1].set_position(panel_x + btn3_width + 8, y)
            self.slice_dim_buttons[2].set_position(panel_x + btn3_width * 2 + 16, y)
            
            self.layout.add_section_break()
            self.slice_slider.label = "Slice Position"
        elif self.vis_type == VisualizationType.HYPERSOLID_4D:
            self.slice_slider.label = "W Value"
        
        # Slider section
        if self.vis_type in [VisualizationType.HYPERSURFACE_4D, VisualizationType.HYPERSOLID_4D]:
            self.header_positions['sliders'] = self.layout.add_header()
            y = self.layout.add_slider()
            self.slice_slider.set_position(panel_x, y)
        
        self.layout.add_section_break()
        
        # Controls section
        self.header_positions['controls'] = self.layout.add_header()
        y = self.layout.add_button()
        self.animate_button.set_position(panel_x, y)
        y = self.layout.add_button()
        self.reset_button.set_position(panel_x, y)
    
    def _update_func_buttons(self):
        """Update function buttons"""
        panel_x = self.layout.panel_x
        panel_width = self.layout.panel_width
        
        if self.vis_type == VisualizationType.SURFACE_3D:
            funcs = SURFACES
        elif self.vis_type == VisualizationType.HYPERSURFACE_4D:
            funcs = HYPERSURFACES
        else:
            funcs = HYPERSOLIDS
        
        self.func_buttons = []
        for func_id, func_data in funcs.items():
            name = func_data[1]
            btn = Button(0, 0, panel_width, 24, f"{func_id}. {name}")
            if func_id == self.current_func_id:
                btn.selected = True
            self.func_buttons.append((func_id, btn))
    
    def _load_function(self):
        """Load the current function"""
        if self.vis_type == VisualizationType.SURFACE_3D:
            if self.current_func_id in SURFACES:
                func, name, formula, domain_x, domain_y = SURFACES[self.current_func_id]
                self.renderer = Surface3DRenderer(func, domain_x, domain_y)
                self.current_name = name
                self.current_formula = formula
        
        elif self.vis_type == VisualizationType.HYPERSURFACE_4D:
            if self.current_func_id in HYPERSURFACES:
                func, name, formula, domain = HYPERSURFACES[self.current_func_id]
                self.renderer = HypersurfaceRenderer(func, domain)
                self.renderer.set_slice('z', 0)
                self.slice_slider.min_val = domain[0]
                self.slice_slider.max_val = domain[1]
                self.slice_slider.value = 0
                self.current_name = name
                self.current_formula = formula
        
        elif self.vis_type == VisualizationType.HYPERSOLID_4D:
            if self.current_func_id in HYPERSOLIDS:
                func, name, formula, dx, dy, dz, dw = HYPERSOLIDS[self.current_func_id]
                self.renderer = HypersolidRenderer(
                    func, dx, dy, dz, dw,
                    resolution=self.config.HYPERSOLID_RESOLUTION
                )
                self.slice_slider.min_val = dw[0]
                self.slice_slider.max_val = dw[1]
                self.slice_slider.value = (dw[0] + dw[1]) / 2
                self.renderer.set_w_value(self.slice_slider.value)
                self.current_name = name
                self.current_formula = formula
        
        self._update_layout()
    
    def _handle_events(self):
        """Handle pygame events"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.view_rect.collidepoint(mouse_pos):
                        self.dragging = True
                        self.last_mouse_pos = mouse_pos
                    
                    # Type buttons
                    for i, btn in enumerate(self.type_buttons):
                        if btn.is_clicked(mouse_pos, True):
                            for b in self.type_buttons:
                                b.selected = False
                            btn.selected = True
                            self.vis_type = list(VisualizationType)[i]
                            self.current_func_id = 1
                            self._load_function()
                    
                    # Function buttons
                    for func_id, btn in self.func_buttons:
                        if btn.is_clicked(mouse_pos, True):
                            for _, b in self.func_buttons:
                                b.selected = False
                            btn.selected = True
                            self.current_func_id = func_id
                            self._load_function()
                    
                    # Colormap buttons
                    cmap_names = ['viridis', 'plasma', 'inferno',  'cool_warm']
                    for i, btn in enumerate(self.cmap_buttons):
                        if btn.is_clicked(mouse_pos, True):
                            for b in self.cmap_buttons:
                                b.selected = False
                            btn.selected = True
                            self.current_colormap = cmap_names[i]
                    
                    # Slice dimension buttons
                    if self.vis_type == VisualizationType.HYPERSURFACE_4D:
                        dims = ['x', 'y', 'z']
                        for i, btn in enumerate(self.slice_dim_buttons):
                            if btn.is_clicked(mouse_pos, True):
                                for b in self.slice_dim_buttons:
                                    b.selected = False
                                btn.selected = True
                                if isinstance(self.renderer, HypersurfaceRenderer):
                                    self.renderer.set_slice(dims[i], self.slice_slider.value)
                    
                    # Animate button
                    if self.animate_button.is_clicked(mouse_pos, True):
                        self.animating = not self.animating
                        self.animate_button.text = "⏸ Pause" if self.animating else "▶ Animate"
                    
                    # Reset button
                    if self.reset_button.is_clicked(mouse_pos, True):
                        self.camera.reset()
                
                elif event.button == 4:
                    if self.view_rect.collidepoint(mouse_pos):
                        self.camera.zoom_in(1.1)
                
                elif event.button == 5:
                    if self.view_rect.collidepoint(mouse_pos):
                        self.camera.zoom_in(0.9)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = mouse_pos[0] - self.last_mouse_pos[0]
                    dy = mouse_pos[1] - self.last_mouse_pos[1]
                    self.camera.rotate(dx * 0.5, -dy * 0.5)
                    self.last_mouse_pos = mouse_pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.animating = not self.animating
                    self.animate_button.text = "⏸ Pause" if self.animating else "▶ Animate"
                elif event.key == pygame.K_r:
                    self.camera.reset()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
        
        # Update button hover states
        for btn in self.type_buttons:
            btn.update(mouse_pos)
        for _, btn in self.func_buttons:
            btn.update(mouse_pos)
        for btn in self.cmap_buttons:
            btn.update(mouse_pos)
        for btn in self.slice_dim_buttons:
            btn.update(mouse_pos)
        self.animate_button.update(mouse_pos)
        self.reset_button.update(mouse_pos)
        
        # Update slider
        old_slice = self.slice_slider.value
        
        if self.vis_type in [VisualizationType.HYPERSURFACE_4D, VisualizationType.HYPERSOLID_4D]:
            self.slice_slider.update(mouse_pos, mouse_pressed)
        
        # Update renderer on slider change
        if abs(self.slice_slider.value - old_slice) > 0.001:
            if isinstance(self.renderer, HypersurfaceRenderer):
                dim = 'z'
                for i, btn in enumerate(self.slice_dim_buttons):
                    if btn.selected:
                        dim = ['x', 'y', 'z'][i]
                self.renderer.set_slice(dim, self.slice_slider.value)
            elif isinstance(self.renderer, HypersolidRenderer):
                self.renderer.set_w_value(self.slice_slider.value)
    
    def _update(self):
        """Update animation"""
        if self.animating:
            self.slice_slider.value += self.animation_speed * self.animation_direction
            
            if self.slice_slider.value >= self.slice_slider.max_val:
                self.animation_direction = -1
            elif self.slice_slider.value <= self.slice_slider.min_val:
                self.animation_direction = 1
            
            self.slice_slider.value = np.clip(
                self.slice_slider.value,
                self.slice_slider.min_val,
                self.slice_slider.max_val
            )
            
            if isinstance(self.renderer, HypersurfaceRenderer):
                dim = 'z'
                for i, btn in enumerate(self.slice_dim_buttons):
                    if btn.selected:
                        dim = ['x', 'y', 'z'][i]
                self.renderer.set_slice(dim, self.slice_slider.value)
            elif isinstance(self.renderer, HypersolidRenderer):
                self.renderer.set_w_value(self.slice_slider.value)
    
    def _render(self):
        """Render the scene"""
        self.screen.fill(self.config.BG_COLOR)
        
        # View area
        pygame.draw.rect(self.screen, (20, 20, 35), self.view_rect)
        pygame.draw.rect(self.screen, self.config.UI_BORDER_COLOR, self.view_rect, 2)
        
        # Title bar
        title_rect = pygame.Rect(self.config.VIEW_X, 10, self.config.VIEW_WIDTH, 70)
        pygame.draw.rect(self.screen, self.config.UI_BG_COLOR, title_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.config.UI_BORDER_COLOR, title_rect, 1, border_radius=5)
        
        title_text = self.title_font.render(f"{self.current_name}", True, self.config.TEXT_COLOR)
        self.screen.blit(title_text, (self.config.VIEW_X + 15, 20))
        
        formula_text = self.font.render(f"{self.current_formula}", True, (150, 150, 170))
        self.screen.blit(formula_text, (self.config.VIEW_X + 15, 45))
        
        if self.vis_type == VisualizationType.HYPERSOLID_4D:
            w_text = self.font.render(f"W = {self.slice_slider.value:.3f}", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(w_text, (self.config.VIEW_X + self.config.VIEW_WIDTH - 100, 45))
        
        # Render 3D content
        if self.renderer:
            colormap = COLORMAPS[self.current_colormap]
            self.renderer.render(self.screen, self.camera, colormap, self.view_rect)
        
        # Axes
        self._draw_axes()
        
        # UI panel
        self._draw_ui_panel()
        
        # Instructions
        self._draw_instructions()
        
        pygame.display.flip()
    
    def _draw_axes(self):
        """Draw coordinate axes"""
        origin_x = self.config.VIEW_X + 60
        origin_y = self.config.VIEW_Y + self.config.VIEW_HEIGHT - 60
        axis_length = 40
        
        R = self.camera.get_rotation_matrix()
        
        axes = [
            (np.array([1, 0, 0]), (255, 100, 100), 'X'),
            (np.array([0, 1, 0]), (100, 255, 100), 'Y'),
            (np.array([0, 0, 1]), (100, 100, 255), 'Z'),
        ]
        
        for axis_vec, color, label in axes:
            rotated = R @ axis_vec
            
            end_x = int(origin_x + rotated[0] * axis_length)
            end_y = int(origin_y - rotated[2] * axis_length)
            
            pygame.draw.line(self.screen, color, (origin_x, origin_y), (end_x, end_y), 2)
            
            dx = end_x - origin_x
            dy = end_y - origin_y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx/length, dy/length
                arrow_size = 6
                ax1 = end_x - arrow_size * (dx + dy * 0.5)
                ay1 = end_y - arrow_size * (dy - dx * 0.5)
                ax2 = end_x - arrow_size * (dx - dy * 0.5)
                ay2 = end_y - arrow_size * (dy + dx * 0.5)
                pygame.draw.polygon(self.screen, color, [(end_x, end_y), (int(ax1), int(ay1)), (int(ax2), int(ay2))])
            
            label_surf = self.small_font.render(label, True, color)
            self.screen.blit(label_surf, (end_x + 5, end_y - 5))
    
    def _draw_ui_panel(self):
        """Draw the UI panel"""
        panel_x = self.layout.panel_x
        panel_width = self.layout.panel_width
        
        # Panel background
        panel_rect = pygame.Rect(panel_x - 10, 10, panel_width + 20, self.config.WINDOW_HEIGHT - 20)
        pygame.draw.rect(self.screen, self.config.UI_BG_COLOR, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.config.UI_BORDER_COLOR, panel_rect, 1, border_radius=10)
        
        # Headers and components
        if 'type' in self.header_positions:
            header = self.title_font.render("Visualization Type", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(header, (panel_x, self.header_positions['type']))
        
        for btn in self.type_buttons:
            btn.render(self.screen, self.font)
        
        if 'function' in self.header_positions:
            header = self.title_font.render("Function", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(header, (panel_x, self.header_positions['function']))
        
        for _, btn in self.func_buttons:
            btn.render(self.screen, self.font)
        
        if 'colormap' in self.header_positions:
            header = self.title_font.render("Colormap", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(header, (panel_x, self.header_positions['colormap']))
        
        for btn in self.cmap_buttons:
            btn.render(self.screen, self.font)
        
        if self.vis_type == VisualizationType.HYPERSURFACE_4D and 'slice_dim' in self.header_positions:
            header = self.title_font.render("Slice Dimension", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(header, (panel_x, self.header_positions['slice_dim']))
            for btn in self.slice_dim_buttons:
                btn.render(self.screen, self.font)
        
        if self.vis_type in [VisualizationType.HYPERSURFACE_4D, VisualizationType.HYPERSOLID_4D]:
            self.slice_slider.render(self.screen, self.font)
        
        if 'controls' in self.header_positions:
            header = self.title_font.render("Controls", True, self.config.HIGHLIGHT_COLOR)
            self.screen.blit(header, (panel_x, self.header_positions['controls']))
        
        self.animate_button.render(self.screen, self.font)
        self.reset_button.render(self.screen, self.font)
    
    def _draw_instructions(self):
        """Draw usage instructions"""
        instructions = [
            "Drag: Rotate",
            "Scroll: Zoom",
            "Space: Animate",
            "R: Reset",
            "Esc: Quit"
        ]
        
        x = self.config.VIEW_X + self.config.VIEW_WIDTH - 100
        y = self.config.VIEW_Y + 10
        
        for inst in instructions:
            text = self.small_font.render(inst, True, (100, 100, 120))
            self.screen.blit(text, (x, y))
            y += 14
    
    def run(self):
        """Main application loop"""
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(self.config.FPS)
        
        pygame.quit()
        sys.exit()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    viewer = HypersolidViewer()
    viewer.run()


if __name__ == "__main__":
    main()