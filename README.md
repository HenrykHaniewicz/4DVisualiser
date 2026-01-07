# Interactive 4D Visualisation Viewer (Pygame)

An interactive visual application for exploring:

- **3D surfaces** (height fields `z = f(x, y)`)
- **4D hypersurfaces** (visualised as **3D slices** of `w = f(x, y, z)`)
- **4D hypersolids** (visualised as **3D cross-sections** of an implicit 4D solid `F(x, y, z, w) = 0`)

Rendering is triangle-mesh based with depth sorting and simple lighting/shading.

---

## Requirements

- Python 3.10+
- `numpy`
- `pygame`

Install dependencies:

```
pip install numpy pygame
```

---

## Run

From the project directory:

```
python visualiser.py
```

A window titled **"4D Hypersolid & Surface Viewer"** should open.

---

## Controls

### Mouse
- **Left-click + drag (in the view area):** rotate camera
- **Mouse wheel:** zoom in/out

### Keyboard
- **Space:** toggle animation (same as the "Animate" button)
- **R:** reset view
- **Esc:** quit

---

## Using the UI Panel

The right-side panel lets you:

1. **Choose Visualisation Type**
   - **3D Surface**
   - **4D Hypersurface**
   - **4D Solid**

2. **Pick a Function**
   - The function list is populated from `functions.py`:
     - `SURFACES` for 3D surfaces
     - `HYPERSURFACES` for 4D hypersurfaces
     - `HYPERSOLIDS` for 4D solids

3. **Select a Colormap**
   - Viridis / Plasma / Inferno / CoolWarm

4. **Slice / Cross-section Control**
   - For **4D Hypersurface**: choose slice dimension (**X/Y/Z**) and adjust **Slice Position** with the slider.
   - For **4D Solid**: adjust **W Value** with the slider to move the 3D cross-section through the 4D shape.
     - If there is no intersection at the current W, you will see: *"No intersection at this W value"*

5. **Controls**
   - **Animate / Pause** button oscillates the slice value between min and max.
   - **Reset View** resets camera elevation/azimuth/zoom.

---

## Adding / Editing Functions

Functions are defined in `functions.py` as dictionaries.

This program expects these structures (based on how `_load_function()` unpacks them):

### 3D Surfaces

`SURFACES[func_id] = (func, name, formula, domain_x, domain_y)`

- `func(X, Y) -> Z` where `X`, `Y`, `Z` are numpy arrays
- `domain_x = (xmin, xmax)`
- `domain_y = (ymin, ymax)`

### 4D Hypersurfaces

`HYPERSURFACES[func_id] = (func, name, formula, domain)`

- `func(X, Y, Z) -> W`
- `domain = (min, max)` (used for X/Y/Z and slice value range)

### 4D Hypersolids (implicit)

`HYPERSOLIDS[func_id] = (func, name, formula, dx, dy, dz, dw)`

- `func(X, Y, Z, W) -> scalar_field`
- The viewer extracts the **isosurface at 0.0** (i.e. `func(...) = 0`) using marching cubes.
- Domains:
  - `dx = (xmin, xmax)`
  - `dy = (ymin, ymax)`
  - `dz = (zmin, zmax)`
  - `dw = (wmin, wmax)` (this is what the slider controls)

Note:
- NaNs/Infs are handled, but extreme values can still cause empty/noisy meshes.

---

## Performance Notes

- The 4D solid cross-section uses marching cubes on a fixed grid.
- Grid resolution is controlled by `Config.HYPERSOLID_RESOLUTION` (default `15`).
  - Higher values look nicer but can get slow quickly.

---

## Troubleshooting

- **Black screen / nothing visible:** try zooming (scroll) or hit **R** to reset view.
- **No intersection (4D solid):** move the **W Value** slider to a different position.
- **Import errors:** confirm `functions.py` and `marching_cubes.py` are present and importable.
- **Slow performance:** reduce hypersolid resolution in `Config` or simplify the function.
