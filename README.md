# Fractal — WebGPU Realtime Explorer

A WebGPU-based realtime Mandelbrot and Julia set explorer built with TypeScript, WGSL compute shaders, and Vite. Maintains a responsive **60Hz** framerate while panning and zooming, automatically refines to full HiDPI resolution with **8× supersampling (SSAA)** and deep iteration counts when stationary, and renders continuously flowing, ever-changing gradients in perceptual **Oklab** color space.

**Live Site:** https://hulkholden.github.io/fractal/

> Requires a browser with WebGPU enabled (Chrome, Edge, or Safari with WebGPU support).

---

## Highlights

- **60Hz Interactive Navigation + Stationary Progressive Refinement**: Dynamically adapts render resolution during pan/zoom/orbit animation to stay near 60Hz, then smoothly refines to native DPR, deep chunked iterations (up to 24,576), and 8× low-discrepancy SSAA once the viewport settles.
- **Automatic Three-Tier Precision (`FP32` → `DF64` → `PERTURB`)**: Seamlessly scales from native 32-bit floats (`1×`), to emulated double-float (`2×10⁴×`), to arbitrary-precision **Perturbation Theory + Zhuoran Rebasing** (`2×10¹¹×` up to **`10¹⁴⁰×`**).
- **Smooth Ever-Changing Gradients (*Living Gradient*)**: Continuous renormalized escape potential rendered through periodic 8-stop Catmull-Rom splines in Oklab color space—continuously cycling along contours while smoothly morphing across 6 curated palettes with subtle hue breathing.
- **Mandelbrot & Julia Sets**: Interactive Julia `Re(c)` / `Im(c)` sliders, 6 smoothly morphing presets, a 60Hz animated `c`-orbit (`0.7885·e^(iθ)`), and `Alt+Click` (or **Explore Julia at Center →**) to jump straight from any Mandelbrot coordinate into its corresponding Julia set.
- **Shareable URLs & High-Res Snapshots**: Share deep-zoom coordinates via query parameters or export full-resolution PNGs directly from the WebGPU canvas.

---

## Implementation Details

### 1. Decoupled Compute & 60Hz Display Pipeline

The renderer splits fractal evaluation and gradient coloring into two independent GPU stages ([`src/engine/renderer.ts`](src/engine/renderer.ts)):

1. **Compute Stage** ([`src/engine/shaders/compute_entry.wgsl`](src/engine/shaders/compute_entry.wgsl)): Evaluates smooth escape values into two `vec4<f32>` storage buffers (`samples0`, `samples1`, holding up to 8 subpixel samples per pixel) plus resumable per-pixel iteration state (`state_z`, `state_meta`).
   - **`main_interactive`**: Register-only single-pass kernel with zero state-buffer traffic, dispatched at an adaptive resolution scale (`Auto 60Hz`, `0.5×`, `0.75×`, or `1×`) while the camera or Julia parameter is moving.
   - **`main_refine`**: Triggered ~95ms after motion stops. Immediately upgrades to native DPR and advances iterations in time-sliced chunks per frame—pixels that already escaped exit on the first check, so refinement accelerates as the exterior resolves while keeping every frame within the 60Hz budget. Each of the 8 dyadic rotated-grid / low-discrepancy subpixels blends into the display only once completed, eliminating flicker.
2. **Display Stage** ([`src/engine/shaders/display.wgsl`](src/engine/shaders/display.wgsl)): A lightweight fullscreen-triangle render pass that runs every frame at 60Hz, even when stationary refinement has converged.
   - In downscaled interactive mode, evaluates the palette per texel and performs bilinear blending in **linear RGB** to avoid interior/exterior halo artifacts.
   - In native stationary mode, resolves up to 8 supersampled escape values in linear light before sRGB encoding.

### 2. Three-Tier Deep-Zoom Precision Engine

Zoom depth is tracked logarithmically (`log2Zoom`) alongside arbitrary-precision binary fixed-point coordinates ([`src/engine/camera.ts`](src/engine/camera.ts), [`src/engine/bigfixed.ts`](src/engine/bigfixed.ts)), selecting the fastest precision tier automatically:

| Tier | Zoom Range | Implementation |
| --- | --- | --- |
| **`FP32`** | `< 2×10⁴` | Native 32-bit IEEE-754 GPU floats; Mandelbrot includes analytical main-cardioid & period-2 bulb early bailout. |
| **`DF64`** | `2×10⁴` – `2×10¹¹` | Emulated double-float (`vec2<f32>` hi/lo pairs, ~48-bit mantissa) in [`common.wgsl`](src/engine/shaders/common.wgsl). Uses hardware `fma()` with a runtime uniform `1.0` factor so GPU fast-math optimizers (notably Apple Metal) cannot algebraically collapse Knuth/Dekker `two_sum` / `quick_two_sum` error terms. |
| **`PERTURB`** | `≥ 2×10¹¹` (up to `10¹⁴⁰`) | **Perturbation Theory + Zhuoran Rebasing** for both Mandelbrot and Julia sets. |

#### How Perturbation Mode Works
- **Arbitrary-Precision Reference Orbit (CPU)**: [`bigfixed.ts`](src/engine/bigfixed.ts) uses zero-dependency native `BigInt` fixed-point arithmetic (auto-scaling from 128 to 512+ fractional bits) to compute a single reference orbit $Z_n$ near the viewport center—probing a surrounding ring if the center escapes early—and uploads $Z_n$ as `array<vec2<f32>>` to the GPU. The orbit is cached across pans/zooms while within `3.5×` viewport span and extended incrementally during stationary deepening.
- **Rescaled Floatexp Deltas (GPU)**: Every pixel iterates only its tiny relative offset $\delta_n = z_n - Z_n$ via $\delta_{n+1} = 2 Z_n \delta_n + \delta_n^2 + \Delta c$ (or without $+\Delta c$ for Julia sets). To prevent `f32` underflow past `10⁻³⁸`, each pixel maintains a factored power-of-two exponent normalized via WGSL built-in `frexp()` / `ldexp()`, allowing clean zooms past `10⁴⁴` and up to `10¹⁴⁰`.
- **Zhuoran Rebasing**: Whenever a pixel's full orbit position $|Z_n + \delta_n|$ drops below $|\delta_n|$ (or hits the end of the reference orbit), its reference index resets cleanly to `0`, preventing perturbation glitches automatically.

### 3. Smooth Ever-Changing Gradients (Oklab)

- **Band-Free Escape Potential**: Iterating to a large bailout radius ($R = 256$, $R^2 = 65536$) makes the renormalized count $\mu = n + 1 - \log_2(\log_2(|z|))$ $C^1$-continuous across iteration boundaries.
- **Balanced Contour Mapping**: Applies $\ln(\mu + 1) \cdot 0.42 + \sqrt{\mu} \cdot 0.045$ so wide exterior gradients sweep smoothly while deep high-iteration spirals retain crisp color separation.
- **Perceptual Interpolation**: Curated 8-stop periodic palettes ([`src/engine/palettes.ts`](src/engine/palettes.ts)) are converted to **Oklab** on the CPU and interpolated on the GPU with a periodic Catmull-Rom spline (`catmull_rom`), avoiding the muddy desaturated midpoints of sRGB gradients. When *Living Morph* is enabled, palettes smoothly cross-fade around the ring with gentle Oklab `(a, b)` hue rotation.

### 4. Modular Fractal Registry

[`src/engine/fractals.ts`](src/engine/fractals.ts) defines a shared `FractalDefinition` interface and assembles each compute shader by concatenating `common.wgsl` + `<fractal>_kernel.wgsl` + `compute_entry.wgsl`. Compiled compute pipelines are cached per fractal ID for instant switching.

---

## Controls & Shortcuts

| Input | Action |
| --- | --- |
| **Drag** | Pan viewport (60Hz adaptive) |
| **Scroll / Trackpad Pinch** | Zoom smoothly toward cursor |
| **Double-click** (`Shift` + Double-click) | Smooth fly-zoom in (or out) |
| **Alt + Click** *(on Mandelbrot)* | Open the Julia set for that exact complex coordinate `c` |
| `Space` | Pause / resume gradient flow |
| `M` | Toggle Living Gradient auto-morph |
| `O` | Toggle animated Julia `c`-orbit |
| `1` – `6` | Select gradient palette directly |
| `R` | Reset view to default framing |
| `H` | Hide / show entire HUD |
| `+` / `-` / Arrow keys | Keyboard zoom & pan |

### URL Query Parameters

Share or deep-link directly to any location and palette:

- `fractal` — `mandelbrot` or `julia`
- `bookmark` — preset destination ID (e.g. `seahorse-valley`, `mini-brot`, `spiral-abyss-18`, `subfloat-44`, `julia-galaxy`, `julia-ultra-24`)
- `x`, `y` — center coordinates (supports arbitrary-precision decimal strings)
- `z` or `logz` — linear zoom (`z=420`) or $\log_{10}(\text{zoom})$ (`logz=44`)
- `cre`, `cim` — Julia parameter $c = \text{Re}(c) + i\,\text{Im}(c)$
- `orbit=1` — start with animated Julia `c`-orbit enabled
- `palette` — `ultra-classic`, `prism-dream`, `bioluminescent`, `aurora`, `solar-corona`, or `iridescent-opal`

Example: `https://hulkholden.github.io/fractal/?bookmark=subfloat-44&palette=solar-corona`

---

## Project Structure

```text
├── index.html
├── src/
│   ├── main.ts                        # App bootstrap, URL params, input handling, 60Hz loop
│   ├── style.css                      # Glassmorphic dark HUD & responsive mobile drawer
│   ├── engine/
│   │   ├── bigfixed.ts                # Arbitrary-precision BigInt fixed-point & reference orbits
│   │   ├── camera.ts                  # Arbitrary-depth camera, cursor zoom, exponential flights
│   │   ├── fractals.ts                # Fractal registry, Julia presets & c-orbit animation
│   │   ├── palettes.ts                # Oklab palette conversion & Living Gradient state
│   │   ├── renderer.ts                # WebGPU pipelines, 60Hz budget, progressive SSAA scheduler
│   │   └── shaders/
│   │       ├── common.wgsl            # Uniforms, fma DF64 math, rescaled perturbation + rebasing
│   │       ├── compute_entry.wgsl     # Interactive & chunked stationary compute entry points
│   │       ├── mandelbrot_kernel.wgsl # Mandelbrot kernel + cardioid/bulb early bailout
│   │       ├── julia_kernel.wgsl      # Quadratic Julia kernel
│   │       └── display.wgsl           # 60Hz Oklab spline coloring & linear-light SSAA resolve
│   └── ui/
│       ├── bookmarks.ts               # Curated FP32, DF64, and ultra-deep perturbation destinations
│       └── controls.ts                # Telemetry HUD, drawer controls, live gradient bar, PNG export
└── .github/workflows/deploy.yml       # GitHub Pages deployment via Bun + Vite
```

---

## Local Development

```bash
# Using Bun (matches CI deployment)
bun install
bun run dev

# Or using npm
npm install
npm run dev
```

Build and type-check for production:

```bash
bun run build   # runs `tsc --noEmit && vite build`
```
