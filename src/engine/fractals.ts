import commonWgsl from './shaders/common.wgsl?raw';
import computeEntryWgsl from './shaders/compute_entry.wgsl?raw';
import juliaKernelWgsl from './shaders/julia_kernel.wgsl?raw';
import mandelbrotKernelWgsl from './shaders/mandelbrot_kernel.wgsl?raw';

export interface FractalDefinition {
  id: 'mandelbrot' | 'julia';
  name: string;
  formula: string;
  description: string;
  defaultCenter: [number, number];
  defaultZoom: number;
  bailoutRadius: number;
  hasJuliaParam: boolean;
  kernelWgsl: string;
}

export interface JuliaPreset {
  id: string;
  name: string;
  subtitle: string;
  re: number;
  im: number;
}

export interface JuliaParamState {
  re: number;
  im: number;
  /** Continuously orbits c along 0.7885 * e^(i*theta) when enabled */
  animateOrbit: boolean;
  /** Orbit angular velocity in radians/sec */
  orbitSpeed: number;
  /** Current orbit angle in radians */
  orbitAngle: number;
  /** Optional smooth transition target when selecting a preset */
  transition: {
    startRe: number;
    startIm: number;
    endRe: number;
    endIm: number;
    elapsed: number;
    duration: number;
  } | null;
}

export const JULIA_ORBIT_RADIUS = 0.7885;

export const JULIA_PRESETS: readonly JuliaPreset[] = [
  {
    id: 'seahorse-spiral',
    name: 'Seahorse Spiral',
    subtitle: 'Filigree curls',
    re: -0.74543,
    im: 0.11301,
  },
  {
    id: 'golden-dragon',
    name: 'Golden Dragon',
    subtitle: 'Electric tendrils',
    re: -0.8,
    im: 0.156,
  },
  {
    id: 'galaxy-vortex',
    name: 'Galaxy Vortex',
    subtitle: 'Twin spiral arms',
    re: -0.4,
    im: 0.6,
  },
  {
    id: 'douady-rabbit',
    name: 'Douady Rabbit',
    subtitle: '3-lobed clusters',
    re: -0.12256,
    im: 0.74486,
  },
  {
    id: 'spiral-web',
    name: 'Spiral Web',
    subtitle: 'Luminous lace',
    re: 0.285,
    im: 0.01,
  },
  {
    id: 'siegel-disk',
    name: 'Siegel Disk',
    subtitle: 'Golden-ratio swirls',
    re: -0.39054,
    im: -0.58679,
  },
];

export function createDefaultJuliaState(): JuliaParamState {
  const initial = JULIA_PRESETS[0];
  return {
    re: initial.re,
    im: initial.im,
    animateOrbit: false,
    orbitSpeed: 0.22,
    orbitAngle: Math.atan2(initial.im, initial.re),
    transition: null,
  };
}

/** Advance animated Julia c-orbit or smooth preset morph; returns true if c changed */
export function stepJuliaParam(state: JuliaParamState, dtSeconds: number): boolean {
  const dt = Math.min(dtSeconds, 0.05);

  if (state.transition) {
    const tr = state.transition;
    tr.elapsed += dt;
    const rawT = Math.min(1, tr.elapsed / tr.duration);
    const t = rawT < 0.5 ? 4 * rawT * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 3) / 2;
    state.re = tr.startRe + (tr.endRe - tr.startRe) * t;
    state.im = tr.startIm + (tr.endIm - tr.startIm) * t;
    state.orbitAngle = Math.atan2(state.im, state.re);
    if (rawT >= 1) {
      state.re = tr.endRe;
      state.im = tr.endIm;
      state.transition = null;
    }
    return true;
  }

  if (state.animateOrbit) {
    state.orbitAngle = (state.orbitAngle + state.orbitSpeed * dt) % (Math.PI * 2);
    state.re = JULIA_ORBIT_RADIUS * Math.cos(state.orbitAngle);
    state.im = JULIA_ORBIT_RADIUS * Math.sin(state.orbitAngle);
    return true;
  }

  return false;
}

export const FRACTAL_REGISTRY: readonly FractalDefinition[] = [
  {
    id: 'mandelbrot',
    name: 'Mandelbrot',
    formula: 'zₙ₊₁ = zₙ² + c',
    description: 'The classic quadratic Mandelbrot set in the complex plane.',
    defaultCenter: [-0.55, 0.0],
    defaultZoom: 1.0,
    bailoutRadius: 256.0,
    hasJuliaParam: false,
    kernelWgsl: mandelbrotKernelWgsl,
  },
  {
    id: 'julia',
    name: 'Julia Set',
    formula: 'zₙ₊₁ = zₙ² + c (fixed c)',
    description: 'Quadratic Julia set for a fixed complex parameter c.',
    defaultCenter: [0.0, 0.0],
    defaultZoom: 0.92,
    bailoutRadius: 256.0,
    hasJuliaParam: true,
    kernelWgsl: juliaKernelWgsl,
  },
];

export function getFractalById(id: string): FractalDefinition {
  return FRACTAL_REGISTRY.find((f) => f.id === id) ?? FRACTAL_REGISTRY[0];
}

export function assembleComputeShader(fractal: FractalDefinition): string {
  return `${commonWgsl}\n\n${fractal.kernelWgsl}\n\n${computeEntryWgsl}`;
}
