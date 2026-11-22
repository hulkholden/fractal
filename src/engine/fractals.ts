import commonWgsl from './shaders/common.wgsl?raw';
import computeEntryWgsl from './shaders/compute_entry.wgsl?raw';
import mandelbrotKernelWgsl from './shaders/mandelbrot_kernel.wgsl?raw';

export interface FractalDefinition {
  id: string;
  name: string;
  formula: string;
  description: string;
  defaultCenter: [number, number];
  defaultZoom: number;
  bailoutRadius: number;
  kernelWgsl: string;
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
    kernelWgsl: mandelbrotKernelWgsl,
  },
];

export function getFractalById(id: string): FractalDefinition {
  return FRACTAL_REGISTRY.find((f) => f.id === id) ?? FRACTAL_REGISTRY[0];
}

export function assembleComputeShader(fractal: FractalDefinition): string {
  return `${commonWgsl}\n\n${fractal.kernelWgsl}\n\n${computeEntryWgsl}`;
}
