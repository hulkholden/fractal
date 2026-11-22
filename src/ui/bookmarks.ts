export interface FractalBookmark {
  id: string;
  name: string;
  subtitle: string;
  fractalId: string;
  centerX: number;
  centerY: number;
  zoom: number;
}

export const BOOKMARKS: readonly FractalBookmark[] = [
  {
    id: 'overview',
    name: 'Full Mandelbrot',
    subtitle: 'Classic cardioid & bulbs',
    fractalId: 'mandelbrot',
    centerX: -0.55,
    centerY: 0.0,
    zoom: 1.0,
  },
  {
    id: 'seahorse-valley',
    name: 'Seahorse Valley',
    subtitle: 'Double-spiral curls',
    fractalId: 'mandelbrot',
    centerX: -0.7436438870371587,
    centerY: 0.13182590420531197,
    zoom: 420.0,
  },
  {
    id: 'elephant-valley',
    name: 'Elephant Valley',
    subtitle: 'Ornamental trunks',
    fractalId: 'mandelbrot',
    centerX: 0.27485,
    centerY: 0.00595,
    zoom: 160.0,
  },
  {
    id: 'triple-spiral',
    name: 'Triple Spiral',
    subtitle: 'Nested vortex arms',
    fractalId: 'mandelbrot',
    centerX: -0.088,
    centerY: 0.654,
    zoom: 42.0,
  },
  {
    id: 'mini-brot',
    name: 'Deep Mini-Brot',
    subtitle: 'DF64 precision showcase',
    fractalId: 'mandelbrot',
    centerX: -1.7497591451303665,
    centerY: -0.000000003685138,
    zoom: 6.5e6,
  },
  {
    id: 'seahorse-deep',
    name: 'Filament Nexus',
    subtitle: 'Deep tendrils (10⁸×)',
    fractalId: 'mandelbrot',
    centerX: -0.743643887037151,
    centerY: 0.13182590420533,
    zoom: 2.2e8,
  },
];
