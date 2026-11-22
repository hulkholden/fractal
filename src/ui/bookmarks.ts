export interface FractalBookmark {
  id: string;
  name: string;
  subtitle: string;
  fractalId: 'mandelbrot' | 'julia';
  centerX: number;
  centerY: number;
  zoom: number;
  juliaC?: [number, number];
}

export const BOOKMARKS: readonly FractalBookmark[] = [
  // Mandelbrot destinations
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

  // Julia destinations
  {
    id: 'julia-seahorse',
    name: 'Seahorse Overview',
    subtitle: 'Full filigree web',
    fractalId: 'julia',
    centerX: 0.0,
    centerY: 0.0,
    zoom: 0.92,
    juliaC: [-0.74543, 0.11301],
  },
  {
    id: 'julia-dragon',
    name: 'Golden Dragon',
    subtitle: 'Sweeping spiral spine',
    fractalId: 'julia',
    centerX: 0.0,
    centerY: 0.0,
    zoom: 0.98,
    juliaC: [-0.8, 0.156],
  },
  {
    id: 'julia-galaxy',
    name: 'Galaxy Vortex',
    subtitle: 'Twin cyclone arms',
    fractalId: 'julia',
    centerX: 0.0,
    centerY: 0.0,
    zoom: 1.05,
    juliaC: [-0.4, 0.6],
  },
  {
    id: 'julia-spiral-core',
    name: 'Spiral Heart',
    subtitle: 'Logarithmic spiral eye',
    fractalId: 'julia',
    centerX: -0.49931336633391377,
    centerY: 0.05654382489378136,
    zoom: 18.0,
    juliaC: [-0.74543, 0.11301],
  },
  {
    id: 'julia-web-zoom',
    name: 'Luminous Lace',
    subtitle: 'Infinite petal weave',
    fractalId: 'julia',
    centerX: 0.0,
    centerY: 0.0,
    zoom: 1.12,
    juliaC: [0.285, 0.01],
  },
  {
    id: 'julia-deep-spiral',
    name: 'Deep Julia Eye',
    subtitle: 'DF64 spiral core (10⁵×)',
    fractalId: 'julia',
    centerX: -0.49931336633391377,
    centerY: 0.05654382489378136,
    zoom: 1.8e5,
    juliaC: [-0.74543, 0.11301],
  },
];
