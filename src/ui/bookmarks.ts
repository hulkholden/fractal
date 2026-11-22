export interface FractalBookmark {
  id: string;
  name: string;
  subtitle: string;
  fractalId: 'mandelbrot' | 'julia';
  centerX: number | string;
  centerY: number | string;
  zoom: number;
  /** Optional log10(zoom) for ultra-deep perturbation destinations (> 10^12) */
  log10Zoom?: number;
  juliaC?: [number, number];
}

export const BOOKMARKS: readonly FractalBookmark[] = [
  // Mandelbrot destinations (spanning FP32 -> DF64 -> Perturbation 10^44!)
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
    id: 'mini-brot',
    name: 'Deep Mini-Brot',
    subtitle: 'DF64 star (6.5×10⁶)',
    fractalId: 'mandelbrot',
    centerX: -1.7497591451303665,
    centerY: -0.000000003685138,
    zoom: 6.5e6,
  },
  {
    id: 'spiral-abyss-18',
    name: 'Spiral Abyss 10¹⁸',
    subtitle: 'Perturbation (past f64)',
    fractalId: 'mandelbrot',
    centerX: '-0.743708650620633856124999',
    centerY: '0.131706613679699927874999',
    zoom: 1e18,
    log10Zoom: 18,
  },
  {
    id: 'quantum-vortex-32',
    name: 'Quantum Vortex',
    subtitle: 'Perturbation (10³²×)',
    fractalId: 'mandelbrot',
    centerX: '-0.74370865062063385623368740240911374999',
    centerY: '0.13170661367969992778998288491324624999',
    zoom: 1e32,
    log10Zoom: 32,
  },
  {
    id: 'subfloat-44',
    name: 'Sub-Float 10⁴⁴',
    subtitle: 'Past f32 underflow!',
    fractalId: 'mandelbrot',
    centerX: '-0.74370865062063385623368740240911265915579877624999',
    centerY: '0.13170661367969992778998288491324625104322497999999',
    zoom: 1e44,
    log10Zoom: 44,
  },

  // Julia destinations (FP32 -> DF64 -> Perturbation 10^24!)
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
    centerX: '-0.499313366333913602938447748723831600660883361',
    centerY: '0.056543824893781364379145882497764206633629132',
    zoom: 18.0,
    juliaC: [-0.74543, 0.11301],
  },
  {
    id: 'julia-deep-spiral',
    name: 'Deep Julia Eye',
    subtitle: 'DF64 spiral core (10⁵×)',
    fractalId: 'julia',
    centerX: '-0.499313366333913602938447748723831600660883361',
    centerY: '0.056543824893781364379145882497764206633629132',
    zoom: 1.8e5,
    juliaC: [-0.74543, 0.11301],
  },
  {
    id: 'julia-ultra-24',
    name: 'Ultra Julia 10²⁴',
    subtitle: 'Perturbation dragon eye',
    fractalId: 'julia',
    centerX: '-0.5275031186435346107897464024449153375667',
    centerY: '0.0759121783522878670718141948263480463664',
    zoom: 1e24,
    log10Zoom: 24,
    juliaC: [-0.8, 0.156],
  },
];
