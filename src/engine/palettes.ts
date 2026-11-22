export interface PaletteDefinition {
  id: string;
  name: string;
  /** 8 periodic hex stops (loops seamlessly back to stop 0) */
  hexStops: readonly [string, string, string, string, string, string, string, string];
  /** Precomputed Oklab stops as flat Float32Array (8 * 4 floats: L, a, b, 0) */
  oklabData: Float32Array;
}

function srgbChannelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function hexToLinearRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  return [srgbChannelToLinear(r), srgbChannelToLinear(g), srgbChannelToLinear(b)];
}

/** Convert linear sRGB to Oklab (Björn Ottosson) */
export function linearSrgbToOklab(r: number, g: number, b: number): [number, number, number] {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(Math.max(0, l));
  const m_ = Math.cbrt(Math.max(0, m));
  const s_ = Math.cbrt(Math.max(0, s));

  const L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  const a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
  const bLab = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
  return [L, a, bLab];
}

function oklabToSrgbHex(L: number, a: number, b: number): string {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  const rLin = Math.min(1, Math.max(0, 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s));
  const gLin = Math.min(1, Math.max(0, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s));
  const bLin = Math.min(1, Math.max(0, -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s));

  const enc = (c: number) =>
    Math.round(
      Math.min(1, Math.max(0, c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)) *
        255,
    )
      .toString(16)
      .padStart(2, '0');

  return `#${enc(rLin)}${enc(gLin)}${enc(bLin)}`;
}

function createPalette(
  id: string,
  name: string,
  hexStops: readonly [string, string, string, string, string, string, string, string],
): PaletteDefinition {
  const oklabData = new Float32Array(8 * 4);
  for (let i = 0; i < 8; i++) {
    const [r, g, b] = hexToLinearRgb(hexStops[i]);
    const [L, a, bLab] = linearSrgbToOklab(r, g, b);
    oklabData[i * 4 + 0] = L;
    oklabData[i * 4 + 1] = a;
    oklabData[i * 4 + 2] = bLab;
    oklabData[i * 4 + 3] = 0;
  }
  return { id, name, hexStops, oklabData };
}

export const PALETTES: readonly PaletteDefinition[] = [
  createPalette('ultra-classic', 'Ultra Classic', [
    '#020722',
    '#0b2e8a',
    '#1f7ee0',
    '#74d7fa',
    '#fff4d2',
    '#f69c2a',
    '#a33617',
    '#1c0e40',
  ]),
  createPalette('prism-dream', 'Prism Dream', [
    '#14062e',
    '#5e149c',
    '#de2892',
    '#ff884d',
    '#ffe785',
    '#43edb8',
    '#1785f0',
    '#221669',
  ]),
  createPalette('bioluminescent', 'Bioluminescent', [
    '#020a14',
    '#05334f',
    '#0b7a8c',
    '#24d4c6',
    '#d9fff6',
    '#4db8ff',
    '#2a46d4',
    '#0e1240',
  ]),
  createPalette('aurora', 'Aurora Borealis', [
    '#040917',
    '#0b4d47',
    '#15c989',
    '#a8f569',
    '#ecffbd',
    '#8d52ff',
    '#2464f0',
    '#0a1b45',
  ]),
  createPalette('solar-corona', 'Solar Corona', [
    '#090208',
    '#4d0724',
    '#b5192e',
    '#f75c1e',
    '#ffc847',
    '#fff6cc',
    '#d9485b',
    '#2e0c38',
  ]),
  createPalette('iridescent-opal', 'Iridescent Opal', [
    '#0d1226',
    '#3949a8',
    '#8785ff',
    '#ff91bd',
    '#fff0d1',
    '#7af0dc',
    '#3498db',
    '#1e2052',
  ]),
];

export interface GradientState {
  /** Active palette index when livingMorph is off, or starting palette when on */
  paletteIndex: number;
  /** Continuously advances along the gradient contours (cycles/sec) */
  flowSpeed: number;
  /** Whether the color cycling animation is currently playing */
  playing: boolean;
  /** Ever-changing palette morphing & subtle Oklab hue breathing */
  livingMorph: boolean;
  /** Multiplier for gradient contour density */
  density: number;
  /** Current accumulated phase [0, 1) */
  phase: number;
  /** Current morph position across the palette ring [0, PALETTES.length) */
  morphCursor: number;
  /** Slow Oklab hue drift angle in radians */
  hueDrift: number;
}

export function createDefaultGradientState(): GradientState {
  return {
    paletteIndex: 0,
    flowSpeed: 0.085,
    playing: true,
    livingMorph: true,
    density: 1.0,
    phase: 0.0,
    morphCursor: 0.0,
    hueDrift: 0.0,
  };
}

export interface EvaluatedGradient {
  paletteA: PaletteDefinition;
  paletteB: PaletteDefinition;
  morphBlend: number;
  hueDrift: number;
}

export function stepGradient(state: GradientState, dtSeconds: number): EvaluatedGradient {
  const clampedDt = Math.min(dtSeconds, 0.1);

  if (state.playing) {
    state.phase = (state.phase + state.flowSpeed * clampedDt) % 1.0;
    if (state.phase < 0) state.phase += 1.0;

    if (state.livingMorph) {
      // Smoothly glide through one full palette every ~18 seconds
      state.morphCursor = (state.morphCursor + clampedDt / 18.0) % PALETTES.length;
      // Gentle perceptual hue breathing (±0.28 rad)
      state.hueDrift += clampedDt * 0.11;
    }
  }

  if (!state.livingMorph) {
    const p = PALETTES[state.paletteIndex % PALETTES.length];
    return {
      paletteA: p,
      paletteB: p,
      morphBlend: 0,
      hueDrift: 0,
    };
  }

  const count = PALETTES.length;
  const idxA = Math.floor(state.morphCursor) % count;
  const idxB = (idxA + 1) % count;
  const rawFrac = state.morphCursor - Math.floor(state.morphCursor);
  // Smoothstep blend so each palette lingers nicely before gliding into the next
  const smoothBlend = rawFrac * rawFrac * (3 - 2 * rawFrac);
  const hueOffset = Math.sin(state.hueDrift) * 0.26;

  return {
    paletteA: PALETTES[idxA],
    paletteB: PALETTES[idxB],
    morphBlend: smoothBlend,
    hueDrift: hueOffset,
  };
}

/** Generate a CSS linear-gradient string reflecting the current live animated palette */
export function buildLivePreviewCss(state: GradientState, evaluated: EvaluatedGradient): string {
  const stops: string[] = [];
  const sampleCount = 16;
  const ch = Math.cos(evaluated.hueDrift);
  const sh = Math.sin(evaluated.hueDrift);

  for (let i = 0; i <= sampleCount; i++) {
    const u = ((i / sampleCount - state.phase) % 1 + 1) % 1;
    const scaled = u * 8;
    const idx0 = Math.floor(scaled) & 7;
    const idx1 = (idx0 + 1) & 7;
    const frac = scaled - Math.floor(scaled);

    const aA = evaluated.paletteA.oklabData;
    const aB = evaluated.paletteB.oklabData;
    const m = evaluated.morphBlend;

    const L0 = aA[idx0 * 4] * (1 - m) + aB[idx0 * 4] * m;
    const a0 = aA[idx0 * 4 + 1] * (1 - m) + aB[idx0 * 4 + 1] * m;
    const b0 = aA[idx0 * 4 + 2] * (1 - m) + aB[idx0 * 4 + 2] * m;

    const L1 = aA[idx1 * 4] * (1 - m) + aB[idx1 * 4] * m;
    const a1 = aA[idx1 * 4 + 1] * (1 - m) + aB[idx1 * 4 + 1] * m;
    const b1 = aA[idx1 * 4 + 2] * (1 - m) + aB[idx1 * 4 + 2] * m;

    const L = L0 + (L1 - L0) * frac;
    const aRaw = a0 + (a1 - a0) * frac;
    const bRaw = b0 + (b1 - b0) * frac;

    const aRot = aRaw * ch - bRaw * sh;
    const bRot = aRaw * sh + bRaw * ch;

    stops.push(`${oklabToSrgbHex(L, aRot, bRot)} ${Math.round((i / sampleCount) * 100)}%`);
  }

  return `linear-gradient(90deg, ${stops.join(', ')})`;
}
