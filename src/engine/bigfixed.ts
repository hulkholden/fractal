/**
 * Arbitrary-precision binary fixed-point arithmetic using native JS BigInt.
 * Represents real numbers x in [-2^30, 2^30] as (value / 2^fracBits).
 * Powers ultra-deep zoom coordinates and reference orbits past 10^140.
 */

export interface BigFixed {
  val: bigint;
  bits: number;
}

const TEN = 10n;

/** Required fractional bits for a given log2(zoom) */
export function requiredFracBits(log2Zoom: number): number {
  const needed = Math.max(96, Math.ceil(log2Zoom) + 72);
  return Math.ceil(needed / 64) * 64;
}

export function bigFromNumber(x: number, bits = 128): BigFixed {
  if (!Number.isFinite(x) || x === 0) {
    return { val: 0n, bits };
  }
  const sign = x < 0 ? -1n : 1n;
  const ax = Math.abs(x);
  const exp = Math.floor(Math.log2(ax));
  // Extract exact 53-bit integer mantissa
  const mant53 = Math.round(ax * Math.pow(2, 52 - exp));
  const shift = bits + exp - 52;
  const val = shift >= 0 ? BigInt(mant53) << BigInt(shift) : BigInt(mant53) >> BigInt(-shift);
  return { val: sign * val, bits };
}

/** Parse an arbitrary-precision decimal string (e.g. "-0.74364388703715870475...") exactly */
export function bigFromDecimal(str: string, bits = 256): BigFixed {
  const s = str.trim();
  if (!s) return { val: 0n, bits };

  const negative = s.startsWith('-');
  const unsigned = negative || s.startsWith('+') ? s.slice(1) : s;

  // Handle scientific notation (e.g. "1.23e-45")
  const [mantPart, expPart] = unsigned.toLowerCase().split('e');
  const exp10 = expPart ? parseInt(expPart, 10) : 0;

  const [intStr, fracStr = ''] = mantPart.split('.');
  const digits = (intStr + fracStr).replace(/^0+/, '') || '0';
  const totalExp10 = exp10 - fracStr.length;

  let num = BigInt(digits) << BigInt(bits);
  if (totalExp10 >= 0) {
    num = num * TEN ** BigInt(totalExp10);
  } else {
    const den = TEN ** BigInt(-totalExp10);
    // Round to nearest
    num = (num + (den >> 1n)) / den;
  }

  return { val: negative ? -num : num, bits };
}

/** Rescale a BigFixed to a new fractional bit precision */
export function bigRescale(a: BigFixed, targetBits: number): BigFixed {
  if (a.bits === targetBits) return a;
  if (targetBits > a.bits) {
    return { val: a.val << BigInt(targetBits - a.bits), bits: targetBits };
  }
  return { val: a.val >> BigInt(a.bits - targetBits), bits: targetBits };
}

export function bigAdd(a: BigFixed, b: BigFixed): BigFixed {
  const bits = Math.max(a.bits, b.bits);
  const aa = bigRescale(a, bits);
  const bb = bigRescale(b, bits);
  return { val: aa.val + bb.val, bits };
}

export function bigSub(a: BigFixed, b: BigFixed): BigFixed {
  const bits = Math.max(a.bits, b.bits);
  const aa = bigRescale(a, bits);
  const bb = bigRescale(b, bits);
  return { val: aa.val - bb.val, bits };
}

/** Add a float scaled by 2^exp2 (i.e. value * 2^exp2) without f64 underflow */
export function bigAddScaledFloat(a: BigFixed, value: number, exp2: number): BigFixed {
  if (!Number.isFinite(value) || value === 0) return a;
  const sign = value < 0 ? -1n : 1n;
  const ax = Math.abs(value);
  const vExp = Math.floor(Math.log2(ax));
  const mant53 = Math.round(ax * Math.pow(2, 52 - vExp));
  const shift = a.bits + exp2 + vExp - 52;
  const delta =
    shift >= 0 ? BigInt(mant53) << BigInt(shift) : BigInt(mant53) >> BigInt(-shift);
  return { val: a.val + sign * delta, bits: a.bits };
}

/** Compute (a - b) / 2^exp2 as an IEEE-754 f64 without underflow even when |a - b| ~ 10^-150 */
export function bigSubDivPow2(a: BigFixed, b: BigFixed, exp2: number): number {
  const bits = Math.max(a.bits, b.bits);
  const diff = bigRescale(a, bits).val - bigRescale(b, bits).val;
  if (diff === 0n) return 0;

  const neg = diff < 0n;
  const absDiff = neg ? -diff : diff;

  // We want absDiff * 2^(-bits - exp2)
  // Extract top ~54 bits safely via bit length
  const bitLen = absDiff.toString(2).length;
  const shiftToTop54 = bitLen - 54;
  const top54 =
    shiftToTop54 >= 0 ? Number(absDiff >> BigInt(shiftToTop54)) : Number(absDiff) * Math.pow(2, -shiftToTop54);
  const finalExp = shiftToTop54 - bits - exp2;
  const result = top54 * Math.pow(2, finalExp);
  return neg ? -result : result;
}

export function bigToNumber(a: BigFixed): number {
  return bigSubDivPow2(a, { val: 0n, bits: a.bits }, 0);
}

/** Format a BigFixed as a signed decimal string with `fracDigits` digits after the decimal point */
export function bigToDecimal(a: BigFixed, fracDigits: number): string {
  const digits = Math.max(1, Math.min(160, fracDigits));
  const neg = a.val < 0n;
  const absVal = neg ? -a.val : a.val;
  const scaled = (absVal * TEN ** BigInt(digits) + (1n << BigInt(a.bits - 1))) >> BigInt(a.bits);
  const str = scaled.toString().padStart(digits + 1, '0');
  const intPart = str.slice(0, str.length - digits);
  const fracPart = str.slice(str.length - digits);
  return `${neg ? '-' : '+'}${intPart}.${fracPart}`;
}

/** Linear interpolation a + (b - a) * t where t in [0, 1] */
export function bigLerp(a: BigFixed, b: BigFixed, t: number): BigFixed {
  const bits = Math.max(a.bits, b.bits);
  const aa = bigRescale(a, bits).val;
  const bb = bigRescale(b, bits).val;
  const diff = bb - aa;
  if (diff === 0n || t === 0) return { val: aa, bits };
  if (t === 1) return { val: bb, bits };
  const tScaled = BigInt(Math.round(t * 2 ** 52));
  return { val: aa + ((diff * tScaled) >> 52n), bits };
}

export interface ReferenceOrbit {
  /** Reference complex point in BigFixed */
  refX: BigFixed;
  refY: BigFixed;
  /** Interleaved (Z_n.re, Z_n.im) as f32 pairs; length = (escapedAt + 1) * 2 */
  orbitData: Float32Array;
  /** Number of valid steps in orbitData (indices 0..length are valid) */
  length: number;
  /** Whether the reference point escaped before reaching requested maxIter */
  escaped: boolean;
  /** Saved high-precision state so we can extend the orbit incrementally without recomputing */
  lastZx: bigint;
  lastZy: bigint;
  bits: number;
  fractalId: 'mandelbrot' | 'julia';
  juliaRe?: BigFixed;
  juliaIm?: BigFixed;
}

const BAILOUT_FIXED_CHECK = 65536; // 256^2

/** Fast BigInt fixed-point -> f64 with full sub-1e-15 dynamic range preservation */
function bigIntToNumber(v: bigint, bits: number, toFloatShift: bigint, toFloatScale: number): number {
  if (v === 0n) return 0;
  const shifted = v >> toFloatShift;
  if (shifted !== 0n && shifted !== -1n) {
    return Number(shifted) * toFloatScale;
  }
  return bigSubDivPow2({ val: v, bits }, { val: 0n, bits }, 0);
}

function runHighPrecisionOrbit(
  refX: BigFixed,
  refY: BigFixed,
  maxIter: number,
  fractalId: 'mandelbrot' | 'julia',
  juliaRe?: BigFixed,
  juliaIm?: BigFixed,
): ReferenceOrbit {
  const bits = refX.bits;
  const bShift = BigInt(bits);
  const bShiftM1 = BigInt(bits - 1);
  // Shift to convert BigInt fixed-point -> f64 (keep top 53 bits around integer range)
  const toFloatShift = BigInt(Math.max(0, bits - 48));
  const toFloatScale = Math.pow(2, -(bits - Math.max(0, bits - 48)));

  const cx = fractalId === 'julia' && juliaRe ? bigRescale(juliaRe, bits).val : refX.val;
  const cy = fractalId === 'julia' && juliaIm ? bigRescale(juliaIm, bits).val : refY.val;

  let zx = fractalId === 'julia' ? refX.val : 0n;
  let zy = fractalId === 'julia' ? refY.val : 0n;

  const data = new Float32Array((maxIter + 1) * 2);
  let escaped = false;
  let len = 0;

  for (let i = 0; i <= maxIter; i++) {
    const fx = bigIntToNumber(zx, bits, toFloatShift, toFloatScale);
    const fy = bigIntToNumber(zy, bits, toFloatShift, toFloatScale);
    data[i * 2] = fx;
    data[i * 2 + 1] = fy;
    len = i;

    if (fx * fx + fy * fy > BAILOUT_FIXED_CHECK) {
      escaped = true;
      break;
    }

    if (i === maxIter) break;

    const zx2 = (zx * zx) >> bShift;
    const zy2 = (zy * zy) >> bShift;
    const nextZy = ((zx * zy) >> bShiftM1) + cy;
    zx = zx2 - zy2 + cx;
    zy = nextZy;
  }

  return {
    refX,
    refY,
    orbitData: data.subarray(0, (len + 1) * 2),
    length: len,
    escaped,
    lastZx: zx,
    lastZy: zy,
    bits,
    fractalId,
    juliaRe,
    juliaIm,
  };
}

/** Extend an existing non-escaped reference orbit to a deeper iteration target */
export function extendReferenceOrbit(
  existing: ReferenceOrbit,
  newMaxIter: number,
): ReferenceOrbit {
  if (existing.escaped || existing.length >= newMaxIter) {
    return existing;
  }

  const bits = existing.bits;
  const bShift = BigInt(bits);
  const bShiftM1 = BigInt(bits - 1);
  const toFloatShift = BigInt(Math.max(0, bits - 48));
  const toFloatScale = Math.pow(2, -(bits - Math.max(0, bits - 48)));

  const cx =
    existing.fractalId === 'julia' && existing.juliaRe
      ? bigRescale(existing.juliaRe, bits).val
      : existing.refX.val;
  const cy =
    existing.fractalId === 'julia' && existing.juliaIm
      ? bigRescale(existing.juliaIm, bits).val
      : existing.refY.val;

  const data = new Float32Array((newMaxIter + 1) * 2);
  data.set(existing.orbitData);

  let zx = existing.lastZx;
  let zy = existing.lastZy;
  let escaped = false;
  let len = existing.length;

  for (let i = existing.length; i < newMaxIter; i++) {
    const zx2 = (zx * zx) >> bShift;
    const zy2 = (zy * zy) >> bShift;
    const nextZy = ((zx * zy) >> bShiftM1) + cy;
    zx = zx2 - zy2 + cx;
    zy = nextZy;

    const nextIdx = i + 1;
    const fx = bigIntToNumber(zx, bits, toFloatShift, toFloatScale);
    const fy = bigIntToNumber(zy, bits, toFloatShift, toFloatScale);
    data[nextIdx * 2] = fx;
    data[nextIdx * 2 + 1] = fy;
    len = nextIdx;

    if (fx * fx + fy * fy > BAILOUT_FIXED_CHECK) {
      escaped = true;
      break;
    }
  }

  return {
    ...existing,
    orbitData: data.subarray(0, (len + 1) * 2),
    length: len,
    escaped,
    lastZx: zx,
    lastZy: zy,
  };
}

/**
 * Compute a high-quality reference orbit near the viewport center.
 * Tests center first; if center escapes early, probes surrounding viewport points
 * to find a deeper/bounded reference orbit so Zhuoran rebasing stays rare and fast.
 */
export function computeBestReferenceOrbit(
  centerX: BigFixed,
  centerY: BigFixed,
  scaleExp: number,
  maxIter: number,
  fractalId: 'mandelbrot' | 'julia',
  juliaC?: [BigFixed, BigFixed],
): ReferenceOrbit {
  const bits = Math.max(centerX.bits, centerY.bits);
  const cx0 = bigRescale(centerX, bits);
  const cy0 = bigRescale(centerY, bits);

  let best = runHighPrecisionOrbit(cx0, cy0, maxIter, fractalId, juliaC?.[0], juliaC?.[1]);
  if (!best.escaped || best.length >= maxIter) {
    return best;
  }

  // Probe a small ring around viewport center if center escaped early
  const offsets: readonly [number, number][] = [
    [-0.28, 0.0],
    [0.28, 0.0],
    [0.0, -0.25],
    [0.0, 0.25],
    [-0.22, -0.2],
    [0.22, -0.2],
    [-0.22, 0.2],
    [0.22, 0.2],
  ];

  const probeLimit = Math.min(maxIter, 1500);
  for (const [ox, oy] of offsets) {
    const px = bigAddScaledFloat(cx0, ox, scaleExp);
    const py = bigAddScaledFloat(cy0, oy, scaleExp);
    const candidate = runHighPrecisionOrbit(
      px,
      py,
      probeLimit,
      fractalId,
      juliaC?.[0],
      juliaC?.[1],
    );
    if (!candidate.escaped || candidate.length > best.length) {
      best = candidate;
      if (!candidate.escaped) break;
    }
  }

  if (!best.escaped && best.length < maxIter) {
    best = extendReferenceOrbit(best, maxIter);
  }

  return best;
}
