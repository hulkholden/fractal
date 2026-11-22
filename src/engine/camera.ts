import {
  BigFixed,
  bigAddScaledFloat,
  bigFromDecimal,
  bigFromNumber,
  bigLerp,
  bigRescale,
  bigSubDivPow2,
  bigToDecimal,
  bigToNumber,
  requiredFracBits,
} from './bigfixed';

export type PrecisionMode = 'fp32' | 'df64' | 'perturb';

export interface CameraState {
  centerX: number | string | BigFixed;
  centerY: number | string | BigFixed;
  /** Zoom level (use log10Zoom when > 1e300 or for exact deep bookmarks) */
  zoom?: number;
  log10Zoom?: number;
}

export interface CameraFlight {
  startX: BigFixed;
  startY: BigFixed;
  startLog2Zoom: number;
  endX: BigFixed;
  endY: BigFixed;
  endLog2Zoom: number;
  elapsed: number;
  duration: number;
}

/** Complex plane height at zoom = 1.0 */
export const BASE_COMPLEX_HEIGHT = 3.0;
const LOG2_BASE_HEIGHT = Math.log2(BASE_COMPLEX_HEIGHT);
const LOG2_10 = Math.LOG2E * Math.LN10;

/** Precision tier thresholds in log10(zoom) */
export const DF64_LOG10_THRESHOLD = Math.log10(2.0e4); // ~4.30
export const PERTURB_LOG10_THRESHOLD = Math.log10(2.0e11); // ~11.30

/** Ultra-deep zoom limit: 10^140 (requires ~512 fractional bits in BigFixed) */
export const MAX_LOG10_ZOOM = 140.0;
export const MAX_LOG2_ZOOM = MAX_LOG10_ZOOM * LOG2_10;
export const MIN_LOG2_ZOOM = Math.log2(0.35);

/** Split an IEEE-754 f64 into a (hi, lo) f32 pair for WGSL df64 uniforms */
export function splitDouble(value: number): [number, number] {
  const hi = Math.fround(value);
  const lo = Math.fround(value - hi);
  return [hi, lo];
}

function toBigFixed(v: number | string | BigFixed, bits: number): BigFixed {
  if (typeof v === 'number') return bigFromNumber(v, bits);
  if (typeof v === 'string') return bigFromDecimal(v, bits);
  return bigRescale(v, bits);
}

export class CameraController {
  public centerXBig: BigFixed;
  public centerYBig: BigFixed;
  /** Authoritative zoom representation: zoom = 2^log2Zoom (never overflows f64!) */
  public log2Zoom: number;

  private flight: CameraFlight | null = null;
  private dirty = true;
  private lastMoveTimestamp = performance.now();

  constructor(initial: CameraState) {
    this.log2Zoom =
      initial.log10Zoom !== undefined
        ? initial.log10Zoom * LOG2_10
        : Math.log2(Math.max(0.35, initial.zoom ?? 1.0));
    this.log2Zoom = Math.min(MAX_LOG2_ZOOM, Math.max(MIN_LOG2_ZOOM, this.log2Zoom));
    const bits = requiredFracBits(this.log2Zoom);
    this.centerXBig = toBigFixed(initial.centerX, bits);
    this.centerYBig = toBigFixed(initial.centerY, bits);
  }

  public get log10Zoom(): number {
    return this.log2Zoom / LOG2_10;
  }

  public get zoom(): number {
    return Math.pow(2, this.log2Zoom);
  }

  public get centerX(): number {
    return bigToNumber(this.centerXBig);
  }

  public get centerY(): number {
    return bigToNumber(this.centerYBig);
  }

  public get precisionBits(): number {
    return this.centerXBig.bits;
  }

  public get precisionMode(): PrecisionMode {
    const l10 = this.log10Zoom;
    if (l10 >= PERTURB_LOG10_THRESHOLD) return 'perturb';
    if (l10 >= DF64_LOG10_THRESHOLD) return 'df64';
    return 'fp32';
  }

  public get useDf64(): boolean {
    return this.precisionMode === 'df64';
  }

  public get usePerturb(): boolean {
    return this.precisionMode === 'perturb';
  }

  public get isFlying(): boolean {
    return this.flight !== null;
  }

  /** Only accurate for f64-range zooms; for deep zooms use getScaledHeight() */
  public get complexHeight(): number {
    return BASE_COMPLEX_HEIGHT * Math.pow(2, -this.log2Zoom);
  }

  /**
   * Decompose complexHeight = scaledHeight * 2^scaleExp where scaledHeight in [1, 2).
   * Immune to f64 underflow at arbitrary zoom depths (10^140+).
   */
  public getScaledHeight(): { scaledHeight: number; scaleExp: number } {
    const log2H = LOG2_BASE_HEIGHT - this.log2Zoom;
    const scaleExp = Math.floor(log2H);
    const scaledHeight = Math.pow(2, log2H - scaleExp);
    return { scaledHeight, scaleExp };
  }

  private ensurePrecision(): void {
    const needed = requiredFracBits(this.log2Zoom);
    if (needed !== this.centerXBig.bits) {
      this.centerXBig = bigRescale(this.centerXBig, needed);
      this.centerYBig = bigRescale(this.centerYBig, needed);
    }
  }

  public consumeDirty(): boolean {
    const wasDirty = this.dirty;
    this.dirty = false;
    return wasDirty;
  }

  public markDirty(): void {
    this.dirty = true;
    this.lastMoveTimestamp = performance.now();
  }

  public msSinceLastMove(now: number): number {
    if (this.flight) return 0;
    return now - this.lastMoveTimestamp;
  }

  public cancelFlight(): void {
    this.flight = null;
  }

  /** Pan by delta CSS pixels (exact at any zoom depth up to 10^140) */
  public panByPixels(deltaPxX: number, deltaPxY: number, viewportHeightCss: number): void {
    this.cancelFlight();
    this.ensurePrecision();
    const { scaledHeight, scaleExp } = this.getScaledHeight();
    const stepScaled = scaledHeight / Math.max(1, viewportHeightCss);
    this.centerXBig = bigAddScaledFloat(this.centerXBig, -deltaPxX * stepScaled, scaleExp);
    this.centerYBig = bigAddScaledFloat(this.centerYBig, deltaPxY * stepScaled, scaleExp);
    this.markDirty();
  }

  /** Zoom by factor (>1 zooms in, <1 zooms out), anchored at CSS pixel (anchorX, anchorY) */
  public zoomAtPixel(
    factor: number,
    anchorX: number,
    anchorY: number,
    viewportWidthCss: number,
    viewportHeightCss: number,
  ): void {
    this.cancelFlight();
    const nextLog2 = Math.min(
      MAX_LOG2_ZOOM,
      Math.max(MIN_LOG2_ZOOM, this.log2Zoom + Math.log2(factor)),
    );
    if (Math.abs(nextLog2 - this.log2Zoom) < 1e-12) return;

    const neededBits = requiredFracBits(Math.max(this.log2Zoom, nextLog2));
    this.centerXBig = bigRescale(this.centerXBig, neededBits);
    this.centerYBig = bigRescale(this.centerYBig, neededBits);

    const h = Math.max(1, viewportHeightCss);
    const dxPx = anchorX - viewportWidthCss * 0.5;
    const dyPx = h * 0.5 - anchorY;

    const oldScaled = this.getScaledHeight();
    const oldStep = oldScaled.scaledHeight / h;
    const anchorRe = bigAddScaledFloat(this.centerXBig, dxPx * oldStep, oldScaled.scaleExp);
    const anchorIm = bigAddScaledFloat(this.centerYBig, dyPx * oldStep, oldScaled.scaleExp);

    this.log2Zoom = nextLog2;
    const newScaled = this.getScaledHeight();
    const newStep = newScaled.scaledHeight / h;

    this.centerXBig = bigAddScaledFloat(anchorRe, -dxPx * newStep, newScaled.scaleExp);
    this.centerYBig = bigAddScaledFloat(anchorIm, -dyPx * newStep, newScaled.scaleExp);
    this.ensurePrecision();
    this.markDirty();
  }

  /** Compute pixel complex coordinate as BigFixed (e.g. for Alt+Click or double-click) */
  public pixelToComplexBig(
    pxCss: number,
    pyCss: number,
    viewportWidthCss: number,
    viewportHeightCss: number,
  ): [BigFixed, BigFixed] {
    this.ensurePrecision();
    const h = Math.max(1, viewportHeightCss);
    const { scaledHeight, scaleExp } = this.getScaledHeight();
    const step = scaledHeight / h;
    const dxPx = pxCss - viewportWidthCss * 0.5;
    const dyPx = h * 0.5 - pyCss;
    return [
      bigAddScaledFloat(this.centerXBig, dxPx * step, scaleExp),
      bigAddScaledFloat(this.centerYBig, dyPx * step, scaleExp),
    ];
  }

  /** Smoothly fly to a target location (supports arbitrary-precision decimal strings & log10Zoom) */
  public flyTo(
    targetX: number | string | BigFixed,
    targetY: number | string | BigFixed,
    targetZoom: number,
    duration = 1.15,
    targetLog10Zoom?: number,
  ): void {
    const endLog2 = Math.min(
      MAX_LOG2_ZOOM,
      Math.max(
        MIN_LOG2_ZOOM,
        targetLog10Zoom !== undefined
          ? targetLog10Zoom * LOG2_10
          : Math.log2(Math.max(0.35, targetZoom)),
      ),
    );
    const bits = requiredFracBits(Math.max(this.log2Zoom, endLog2));
    this.flight = {
      startX: bigRescale(this.centerXBig, bits),
      startY: bigRescale(this.centerYBig, bits),
      startLog2Zoom: this.log2Zoom,
      endX: toBigFixed(targetX, bits),
      endY: toBigFixed(targetY, bits),
      endLog2Zoom: endLog2,
      elapsed: 0,
      duration: Math.max(0.2, duration),
    };
    this.markDirty();
  }

  /** Advance active camera flight; returns true if camera moved this frame */
  public update(dtSeconds: number): boolean {
    if (!this.flight) return false;

    const f = this.flight;
    f.elapsed += Math.min(dtSeconds, 0.05);
    const rawT = Math.min(1, f.elapsed / f.duration);
    // Quintic ease-in-out
    const t =
      rawT < 0.5
        ? 16 * rawT * rawT * rawT * rawT * rawT
        : 1 - Math.pow(-2 * rawT + 2, 5) / 2;

    const log2Delta = f.endLog2Zoom - f.startLog2Zoom;
    let posT = t;
    if (Math.abs(log2Delta) > 0.75) {
      // Exact exponential camera path interpolation (clamped safely for extreme zoom flights)
      const lnDelta = log2Delta * Math.LN2;
      if (lnDelta > 40) {
        // Deep zoom-in: move onto target quickly in the first phase, then zoom straight down
        posT = 1 - Math.exp(-lnDelta * t);
      } else if (lnDelta < -40) {
        posT = Math.exp(lnDelta * (1 - t));
      } else {
        const num = Math.exp(-lnDelta * t) - 1;
        const den = Math.exp(-lnDelta) - 1;
        posT = num / den;
      }
    }

    this.centerXBig = bigLerp(f.startX, f.endX, posT);
    this.centerYBig = bigLerp(f.startY, f.endY, posT);
    this.log2Zoom = f.startLog2Zoom + log2Delta * t;
    this.markDirty();

    if (rawT >= 1) {
      this.centerXBig = f.endX;
      this.centerYBig = f.endY;
      this.log2Zoom = f.endLog2Zoom;
      this.ensurePrecision();
      this.flight = null;
    }

    return true;
  }

  public formatZoom(): string {
    const l10 = this.log10Zoom;
    if (l10 < 3) {
      return `${Math.pow(10, l10).toFixed(2)}×`;
    }
    const exp = Math.floor(l10);
    const mant = Math.pow(10, l10 - exp);
    return `${mant.toFixed(2)}×10^${exp}`;
  }

  public formatCenterX(maxDisplayDigits = 18): string {
    const digits = Math.min(maxDisplayDigits, Math.max(6, Math.ceil(Math.max(0, this.log10Zoom)) + 4));
    return bigToDecimal(this.centerXBig, digits);
  }

  public formatCenterY(maxDisplayDigits = 18): string {
    const digits = Math.min(maxDisplayDigits, Math.max(6, Math.ceil(Math.max(0, this.log10Zoom)) + 4));
    return bigToDecimal(this.centerYBig, digits);
  }

  /** Offset from a reference point divided by 2^scaleExp */
  public offsetFromReference(refX: BigFixed, refY: BigFixed, scaleExp: number): [number, number] {
    return [
      bigSubDivPow2(this.centerXBig, refX, scaleExp),
      bigSubDivPow2(this.centerYBig, refY, scaleExp),
    ];
  }
}
