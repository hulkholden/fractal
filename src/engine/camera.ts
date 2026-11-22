export interface CameraState {
  centerX: number;
  centerY: number;
  /** Zoom level: 1.0 means complex plane vertical span = 3.0 */
  zoom: number;
}

export interface CameraFlight {
  startX: number;
  startY: number;
  startLogZoom: number;
  endX: number;
  endY: number;
  endLogZoom: number;
  elapsed: number;
  duration: number;
}

/** Complex plane height at zoom = 1.0 */
export const BASE_COMPLEX_HEIGHT = 3.0;

/** Zoom threshold above which WGSL df64 emulated double-precision activates automatically */
export const DF64_ZOOM_THRESHOLD = 2.0e4;

/** Maximum practical zoom with df64 (~48-bit mantissa) */
export const MAX_ZOOM = 5.0e12;
export const MIN_ZOOM = 0.35;

/** Split an IEEE-754 f64 into a (hi, lo) f32 pair for WGSL df64 uniforms */
export function splitDouble(value: number): [number, number] {
  const hi = Math.fround(value);
  const lo = Math.fround(value - hi);
  return [hi, lo];
}

export class CameraController {
  public centerX: number;
  public centerY: number;
  public zoom: number;

  private flight: CameraFlight | null = null;
  private dirty = true;
  private lastMoveTimestamp = performance.now();

  constructor(initial: CameraState) {
    this.centerX = initial.centerX;
    this.centerY = initial.centerY;
    this.zoom = initial.zoom;
  }

  public get isFlying(): boolean {
    return this.flight !== null;
  }

  public get useDf64(): boolean {
    return this.zoom >= DF64_ZOOM_THRESHOLD;
  }

  public get complexHeight(): number {
    return BASE_COMPLEX_HEIGHT / this.zoom;
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

  /** Pan by delta CSS pixels */
  public panByPixels(deltaPxX: number, deltaPxY: number, viewportHeightCss: number): void {
    this.cancelFlight();
    const complexPerCssPx = this.complexHeight / Math.max(1, viewportHeightCss);
    this.centerX -= deltaPxX * complexPerCssPx;
    this.centerY += deltaPxY * complexPerCssPx;
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
    const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
    const appliedFactor = nextZoom / this.zoom;
    if (Math.abs(appliedFactor - 1) < 1e-9) return;

    const oldStep = this.complexHeight / Math.max(1, viewportHeightCss);
    const dxPx = anchorX - viewportWidthCss * 0.5;
    const dyPx = viewportHeightCss * 0.5 - anchorY;

    // Complex point currently under the cursor
    const anchorRe = this.centerX + dxPx * oldStep;
    const anchorIm = this.centerY + dyPx * oldStep;

    this.zoom = nextZoom;
    const newStep = this.complexHeight / Math.max(1, viewportHeightCss);

    // Keep anchorRe/anchorIm fixed at the same cursor pixel
    this.centerX = anchorRe - dxPx * newStep;
    this.centerY = anchorIm - dyPx * newStep;
    this.markDirty();
  }

  /** Smoothly fly to a target location with logarithmic zoom easing */
  public flyTo(targetX: number, targetY: number, targetZoom: number, duration = 1.15): void {
    this.flight = {
      startX: this.centerX,
      startY: this.centerY,
      startLogZoom: Math.log(this.zoom),
      endX: targetX,
      endY: targetY,
      endLogZoom: Math.log(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, targetZoom))),
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

    // Weight position interpolation so panning completes early when zooming in deeply
    const zoomDelta = f.endLogZoom - f.startLogZoom;
    let posT = t;
    if (Math.abs(zoomDelta) > 0.5) {
      // Exact exponential camera path interpolation (keeps target centered visually)
      const num = Math.exp(-zoomDelta * t) - 1;
      const den = Math.exp(-zoomDelta) - 1;
      posT = num / den;
    }

    this.centerX = f.startX + (f.endX - f.startX) * posT;
    this.centerY = f.startY + (f.endY - f.startY) * posT;
    this.zoom = Math.exp(f.startLogZoom + zoomDelta * t);
    this.markDirty();

    if (rawT >= 1) {
      this.centerX = f.endX;
      this.centerY = f.endY;
      this.zoom = Math.exp(f.endLogZoom);
      this.flight = null;
    }

    return true;
  }
}
