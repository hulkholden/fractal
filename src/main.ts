import './style.css';
import { CameraController } from './engine/camera';
import {
  createDefaultJuliaState,
  FractalDefinition,
  getFractalById,
  stepJuliaParam,
} from './engine/fractals';
import { createDefaultGradientState, PALETTES, stepGradient } from './engine/palettes';
import { FractalRenderer } from './engine/renderer';
import { BOOKMARKS } from './ui/bookmarks';
import { HudController } from './ui/controls';

// Conditionally polyfill modern Popover & Invoker Commands APIs only when natively absent
if (!Object.hasOwn(HTMLElement.prototype, 'popover')) {
  void import('@oddbird/popover-polyfill');
}
if (!('commandForElement' in HTMLButtonElement.prototype)) {
  void import('invokers-polyfill');
}

const seenErrors = new Set<string>();
function showError(message: string): void {
  if (seenErrors.has(message) || seenErrors.size >= 5) return;
  seenErrors.add(message);
  const overlay = document.getElementById('error-overlay');
  const msgEl = document.getElementById('error-message');
  if (msgEl) {
    msgEl.style.whiteSpace = 'pre-wrap';
    msgEl.textContent = [...seenErrors].join('\n\n');
  }
  if (overlay) overlay.hidden = false;
}

window.addEventListener('error', (e) => showError(`[Error] ${e.message}`));
window.addEventListener('unhandledrejection', (e) => showError(`[Rejection] ${String(e.reason)}`));

async function bootstrap(): Promise<void> {
  const canvas = document.getElementById('fractal-canvas') as HTMLCanvasElement | null;
  if (!canvas) return;

  const params = new URLSearchParams(window.location.search);
  const julia = createDefaultJuliaState();

  const bmParam = params.get('bookmark');
  const matchedBm = bmParam ? BOOKMARKS.find((b) => b.id === bmParam) : undefined;

  let activeFractal: FractalDefinition = getFractalById(
    params.get('fractal') ?? matchedBm?.fractalId ?? 'mandelbrot',
  );

  let startX = matchedBm?.centerX ?? activeFractal.defaultCenter[0];
  let startY = matchedBm?.centerY ?? activeFractal.defaultCenter[1];
  let startZoom = matchedBm?.zoom ?? activeFractal.defaultZoom;

  if (matchedBm?.juliaC) {
    julia.re = matchedBm.juliaC[0];
    julia.im = matchedBm.juliaC[1];
    julia.orbitAngle = Math.atan2(julia.im, julia.re);
  }

  const xParam = Number(params.get('x'));
  const yParam = Number(params.get('y'));
  const zParam = Number(params.get('z'));
  if (params.has('x') && Number.isFinite(xParam)) startX = xParam;
  if (params.has('y') && Number.isFinite(yParam)) startY = yParam;
  if (params.has('z') && Number.isFinite(zParam) && zParam > 0) startZoom = zParam;

  const creParam = Number(params.get('cre'));
  const cimParam = Number(params.get('cim'));
  if (params.has('cre') && Number.isFinite(creParam)) julia.re = creParam;
  if (params.has('cim') && Number.isFinite(cimParam)) julia.im = cimParam;
  if (params.get('orbit') === '1') julia.animateOrbit = true;

  const camera = new CameraController({
    centerX: startX,
    centerY: startY,
    zoom: startZoom,
  });

  const gradient = createDefaultGradientState();
  const palParam = params.get('palette');
  if (palParam) {
    const idx = PALETTES.findIndex((p) => p.id === palParam);
    if (idx >= 0) {
      gradient.paletteIndex = idx;
      gradient.morphCursor = idx;
    }
  }

  const renderer = new FractalRenderer(canvas, activeFractal, (gpuErr) => {
    showError(gpuErr);
  });

  const stageEl = document.getElementById('stage-label');
  try {
    await renderer.init((stage) => {
      if (stageEl) stageEl.textContent = stage;
    });
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
    return;
  }

  const switchFractal = async (nextFractal: FractalDefinition, preserveCamera = false) => {
    activeFractal = nextFractal;
    await renderer.setFractal(nextFractal);
    if (!preserveCamera) {
      camera.flyTo(
        nextFractal.defaultCenter[0],
        nextFractal.defaultCenter[1],
        nextFractal.defaultZoom,
        0.75,
      );
    } else {
      camera.markDirty();
    }
  };

  const exploreJuliaAt = async (re: number, im: number) => {
    const juliaDef = getFractalById('julia');
    julia.animateOrbit = false;
    julia.transition = null;
    julia.re = re;
    julia.im = im;
    julia.orbitAngle = Math.atan2(im, re);
    hud.setActiveFractal(juliaDef);
    await switchFractal(juliaDef, false);
  };

  const hud = new HudController(
    renderer,
    camera,
    gradient,
    julia,
    activeFractal,
    (nextFractal, preserveCamera) => {
      void switchFractal(nextFractal, preserveCamera);
    },
    (re, im) => {
      void exploreJuliaAt(re, im);
    },
  );

  // --- Pointer & Touch Interaction (60Hz Pan + Cursor-Anchored Zoom) ---
  const activePointers = new Map<number, { x: number; y: number }>();
  let lastPinchDist = 0;

  canvas.addEventListener('pointerdown', (e) => {
    // Alt+Click on Mandelbrot opens the corresponding Julia set for that exact c coordinate
    if (e.altKey && activeFractal.id === 'mandelbrot') {
      const rect = canvas.getBoundingClientRect();
      const step = camera.complexHeight / Math.max(1, rect.height);
      const pickedRe = camera.centerX + (e.clientX - rect.left - rect.width * 0.5) * step;
      const pickedIm = camera.centerY + (rect.height * 0.5 - (e.clientY - rect.top)) * step;
      void exploreJuliaAt(pickedRe, pickedIm);
      return;
    }

    canvas.setPointerCapture(e.pointerId);
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (activePointers.size === 2) {
      const pts = [...activePointers.values()];
      lastPinchDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const prev = activePointers.get(e.pointerId);
    if (!prev) return;

    const rect = canvas.getBoundingClientRect();

    if (activePointers.size === 1) {
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      prev.x = e.clientX;
      prev.y = e.clientY;
      camera.panByPixels(dx, dy, rect.height);
    } else if (activePointers.size === 2) {
      prev.x = e.clientX;
      prev.y = e.clientY;
      const pts = [...activePointers.values()];
      const dist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      const midX = (pts[0].x + pts[1].x) * 0.5 - rect.left;
      const midY = (pts[0].y + pts[1].y) * 0.5 - rect.top;

      if (lastPinchDist > 4 && dist > 4) {
        camera.zoomAtPixel(dist / lastPinchDist, midX, midY, rect.width, rect.height);
      }
      lastPinchDist = dist;
    }
  });

  const endPointer = (e: PointerEvent) => {
    activePointers.delete(e.pointerId);
    if (activePointers.size < 2) {
      lastPinchDist = 0;
    }
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  // Smooth wheel / trackpad zoom anchored under the cursor
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sensitivity = e.ctrlKey ? 0.01 : 0.0018;
      const clampedDelta = Math.max(-220, Math.min(220, e.deltaY));
      const factor = Math.exp(-clampedDelta * sensitivity);
      camera.zoomAtPixel(
        factor,
        e.clientX - rect.left,
        e.clientY - rect.top,
        rect.width,
        rect.height,
      );
    },
    { passive: false },
  );

  // Double-click to smoothly fly-zoom into the clicked point (Shift to zoom out)
  canvas.addEventListener('dblclick', (e) => {
    const rect = canvas.getBoundingClientRect();
    const step = camera.complexHeight / Math.max(1, rect.height);
    const targetRe = camera.centerX + (e.clientX - rect.left - rect.width * 0.5) * step;
    const targetIm = camera.centerY + (rect.height * 0.5 - (e.clientY - rect.top)) * step;
    const nextZoom = e.shiftKey ? camera.zoom / 3.5 : camera.zoom * 4.0;
    camera.flyTo(targetRe, targetIm, nextZoom, 0.75);
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

    const rect = canvas.getBoundingClientRect();

    switch (e.key) {
      case ' ':
        e.preventDefault();
        hud.togglePlayPause();
        break;
      case 'r':
      case 'R':
        hud.resetCamera();
        break;
      case 'h':
      case 'H':
        hud.toggleFullHud();
        break;
      case 'm':
      case 'M':
        hud.toggleLivingMorphMode();
        break;
      case 'o':
      case 'O':
        hud.toggleJuliaOrbitMode();
        break;
      case '+':
      case '=':
        camera.zoomAtPixel(1.35, rect.width * 0.5, rect.height * 0.5, rect.width, rect.height);
        break;
      case '-':
      case '_':
        camera.zoomAtPixel(1 / 1.35, rect.width * 0.5, rect.height * 0.5, rect.width, rect.height);
        break;
      case 'ArrowLeft':
        camera.panByPixels(60, 0, rect.height);
        break;
      case 'ArrowRight':
        camera.panByPixels(-60, 0, rect.height);
        break;
      case 'ArrowUp':
        camera.panByPixels(0, 60, rect.height);
        break;
      case 'ArrowDown':
        camera.panByPixels(0, -60, rect.height);
        break;
      default:
        if (e.key >= '1' && e.key <= '6') {
          gradient.livingMorph = false;
          hud.selectPalette(Number(e.key) - 1);
        }
        break;
    }
  });

  const resizeObserver = new ResizeObserver(() => {
    if (renderer.resize()) {
      camera.markDirty();
    }
  });
  resizeObserver.observe(canvas);

  // --- Main 60Hz Animation & Progressive Refinement Loop ---
  let prevTime = performance.now();
  let rafId = 0;
  let disposed = false;

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(rafId);
    resizeObserver.disconnect();
    renderer.destroy();
  };

  window.addEventListener('pagehide', cleanup);
  import.meta.hot?.dispose(cleanup);

  const frame = (now: number) => {
    if (disposed) return;
    const dtSeconds = Math.max(0, (now - prevTime) / 1000);
    prevTime = now;

    camera.update(dtSeconds);
    if (activeFractal.hasJuliaParam && stepJuliaParam(julia, dtSeconds)) {
      camera.markDirty();
    }

    const evaluatedGradient = stepGradient(gradient, dtSeconds);
    const telemetry = renderer.renderFrame(now, camera, gradient, evaluatedGradient, [
      julia.re,
      julia.im,
    ]);
    hud.updateTelemetry(now, telemetry, evaluatedGradient);

    rafId = requestAnimationFrame(frame);
  };

  rafId = requestAnimationFrame(frame);
}

void bootstrap();
