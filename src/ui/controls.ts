import { CameraController } from '../engine/camera';
import {
  FRACTAL_REGISTRY,
  FractalDefinition,
  JULIA_PRESETS,
  JuliaParamState,
} from '../engine/fractals';
import {
  buildLivePreviewCss,
  EvaluatedGradient,
  GradientState,
  PALETTES,
} from '../engine/palettes';
import { FractalRenderer, RenderTelemetry } from '../engine/renderer';
import { BOOKMARKS } from './bookmarks';

function formatZoom(zoom: number): string {
  if (zoom < 1000) {
    return `${zoom.toFixed(2)}×`;
  }
  const exp = Math.floor(Math.log10(zoom));
  const mant = zoom / Math.pow(10, exp);
  return `${mant.toFixed(2)}×10^${exp}`;
}

function formatCoord(val: number, zoom: number): string {
  const digits = Math.min(15, Math.max(6, Math.ceil(Math.log10(Math.max(1, zoom))) + 4));
  return `${val >= 0 ? '+' : ''}${val.toFixed(digits)}`;
}

function formatComplexCompact(re: number, im: number): string {
  const sign = im >= 0 ? '+' : '-';
  return `${re.toFixed(4)} ${sign} ${Math.abs(im).toFixed(4)}i`;
}

export class HudController {
  private statusDot = document.getElementById('status-dot') as HTMLElement;
  private fractalTitle = document.getElementById('fractal-title') as HTMLElement;
  private fractalFormula = document.getElementById('fractal-formula') as HTMLElement;
  private precisionBadge = document.getElementById('precision-badge') as HTMLElement;
  private fpsReadout = document.getElementById('fps-readout') as HTMLElement;
  private stageLabel = document.getElementById('stage-label') as HTMLElement;
  private zoomReadout = document.getElementById('zoom-readout') as HTMLElement;
  private refineBarFill = document.getElementById('refine-bar-fill') as HTMLElement;
  private resReadout = document.getElementById('res-readout') as HTMLElement;
  private coordRe = document.getElementById('coord-re') as HTMLElement;
  private coordIm = document.getElementById('coord-im') as HTMLElement;
  private gpuName = document.getElementById('gpu-name') as HTMLElement;

  private btnPlayPause = document.getElementById('btn-play-pause') as HTMLButtonElement;
  private playPauseIcon = document.getElementById('play-pause-icon') as HTMLElement;
  private btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  private btnSnapshot = document.getElementById('btn-snapshot') as HTMLButtonElement;
  private btnTogglePanel = document.getElementById('btn-toggle-panel') as HTMLButtonElement;
  private controlDrawer = document.getElementById('control-drawer') as HTMLElement;
  private hudHiddenHint = document.getElementById('hud-hidden-hint') as HTMLElement;

  private selectFractal = document.getElementById('select-fractal') as HTMLSelectElement;
  private btnJuliaFromCenter = document.getElementById(
    'btn-julia-from-center',
  ) as HTMLButtonElement;

  private sectionJulia = document.getElementById('section-julia') as HTMLElement;
  private juliaCReadout = document.getElementById('julia-c-readout') as HTMLElement;
  private juliaPresetGrid = document.getElementById('julia-preset-grid') as HTMLElement;
  private toggleJuliaOrbit = document.getElementById('toggle-julia-orbit') as HTMLInputElement;
  private groupOrbitSpeed = document.getElementById('group-orbit-speed') as HTMLElement;
  private sliderOrbitSpeed = document.getElementById('slider-orbit-speed') as HTMLInputElement;
  private outOrbitSpeed = document.getElementById('out-orbit-speed') as HTMLOutputElement;
  private sliderJuliaRe = document.getElementById('slider-julia-re') as HTMLInputElement;
  private outJuliaRe = document.getElementById('out-julia-re') as HTMLOutputElement;
  private sliderJuliaIm = document.getElementById('slider-julia-im') as HTMLInputElement;
  private outJuliaIm = document.getElementById('out-julia-im') as HTMLOutputElement;

  private paletteGrid = document.getElementById('palette-grid') as HTMLElement;
  private paletteActiveName = document.getElementById('palette-active-name') as HTMLElement;
  private gradientPreviewBar = document.getElementById('gradient-preview-bar') as HTMLElement;
  private toggleLivingMorph = document.getElementById('toggle-living-morph') as HTMLInputElement;
  private sliderFlowSpeed = document.getElementById('slider-flow-speed') as HTMLInputElement;
  private outFlowSpeed = document.getElementById('out-flow-speed') as HTMLOutputElement;
  private sliderDensity = document.getElementById('slider-density') as HTMLInputElement;
  private outDensity = document.getElementById('out-density') as HTMLOutputElement;
  private bookmarkList = document.getElementById('bookmark-list') as HTMLElement;

  private paletteButtons: HTMLButtonElement[] = [];
  private juliaPresetButtons: HTMLButtonElement[] = [];
  private activeFractal: FractalDefinition;
  private lastUiTick = 0;

  constructor(
    private renderer: FractalRenderer,
    private camera: CameraController,
    private gradient: GradientState,
    private julia: JuliaParamState,
    initialFractal: FractalDefinition,
    private onFractalChange: (fractal: FractalDefinition, preserveCamera?: boolean) => void,
    private onExploreJuliaAt: (re: number, im: number) => void,
  ) {
    this.activeFractal = initialFractal;
    this.populateFractals();
    this.populateJuliaPresets();
    this.populatePalettes();
    this.populateBookmarks();
    this.bindEvents();
    this.setupRangeFallbackProgress();
    this.syncControls();
  }

  private populateFractals(): void {
    this.selectFractal.innerHTML = '';
    for (const f of FRACTAL_REGISTRY) {
      const opt = document.createElement('option');
      opt.value = f.id;
      opt.textContent = f.name;
      this.selectFractal.appendChild(opt);
    }
    this.selectFractal.value = this.activeFractal.id;
    this.fractalTitle.textContent = this.activeFractal.name;
    this.fractalFormula.textContent = this.activeFractal.formula;
  }

  private populateJuliaPresets(): void {
    this.juliaPresetGrid.innerHTML = '';
    this.juliaPresetButtons = JULIA_PRESETS.map((preset) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-chip';
      btn.dataset.presetId = preset.id;

      const name = document.createElement('span');
      name.className = 'preset-chip-name';
      name.textContent = preset.name;

      const sub = document.createElement('span');
      sub.className = 'preset-chip-sub';
      sub.textContent = preset.subtitle;

      btn.append(name, sub);
      btn.addEventListener('click', () => {
        this.julia.animateOrbit = false;
        this.julia.transition = {
          startRe: this.julia.re,
          startIm: this.julia.im,
          endRe: preset.re,
          endIm: preset.im,
          elapsed: 0,
          duration: 0.65,
        };
        this.camera.markDirty();
        this.syncControls();
      });

      this.juliaPresetGrid.appendChild(btn);
      return btn;
    });
  }

  private populatePalettes(): void {
    this.paletteGrid.innerHTML = '';
    this.paletteButtons = PALETTES.map((p, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'palette-chip';
      btn.dataset.index = String(idx);

      const swatch = document.createElement('span');
      swatch.className = 'palette-chip-swatch';
      swatch.style.background = `linear-gradient(90deg, ${p.hexStops.join(', ')})`;

      const label = document.createElement('span');
      label.className = 'palette-chip-label';
      label.textContent = p.name;

      btn.append(swatch, label);
      btn.addEventListener('click', () => {
        this.selectPalette(idx);
      });
      this.paletteGrid.appendChild(btn);
      return btn;
    });
  }

  private populateBookmarks(): void {
    this.bookmarkList.innerHTML = '';
    const matching = BOOKMARKS.filter((bm) => bm.fractalId === this.activeFractal.id);
    for (const bm of matching) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'bookmark-btn';

      const name = document.createElement('span');
      name.className = 'bookmark-name';
      name.textContent = bm.name;

      const sub = document.createElement('span');
      sub.className = 'bookmark-sub';
      sub.textContent = bm.subtitle;

      btn.append(name, sub);
      btn.addEventListener('click', () => {
        if (bm.juliaC) {
          this.julia.animateOrbit = false;
          this.julia.transition = {
            startRe: this.julia.re,
            startIm: this.julia.im,
            endRe: bm.juliaC[0],
            endIm: bm.juliaC[1],
            elapsed: 0,
            duration: 0.85,
          };
          this.syncControls();
        }
        this.camera.flyTo(bm.centerX, bm.centerY, bm.zoom, 1.25);
      });
      this.bookmarkList.appendChild(btn);
    }
  }

  public setActiveFractal(fractal: FractalDefinition): void {
    this.activeFractal = fractal;
    this.selectFractal.value = fractal.id;
    this.fractalTitle.textContent = fractal.name;
    this.fractalFormula.textContent = fractal.formula;
    this.populateBookmarks();
    this.syncControls();
  }

  private bindEvents(): void {
    this.selectFractal.addEventListener('change', () => {
      const chosen =
        FRACTAL_REGISTRY.find((f) => f.id === this.selectFractal.value) ?? FRACTAL_REGISTRY[0];
      this.setActiveFractal(chosen);
      this.onFractalChange(chosen);
    });

    this.btnJuliaFromCenter.addEventListener('click', () => {
      this.onExploreJuliaAt(this.camera.centerX, this.camera.centerY);
    });

    this.toggleJuliaOrbit.addEventListener('change', () => {
      this.julia.animateOrbit = this.toggleJuliaOrbit.checked;
      this.julia.transition = null;
      if (this.julia.animateOrbit) {
        this.julia.orbitAngle = Math.atan2(this.julia.im, this.julia.re);
      }
      this.syncControls();
    });

    this.sliderOrbitSpeed.addEventListener('input', () => {
      this.julia.orbitSpeed = parseFloat(this.sliderOrbitSpeed.value);
      this.outOrbitSpeed.textContent = `${this.julia.orbitSpeed.toFixed(2)} rad/s`;
    });

    this.sliderJuliaRe.addEventListener('input', () => {
      this.julia.animateOrbit = false;
      this.julia.transition = null;
      this.julia.re = parseFloat(this.sliderJuliaRe.value);
      this.julia.orbitAngle = Math.atan2(this.julia.im, this.julia.re);
      this.camera.markDirty();
      this.syncJuliaReadouts();
      this.toggleJuliaOrbit.checked = false;
      this.groupOrbitSpeed.hidden = true;
    });

    this.sliderJuliaIm.addEventListener('input', () => {
      this.julia.animateOrbit = false;
      this.julia.transition = null;
      this.julia.im = parseFloat(this.sliderJuliaIm.value);
      this.julia.orbitAngle = Math.atan2(this.julia.im, this.julia.re);
      this.camera.markDirty();
      this.syncJuliaReadouts();
      this.toggleJuliaOrbit.checked = false;
      this.groupOrbitSpeed.hidden = true;
    });

    this.btnPlayPause.addEventListener('click', () => {
      this.togglePlayPause();
    });

    this.btnReset.addEventListener('click', () => {
      this.resetCamera();
    });

    this.btnSnapshot.addEventListener('click', async () => {
      this.btnSnapshot.disabled = true;
      try {
        const blob = await this.renderer.captureSnapshot();
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${this.activeFractal.id}-${Date.now()}.png`;
          a.click();
          URL.revokeObjectURL(url);
        }
      } finally {
        this.btnSnapshot.disabled = false;
      }
    });

    this.btnTogglePanel.addEventListener('click', () => {
      const collapsed = this.controlDrawer.classList.toggle('is-collapsed');
      this.btnTogglePanel.setAttribute('aria-expanded', String(!collapsed));
    });

    this.toggleLivingMorph.addEventListener('change', () => {
      this.gradient.livingMorph = this.toggleLivingMorph.checked;
      if (this.gradient.livingMorph) {
        this.gradient.morphCursor = this.gradient.paletteIndex;
      }
      this.syncControls();
    });

    this.sliderFlowSpeed.addEventListener('input', () => {
      this.gradient.flowSpeed = parseFloat(this.sliderFlowSpeed.value);
      this.outFlowSpeed.textContent = `${this.gradient.flowSpeed.toFixed(2)}×`;
    });

    this.sliderDensity.addEventListener('input', () => {
      this.gradient.density = parseFloat(this.sliderDensity.value);
      this.outDensity.textContent = `${this.gradient.density.toFixed(2)}×`;
    });

    // Segmented buttons: SSAA
    document.querySelectorAll<HTMLButtonElement>('[data-ssaa]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const val = Number(btn.dataset.ssaa) as 1 | 4 | 8;
        this.renderer.updateSettings({ targetSSAA: val });
        this.setActiveSegment('[data-ssaa]', btn);
      });
    });

    // Segmented buttons: Base stationary iterations
    document.querySelectorAll<HTMLButtonElement>('[data-iter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const val = Number(btn.dataset.iter);
        this.renderer.updateSettings({ baseStationaryIter: val });
        this.setActiveSegment('[data-iter]', btn);
      });
    });

    // Segmented buttons: Interactive resolution scale
    document.querySelectorAll<HTMLButtonElement>('[data-scale]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const raw = btn.dataset.scale;
        const mode = raw === 'auto' ? 'auto' : (Number(raw) as 0.5 | 0.75 | 1.0);
        this.renderer.updateSettings({ interactiveScaleMode: mode });
        this.setActiveSegment('[data-scale]', btn);
      });
    });

    // Return focus to trigger when closing the shortcuts popover from inside
    const popoverEl = document.getElementById('shortcuts-popover');
    const triggerEl = document.getElementById('btn-shortcuts');
    popoverEl?.addEventListener('toggle', (ev) => {
      const toggleEvent = ev as ToggleEvent;
      if (toggleEvent.newState === 'closed') {
        triggerEl?.focus();
      }
    });
  }

  private setupRangeFallbackProgress(): void {
    if (CSS.supports('accent-color', '#38bdf8')) return;
    const updateProgress = (el: HTMLInputElement) => {
      const min = parseFloat(el.min);
      const max = parseFloat(el.max);
      const val = parseFloat(el.value);
      const pct = ((val - min) / Math.max(1e-6, max - min)) * 100;
      el.style.setProperty('--progress', `${pct}%`);
    };
    for (const slider of [
      this.sliderFlowSpeed,
      this.sliderDensity,
      this.sliderOrbitSpeed,
      this.sliderJuliaRe,
      this.sliderJuliaIm,
    ]) {
      updateProgress(slider);
      slider.addEventListener('input', () => updateProgress(slider));
    }
  }

  private setActiveSegment(selector: string, activeBtn: HTMLButtonElement): void {
    document.querySelectorAll<HTMLButtonElement>(selector).forEach((b) => {
      b.classList.toggle('active', b === activeBtn);
    });
  }

  public selectPalette(index: number): void {
    const safeIdx = ((index % PALETTES.length) + PALETTES.length) % PALETTES.length;
    this.gradient.paletteIndex = safeIdx;
    this.gradient.morphCursor = safeIdx;
    this.gradient.hueDrift = 0;
    this.syncControls();
  }

  public togglePlayPause(): void {
    this.gradient.playing = !this.gradient.playing;
    this.syncControls();
  }

  public toggleLivingMorphMode(): void {
    this.gradient.livingMorph = !this.gradient.livingMorph;
    if (this.gradient.livingMorph) {
      this.gradient.morphCursor = this.gradient.paletteIndex;
    }
    this.syncControls();
  }

  public toggleJuliaOrbitMode(): void {
    if (!this.activeFractal.hasJuliaParam) return;
    this.julia.animateOrbit = !this.julia.animateOrbit;
    this.julia.transition = null;
    if (this.julia.animateOrbit) {
      this.julia.orbitAngle = Math.atan2(this.julia.im, this.julia.re);
    }
    this.syncControls();
  }

  public resetCamera(): void {
    this.camera.flyTo(
      this.activeFractal.defaultCenter[0],
      this.activeFractal.defaultCenter[1],
      this.activeFractal.defaultZoom,
      0.9,
    );
  }

  public toggleFullHud(): void {
    const hidden = document.body.classList.toggle('hud-hidden');
    this.hudHiddenHint.hidden = !hidden;
  }

  private syncJuliaReadouts(): void {
    const clampedRe = Math.max(-1.5, Math.min(1.5, this.julia.re));
    const clampedIm = Math.max(-1.5, Math.min(1.5, this.julia.im));
    this.sliderJuliaRe.value = String(clampedRe);
    this.sliderJuliaIm.value = String(clampedIm);
    this.outJuliaRe.textContent = `${this.julia.re >= 0 ? '+' : ''}${this.julia.re.toFixed(4)}`;
    this.outJuliaIm.textContent = `${this.julia.im >= 0 ? '+' : ''}${this.julia.im.toFixed(4)}`;
    this.juliaCReadout.textContent = formatComplexCompact(this.julia.re, this.julia.im);

    this.juliaPresetButtons.forEach((btn, idx) => {
      const p = JULIA_PRESETS[idx];
      const near =
        !this.julia.animateOrbit &&
        Math.hypot(this.julia.re - p.re, this.julia.im - p.im) < 0.002;
      btn.classList.toggle('active', near);
    });
  }

  public syncControls(): void {
    this.playPauseIcon.textContent = this.gradient.playing ? '❚❚' : '▶';
    this.btnPlayPause.setAttribute('aria-pressed', String(this.gradient.playing));
    this.toggleLivingMorph.checked = this.gradient.livingMorph;
    this.sliderFlowSpeed.value = String(this.gradient.flowSpeed);
    this.outFlowSpeed.textContent = `${this.gradient.flowSpeed.toFixed(2)}×`;
    this.sliderDensity.value = String(this.gradient.density);
    this.outDensity.textContent = `${this.gradient.density.toFixed(2)}×`;

    const isJulia = this.activeFractal.hasJuliaParam;
    this.sectionJulia.hidden = !isJulia;
    this.btnJuliaFromCenter.hidden = isJulia;
    this.toggleJuliaOrbit.checked = this.julia.animateOrbit;
    this.groupOrbitSpeed.hidden = !this.julia.animateOrbit;
    this.sliderOrbitSpeed.value = String(this.julia.orbitSpeed);
    this.outOrbitSpeed.textContent = `${this.julia.orbitSpeed.toFixed(2)} rad/s`;
    if (isJulia) {
      this.syncJuliaReadouts();
    }

    this.paletteButtons.forEach((btn, idx) => {
      btn.classList.toggle(
        'active',
        !this.gradient.livingMorph && idx === this.gradient.paletteIndex,
      );
    });
  }

  public updateTelemetry(
    now: number,
    telemetry: RenderTelemetry,
    evaluatedGradient: EvaluatedGradient,
  ): void {
    if (now - this.lastUiTick < 50) return;
    this.lastUiTick = now;

    this.statusDot.dataset.stage = telemetry.stage;
    this.fpsReadout.textContent = `${telemetry.fps} fps`;
    this.zoomReadout.textContent = formatZoom(this.camera.zoom);

    this.precisionBadge.textContent = telemetry.useDf64 ? 'DF64' : 'FP32';
    this.precisionBadge.dataset.df64 = String(telemetry.useDf64);

    if (telemetry.stage === 'interactive') {
      const pct = Math.round(telemetry.interactiveScale * 100);
      this.stageLabel.textContent = `Interactive · ${pct}% · ${telemetry.currentIter} it`;
      this.refineBarFill.style.width = '12%';
    } else if (telemetry.stage === 'refining') {
      const sampleDisplay = Math.max(1, telemetry.completedSamples);
      this.stageLabel.textContent = `Refining · ${sampleDisplay}/${telemetry.targetSamples}× · ${telemetry.currentIter}/${telemetry.targetIter} it`;
      this.refineBarFill.style.width = `${Math.round(telemetry.progress * 100)}%`;
    } else {
      this.stageLabel.textContent = `Ultra · ${telemetry.targetSamples}× SSAA · ${telemetry.targetIter} it`;
      this.refineBarFill.style.width = '100%';
    }

    this.resReadout.textContent = `${telemetry.activeWidth}×${telemetry.activeHeight}`;
    this.coordRe.textContent = formatCoord(this.camera.centerX, this.camera.zoom);
    this.coordIm.textContent = formatCoord(this.camera.centerY, this.camera.zoom);
    this.gpuName.textContent = telemetry.gpuName;

    if (this.activeFractal.hasJuliaParam && (this.julia.animateOrbit || this.julia.transition)) {
      this.syncJuliaReadouts();
    }

    // Live gradient preview strip & active palette title
    this.gradientPreviewBar.style.background = buildLivePreviewCss(this.gradient, evaluatedGradient);
    if (this.gradient.livingMorph) {
      const pct = Math.round(evaluatedGradient.morphBlend * 100);
      this.paletteActiveName.textContent =
        pct < 15
          ? `Morph · ${evaluatedGradient.paletteA.name}`
          : pct > 85
            ? `Morph · ${evaluatedGradient.paletteB.name}`
            : `${evaluatedGradient.paletteA.name} → ${evaluatedGradient.paletteB.name}`;
    } else {
      this.paletteActiveName.textContent = evaluatedGradient.paletteA.name;
    }
  }
}
