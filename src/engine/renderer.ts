import {
  bigFromDecimal,
  computeBestReferenceOrbit,
  extendReferenceOrbit,
  ReferenceOrbit,
} from './bigfixed';
import { CameraController, PrecisionMode, splitDouble } from './camera';
import { assembleComputeShader, FractalDefinition } from './fractals';
import { EvaluatedGradient, GradientState } from './palettes';
import displayWgsl from './shaders/display.wgsl?raw';

export type QualityStage = 'interactive' | 'refining' | 'converged';

export interface RendererSettings {
  /** Target SSAA samples when stationary: 1, 4, or 8 */
  targetSSAA: 1 | 4 | 8;
  /** Minimum stationary max iterations (auto-scales higher at deep zooms) */
  baseStationaryIter: number;
  /** 'auto' dynamically adapts to hold 60Hz, or fixed numeric scale (0.5, 0.75, 1.0) */
  interactiveScaleMode: 'auto' | 0.5 | 0.75 | 1.0;
}

export interface RenderTelemetry {
  stage: QualityStage;
  fps: number;
  frameMs: number;
  activeWidth: number;
  activeHeight: number;
  fullWidth: number;
  fullHeight: number;
  interactiveScale: number;
  completedSamples: number;
  targetSamples: number;
  currentIter: number;
  targetIter: number;
  progress: number; // 0..1
  precisionMode: PrecisionMode;
  precisionBits: number;
  gpuName: string;
}

// 8 dyadic low-discrepancy / rotated-grid subpixel offsets in [-0.5, 0.5]
const SUBPIXEL_OFFSETS: readonly [number, number][] = [
  [0.0, 0.0],
  [-0.375, -0.125],
  [0.125, -0.375],
  [0.375, 0.125],
  [-0.125, 0.375],
  [-0.3125, 0.3125],
  [0.3125, -0.3125],
  [-0.1875, -0.3125],
];

const SETTLE_DELAY_MS = 95;
const COMPUTE_UNIFORM_SIZE = 128; // 8 * vec4 (f32/u32/i32)
const DISPLAY_UNIFORM_SIZE = 304; // 3 * vec4 + 8 * vec4 + 8 * vec4
const MAX_PERTURB_ITER = 24576;
const REF_ORBIT_BUFFER_BYTES = (MAX_PERTURB_ITER + 2) * 8; // vec2<f32> per step

interface ComputePipelinePair {
  interactive: GPUComputePipeline;
  refine: GPUComputePipeline;
}

export class FractalRenderer {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private gpuName = 'WebGPU';

  private computeBindGroupLayout!: GPUBindGroupLayout;
  private computePipelineLayout!: GPUPipelineLayout;
  private computePipelines = new Map<string, ComputePipelinePair>();
  private computePipelineInteractive!: GPUComputePipeline;
  private computePipelineRefine!: GPUComputePipeline;
  private displayPipeline!: GPURenderPipeline;

  private computeUniformBuffer!: GPUBuffer;
  private displayUniformBuffer!: GPUBuffer;
  private refOrbitBuffer!: GPUBuffer;

  private samples0Buffer: GPUBuffer | null = null;
  private samples1Buffer: GPUBuffer | null = null;
  private stateZBuffer: GPUBuffer | null = null;
  private stateMetaBuffer: GPUBuffer | null = null;

  private computeBindGroup: GPUBindGroup | null = null;
  private displayBindGroup: GPUBindGroup | null = null;

  private computeUniformRaw = new ArrayBuffer(COMPUTE_UNIFORM_SIZE);
  private computeUniformF32 = new Float32Array(this.computeUniformRaw);
  private computeUniformU32 = new Uint32Array(this.computeUniformRaw);
  private computeUniformI32 = new Int32Array(this.computeUniformRaw);

  private refOrbit: ReferenceOrbit | null = null;
  private refOrbitJuliaRe = 0;
  private refOrbitJuliaIm = 0;

  private displayUniformRaw = new ArrayBuffer(DISPLAY_UNIFORM_SIZE);
  private displayUniformF32 = new Float32Array(this.displayUniformRaw);
  private displayUniformU32 = new Uint32Array(this.displayUniformRaw);

  private fullWidth = 1;
  private fullHeight = 1;
  private activeWidth = 1;
  private activeHeight = 1;
  private upscaleBilinear = false;

  private fractal: FractalDefinition;
  public settings: RendererSettings = {
    targetSSAA: 8,
    baseStationaryIter: 1536,
    interactiveScaleMode: 'auto',
  };

  // Adaptive 60Hz controller state
  private autoScale = 0.75;
  private emaFrameMs = 16.6;
  private emaRefineMs = 12.0;
  private fpsDisplay = 60;
  private lastFrameTimestamp = performance.now();

  // Progressive stationary refinement state
  private stage: QualityStage = 'interactive';
  private refineSampleIdx = 0;
  private refineIter = 0;
  private completedSamples = 0;
  private visibleSampleCount = 1;
  private currentTargetIter = 1536;
  private currentInteractiveIter = 200;
  private needsInitialCompute = true;

  // Last gradient values for instant snapshot re-render
  private lastGradientState: GradientState | null = null;
  private lastEvaluatedGradient: EvaluatedGradient | null = null;
  private onGpuError?: (message: string) => void;

  constructor(
    canvas: HTMLCanvasElement,
    initialFractal: FractalDefinition,
    onGpuError?: (message: string) => void,
  ) {
    this.canvas = canvas;
    this.fractal = initialFractal;
    this.onGpuError = onGpuError;
  }

  private firstFrameValidated = false;

  public async init(onStage?: (label: string) => void): Promise<void> {
    if (!navigator.gpu) {
      throw new Error(
        'WebGPU is not available in this browser. Try Chrome, Edge, or Safari 26+ with WebGPU enabled.',
      );
    }

    onStage?.('Init: requesting GPU adapter...');
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('Failed to acquire a WebGPU adapter on this device.');
    }

    const info = adapter.info;
    this.gpuName = info?.description || info?.architecture || info?.vendor || 'WebGPU GPU';

    onStage?.('Init: requesting GPU device...');
    const desiredStorageSize = Math.min(
      adapter.limits.maxStorageBufferBindingSize,
      256 * 1024 * 1024,
    );
    const desiredBufferSize = Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024);
    const useHigherLimits = desiredStorageSize > 128 * 1024 * 1024;

    const devicePromise = useHigherLimits
      ? adapter
          .requestDevice({
            requiredLimits: {
              maxStorageBufferBindingSize: desiredStorageSize,
              maxBufferSize: desiredBufferSize,
            },
          })
          .catch(() => adapter.requestDevice())
      : adapter.requestDevice();

    this.device = await Promise.race([
      devicePromise,
      new Promise<never>((_, reject) =>
        setTimeout(
          () =>
            reject(
              new Error(
                'WebGPU device initialization timed out. Try reloading the tab or restarting the browser.',
              ),
            ),
          6000,
        ),
      ),
    ]);

    this.device.lost.then((lostInfo) => {
      console.error('WebGPU device lost:', lostInfo.message);
    });

    this.device.addEventListener('uncapturederror', (event) => {
      const msg = `[WebGPU] ${event.error.message}`;
      console.error(msg);
      this.onGpuError?.(msg);
    });

    onStage?.('Init: configuring canvas...');
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) {
      throw new Error('Unable to initialize a WebGPU canvas context.');
    }
    this.context = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    this.computeUniformBuffer = this.device.createBuffer({
      label: 'compute-uniforms',
      size: COMPUTE_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.displayUniformBuffer = this.device.createBuffer({
      label: 'display-uniforms',
      size: DISPLAY_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.refOrbitBuffer = this.device.createBuffer({
      label: 'perturb-ref-orbit',
      size: REF_ORBIT_BUFFER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.computeBindGroupLayout = this.device.createBindGroupLayout({
      label: 'compute-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.computePipelineLayout = this.device.createPipelineLayout({
      label: 'compute-pipeline-layout',
      bindGroupLayouts: [this.computeBindGroupLayout],
    });

    onStage?.('Init: compiling shaders...');
    await this.buildDisplayPipeline();
    await this.buildComputePipelines(this.fractal);
    this.resize();
    onStage?.('Init: ready');
  }

  public destroy(): void {
    this.samples0Buffer?.destroy();
    this.samples1Buffer?.destroy();
    this.stateZBuffer?.destroy();
    this.stateMetaBuffer?.destroy();
    this.computeUniformBuffer?.destroy();
    this.displayUniformBuffer?.destroy();
    this.refOrbitBuffer?.destroy();
    this.device?.destroy();
  }

  public async setFractal(fractal: FractalDefinition): Promise<void> {
    this.fractal = fractal;
    this.refOrbit = null;
    await this.buildComputePipelines(fractal);
    this.rebuildBindGroups();
    this.invalidate();
  }

  public invalidate(): void {
    this.stage = 'interactive';
    this.refineSampleIdx = 0;
    this.refineIter = 0;
    this.completedSamples = 0;
    this.visibleSampleCount = 1;
    this.needsInitialCompute = true;
  }

  public updateSettings(partial: Partial<RendererSettings>): void {
    const prevSSAA = this.settings.targetSSAA;
    const prevIter = this.settings.baseStationaryIter;
    this.settings = { ...this.settings, ...partial };

    if (this.stage === 'interactive') return;

    if (this.settings.baseStationaryIter !== prevIter) {
      this.stage = 'refining';
      this.refineSampleIdx = 0;
      this.refineIter = 0;
      this.completedSamples = 0;
      return;
    }

    if (this.settings.targetSSAA !== prevSSAA) {
      if (this.completedSamples >= this.settings.targetSSAA) {
        this.completedSamples = this.settings.targetSSAA;
        this.visibleSampleCount = this.settings.targetSSAA;
        this.refineSampleIdx = this.settings.targetSSAA;
        this.refineIter = 0;
        this.stage = 'converged';
      } else {
        this.refineSampleIdx = this.completedSamples;
        this.refineIter = 0;
        this.stage = 'refining';
      }
    }
  }

  private async buildDisplayPipeline(): Promise<void> {
    const displayModule = this.device.createShaderModule({
      label: 'display-shader',
      code: displayWgsl,
    });
    const info = await displayModule.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const text = errors
        .map((m) => `[WGSL display:${m.lineNum}:${m.linePos}] ${m.message}`)
        .join('\n');
      throw new Error(text);
    }

    this.displayPipeline = await this.device.createRenderPipelineAsync({
      label: 'display-pipeline',
      layout: 'auto',
      vertex: {
        module: displayModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: displayModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  private async buildComputePipelines(fractal: FractalDefinition): Promise<void> {
    const cached = this.computePipelines.get(fractal.id);
    if (cached) {
      this.computePipelineInteractive = cached.interactive;
      this.computePipelineRefine = cached.refine;
      return;
    }

    const computeCode = assembleComputeShader(fractal);
    const computeModule = this.device.createShaderModule({
      label: `compute-${fractal.id}`,
      code: computeCode,
    });
    const info = await computeModule.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const text = errors
        .map((m) => `[WGSL compute:${m.lineNum}:${m.linePos}] ${m.message}`)
        .join('\n');
      throw new Error(text);
    }

    const interactive = await this.device.createComputePipelineAsync({
      label: `compute-interactive-${fractal.id}`,
      layout: this.computePipelineLayout,
      compute: {
        module: computeModule,
        entryPoint: 'main_interactive',
      },
    });

    const refine = await this.device.createComputePipelineAsync({
      label: `compute-refine-${fractal.id}`,
      layout: this.computePipelineLayout,
      compute: {
        module: computeModule,
        entryPoint: 'main_refine',
      },
    });

    this.computePipelines.set(fractal.id, { interactive, refine });
    this.computePipelineInteractive = interactive;
    this.computePipelineRefine = refine;
  }

  public resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const rect = this.canvas.getBoundingClientRect();
    let targetW = Math.max(1, Math.round(rect.width * dpr));
    let targetH = Math.max(1, Math.round(rect.height * dpr));

    // Ensure largest per-pixel storage buffer (16 bytes/pixel) fits safely inside device limits
    const maxBytes = this.device.limits.maxStorageBufferBindingSize;
    const maxPixels = Math.floor((maxBytes * 0.92) / 16);
    if (targetW * targetH > maxPixels) {
      const shrink = Math.sqrt(maxPixels / (targetW * targetH));
      targetW = Math.max(1, Math.floor(targetW * shrink));
      targetH = Math.max(1, Math.floor(targetH * shrink));
    }

    if (targetW === this.fullWidth && targetH === this.fullHeight && this.samples0Buffer) {
      return false;
    }

    this.fullWidth = targetW;
    this.fullHeight = targetH;
    this.canvas.width = targetW;
    this.canvas.height = targetH;

    this.allocateBuffers(targetW * targetH);
    this.rebuildBindGroups();
    this.invalidate();
    return true;
  }

  private allocateBuffers(pixelCount: number): void {
    this.samples0Buffer?.destroy();
    this.samples1Buffer?.destroy();
    this.stateZBuffer?.destroy();
    this.stateMetaBuffer?.destroy();

    const vec4Bytes = pixelCount * 16;
    const vec2U32Bytes = pixelCount * 8;

    this.samples0Buffer = this.device.createBuffer({
      label: 'samples-0-3',
      size: vec4Bytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.samples1Buffer = this.device.createBuffer({
      label: 'samples-4-7',
      size: vec4Bytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.stateZBuffer = this.device.createBuffer({
      label: 'state-z',
      size: vec4Bytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.stateMetaBuffer = this.device.createBuffer({
      label: 'state-meta',
      size: vec2U32Bytes,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  private rebuildBindGroups(): void {
    if (
      !this.samples0Buffer ||
      !this.samples1Buffer ||
      !this.stateZBuffer ||
      !this.stateMetaBuffer
    ) {
      return;
    }

    this.computeBindGroup = this.device.createBindGroup({
      label: 'compute-bg',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.computeUniformBuffer } },
        { binding: 1, resource: { buffer: this.samples0Buffer } },
        { binding: 2, resource: { buffer: this.samples1Buffer } },
        { binding: 3, resource: { buffer: this.stateZBuffer } },
        { binding: 4, resource: { buffer: this.stateMetaBuffer } },
        { binding: 5, resource: { buffer: this.refOrbitBuffer } },
      ],
    });

    this.displayBindGroup = this.device.createBindGroup({
      label: 'display-bg',
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.displayUniformBuffer } },
        { binding: 1, resource: { buffer: this.samples0Buffer } },
        { binding: 2, resource: { buffer: this.samples1Buffer } },
      ],
    });
  }

  private computeTargetStationaryIter(log10Zoom: number): number {
    const zoomSuggested = Math.round(512 + 380 * Math.max(0, log10Zoom));
    return Math.min(MAX_PERTURB_ITER, Math.max(this.settings.baseStationaryIter, zoomSuggested));
  }

  private computeInteractiveIter(log10Zoom: number, mode: PrecisionMode): number {
    if (mode === 'perturb') {
      return Math.min(1200, Math.max(220, Math.round(180 + 38 * Math.max(0, log10Zoom))));
    }
    const suggested = Math.round(140 + 80 * Math.max(0, log10Zoom));
    const cap = mode === 'df64' ? 380 : 560;
    return Math.min(cap, Math.max(140, suggested));
  }

  private ensureReferenceOrbit(
    camera: CameraController,
    neededIter: number,
    juliaC: [number, number],
  ): { offsetX: number; offsetY: number; scaleExp: number; scaledHeight: number; refLen: number } {
    const { scaledHeight, scaleExp } = camera.getScaledHeight();
    const clampedIter = Math.min(MAX_PERTURB_ITER, Math.max(256, neededIter));
    const isJulia = this.fractal.hasJuliaParam;

    let needRecompute = !this.refOrbit;
    if (this.refOrbit) {
      if (this.refOrbit.fractalId !== this.fractal.id) {
        needRecompute = true;
      } else if (this.refOrbit.bits < camera.precisionBits) {
        needRecompute = true;
      } else if (
        isJulia &&
        (this.refOrbitJuliaRe !== juliaC[0] || this.refOrbitJuliaIm !== juliaC[1])
      ) {
        needRecompute = true;
      } else {
        const [ox, oy] = camera.offsetFromReference(
          this.refOrbit.refX,
          this.refOrbit.refY,
          scaleExp,
        );
        if (!Number.isFinite(ox) || !Number.isFinite(oy) || Math.hypot(ox, oy) > 3.5) {
          needRecompute = true;
        }
      }
    }

    if (needRecompute) {
      const juliaBig: [ReturnType<typeof bigFromDecimal>, ReturnType<typeof bigFromDecimal>] | undefined =
        isJulia
          ? [
              bigFromDecimal(String(juliaC[0]), camera.precisionBits),
              bigFromDecimal(String(juliaC[1]), camera.precisionBits),
            ]
          : undefined;

      this.refOrbit = computeBestReferenceOrbit(
        camera.centerXBig,
        camera.centerYBig,
        scaleExp,
        clampedIter,
        this.fractal.id,
        juliaBig,
      );
      this.refOrbitJuliaRe = juliaC[0];
      this.refOrbitJuliaIm = juliaC[1];
      this.uploadReferenceOrbit(this.refOrbit);
    } else if (this.refOrbit && !this.refOrbit.escaped && this.refOrbit.length < clampedIter) {
      this.refOrbit = extendReferenceOrbit(this.refOrbit, clampedIter);
      this.uploadReferenceOrbit(this.refOrbit);
    }

    const orbit = this.refOrbit!;
    const [offsetX, offsetY] = camera.offsetFromReference(orbit.refX, orbit.refY, scaleExp);
    return {
      offsetX,
      offsetY,
      scaleExp,
      scaledHeight,
      refLen: Math.max(1, orbit.length),
    };
  }

  private uploadReferenceOrbit(orbit: ReferenceOrbit): void {
    const byteLength = Math.min(REF_ORBIT_BUFFER_BYTES, orbit.orbitData.byteLength);
    this.device.queue.writeBuffer(
      this.refOrbitBuffer,
      0,
      orbit.orbitData.buffer,
      orbit.orbitData.byteOffset,
      byteLength,
    );
  }

  private getEffectiveInteractiveScale(): number {
    if (this.settings.interactiveScaleMode === 'auto') {
      return this.autoScale;
    }
    return this.settings.interactiveScaleMode;
  }

  private writeComputeUniforms(
    camera: CameraController,
    width: number,
    height: number,
    jitterX: number,
    jitterY: number,
    startIter: number,
    chunkIter: number,
    maxIter: number,
    sampleIndex: number,
    isFirstChunk: boolean,
    juliaC: [number, number],
    perturbInfo: {
      offsetX: number;
      offsetY: number;
      scaleExp: number;
      scaledHeight: number;
      refLen: number;
    } | null,
  ): void {
    const mode = camera.precisionMode;
    const precCode = mode === 'perturb' ? 2 : mode === 'df64' ? 1 : 0;

    const [cxHi, cxLo] = mode === 'perturb' ? [0, 0] : splitDouble(camera.centerX);
    const [cyHi, cyLo] = mode === 'perturb' ? [0, 0] : splitDouble(camera.centerY);
    const step = mode === 'perturb' ? 0 : camera.complexHeight / Math.max(1, height);
    const [stepHi, stepLo] = splitDouble(step);
    const bailout2 = this.fractal.bailoutRadius * this.fractal.bailoutRadius;
    const [jReHi, jReLo] = splitDouble(juliaC[0]);
    const [jImHi, jImLo] = splitDouble(juliaC[1]);

    const f = this.computeUniformF32;
    const u = this.computeUniformU32;
    const i32 = this.computeUniformI32;

    // 0: center: vec4<f32>
    f[0] = cxHi;
    f[1] = cxLo;
    f[2] = cyHi;
    f[3] = cyLo;

    // 1: scale: vec4<f32> (step_hi, step_lo, bailout2, df_one = 1.0)
    f[4] = stepHi;
    f[5] = stepLo;
    f[6] = bailout2;
    f[7] = 1.0;

    // 2: viewport: vec4<f32>
    f[8] = width;
    f[9] = height;
    f[10] = jitterX;
    f[11] = jitterY;

    // 3: iter_params: vec4<u32>
    u[12] = startIter >>> 0;
    u[13] = chunkIter >>> 0;
    u[14] = maxIter >>> 0;
    u[15] = sampleIndex >>> 0;

    // 4: mode_params: vec4<u32> (precCode, isFirstChunk, activeStride, isJulia)
    u[16] = precCode >>> 0;
    u[17] = isFirstChunk ? 1 : 0;
    u[18] = width >>> 0;
    u[19] = this.fractal.hasJuliaParam ? 1 : 0;

    // 5: julia_c: vec4<f32>
    f[20] = jReHi;
    f[21] = jReLo;
    f[22] = jImHi;
    f[23] = jImLo;

    // 6: perturb_offset: vec4<f32> (offset_x, offset_y, step_scaled, 0)
    if (perturbInfo) {
      f[24] = perturbInfo.offsetX;
      f[25] = perturbInfo.offsetY;
      f[26] = perturbInfo.scaledHeight / Math.max(1, height);
      f[27] = 0;

      // 7: perturb_meta: vec4<i32> (ref_len, scale_exp, 0, 0)
      i32[28] = perturbInfo.refLen | 0;
      i32[29] = perturbInfo.scaleExp | 0;
      i32[30] = 0;
      i32[31] = 0;
    } else {
      f[24] = 0;
      f[25] = 0;
      f[26] = 0;
      f[27] = 0;
      i32[28] = 1;
      i32[29] = 0;
      i32[30] = 0;
      i32[31] = 0;
    }

    this.device.queue.writeBuffer(this.computeUniformBuffer, 0, this.computeUniformRaw);
  }

  private writeDisplayUniforms(
    gradientState: GradientState,
    evaluated: EvaluatedGradient,
  ): void {
    const f = this.displayUniformF32;
    const u = this.displayUniformU32;

    // dims: vec4<f32>
    f[0] = this.fullWidth;
    f[1] = this.fullHeight;
    f[2] = this.activeWidth;
    f[3] = this.activeHeight;

    // color_params: vec4<f32>
    f[4] = gradientState.phase;
    f[5] = gradientState.density;
    f[6] = evaluated.morphBlend;
    f[7] = evaluated.hueDrift;

    // render_flags: vec4<u32>
    u[8] = Math.max(1, this.visibleSampleCount) >>> 0;
    u[9] = this.upscaleBilinear ? 1 : 0;
    u[10] = 0;
    u[11] = 0;

    // palette_a: array<vec4<f32>, 8> starting at float offset 12
    f.set(evaluated.paletteA.oklabData, 12);
    // palette_b: array<vec4<f32>, 8> starting at float offset 44
    f.set(evaluated.paletteB.oklabData, 44);

    this.device.queue.writeBuffer(this.displayUniformBuffer, 0, this.displayUniformRaw);
  }

  public renderFrame(
    now: number,
    camera: CameraController,
    gradientState: GradientState,
    evaluatedGradient: EvaluatedGradient,
    juliaC: [number, number],
  ): RenderTelemetry {
    const dtMs = Math.max(1, Math.min(100, now - this.lastFrameTimestamp));
    this.lastFrameTimestamp = now;
    this.emaFrameMs = this.emaFrameMs * 0.86 + dtMs * 0.14;
    this.fpsDisplay = Math.round(1000 / Math.max(1, this.emaFrameMs));

    this.lastGradientState = gradientState;
    this.lastEvaluatedGradient = evaluatedGradient;

    const movedThisFrame = camera.consumeDirty() || this.needsInitialCompute;
    const msSinceMove = camera.msSinceLastMove(now);
    const precMode = camera.precisionMode;
    const log10Zoom = camera.log10Zoom;

    const shouldValidate = !this.firstFrameValidated;
    if (shouldValidate) {
      this.device.pushErrorScope('validation');
    }

    const encoder = this.device.createCommandEncoder({ label: 'fractal-frame' });

    if (movedThisFrame) {
      this.needsInitialCompute = false;
      this.stage = 'interactive';
      this.refineSampleIdx = 0;
      this.refineIter = 0;
      this.completedSamples = 0;
      this.visibleSampleCount = 1;

      // Adapt interactive resolution scale to stay locked near 60Hz
      if (this.settings.interactiveScaleMode === 'auto') {
        if (this.emaFrameMs > 18.2) {
          this.autoScale = Math.max(0.38, this.autoScale - 0.045);
        } else if (this.emaFrameMs < 13.5) {
          this.autoScale = Math.min(1.0, this.autoScale + 0.025);
        }
      }

      const scale = this.getEffectiveInteractiveScale();
      const fastW = Math.max(64, Math.round(this.fullWidth * scale));
      const fastH = Math.max(64, Math.round(this.fullHeight * scale));
      this.activeWidth = fastW;
      this.activeHeight = fastH;
      this.upscaleBilinear = fastW < this.fullWidth || fastH < this.fullHeight;

      this.currentInteractiveIter = this.computeInteractiveIter(log10Zoom, precMode);

      const perturbInfo =
        precMode === 'perturb'
          ? this.ensureReferenceOrbit(camera, this.currentInteractiveIter, juliaC)
          : null;

      this.writeComputeUniforms(
        camera,
        fastW,
        fastH,
        0,
        0,
        0,
        this.currentInteractiveIter,
        this.currentInteractiveIter,
        0,
        true,
        juliaC,
        perturbInfo,
      );

      if (this.computeBindGroup) {
        const pass = encoder.beginComputePass({ label: 'interactive-compute' });
        pass.setPipeline(this.computePipelineInteractive);
        pass.setBindGroup(0, this.computeBindGroup);
        pass.dispatchWorkgroups(Math.ceil(fastW / 8), Math.ceil(fastH / 8));
        pass.end();
      }
    } else if (msSinceMove >= SETTLE_DELAY_MS && this.stage !== 'converged') {
      // Viewport is stationary: progressively refine to full DPR + deep iterations + SSAA
      this.stage = 'refining';
      this.emaRefineMs = this.emaRefineMs * 0.8 + dtMs * 0.2;

      this.currentTargetIter = this.computeTargetStationaryIter(log10Zoom);
      const targetSSAA = this.settings.targetSSAA;

      // Keep each compute slice comfortably within the 60Hz budget so color cycling stays fluid
      let chunkIter = precMode === 'df64' ? 96 : precMode === 'perturb' ? 512 : 256;
      if (this.emaRefineMs > 19.0) {
        chunkIter = Math.max(32, Math.floor(chunkIter * 0.5));
      } else if (this.emaRefineMs < 12.5 && precMode !== 'df64') {
        chunkIter = precMode === 'perturb' ? 768 : 384;
      }

      const isFirstChunk = this.refineIter === 0;
      const [jx, jy] = SUBPIXEL_OFFSETS[this.refineSampleIdx % SUBPIXEL_OFFSETS.length];

      const perturbInfo =
        precMode === 'perturb'
          ? this.ensureReferenceOrbit(camera, this.currentTargetIter, juliaC)
          : null;

      this.writeComputeUniforms(
        camera,
        this.fullWidth,
        this.fullHeight,
        jx,
        jy,
        this.refineIter,
        chunkIter,
        this.currentTargetIter,
        this.refineSampleIdx,
        isFirstChunk,
        juliaC,
        perturbInfo,
      );

      if (this.computeBindGroup) {
        const pass = encoder.beginComputePass({ label: 'refine-compute' });
        pass.setPipeline(this.computePipelineRefine);
        pass.setBindGroup(0, this.computeBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.fullWidth / 8), Math.ceil(this.fullHeight / 8));
        pass.end();
      }

      // Immediately switch display to crisp 1:1 native DPR as soon as sample 0 starts writing
      if (this.refineSampleIdx === 0) {
        this.activeWidth = this.fullWidth;
        this.activeHeight = this.fullHeight;
        this.upscaleBilinear = false;
        this.visibleSampleCount = 1;
      }

      this.refineIter = Math.min(this.currentTargetIter, this.refineIter + chunkIter);

      if (this.refineIter >= this.currentTargetIter) {
        this.completedSamples = this.refineSampleIdx + 1;
        this.visibleSampleCount = this.completedSamples;
        this.refineSampleIdx++;
        this.refineIter = 0;

        if (this.refineSampleIdx >= targetSSAA) {
          this.stage = 'converged';
        }
      }
    }

    // Always run the 60Hz display pass so the smooth Oklab gradients continuously flow
    this.encodeDisplayPass(encoder, gradientState, evaluatedGradient);
    this.device.queue.submit([encoder.finish()]);

    if (shouldValidate) {
      this.firstFrameValidated = true;
      void this.device.popErrorScope().then((err) => {
        if (err) {
          this.onGpuError?.(`[WebGPU Frame Validation] ${err.message}`);
        }
      });
    }

    const targetSSAA = this.settings.targetSSAA;
    let progress = 0;
    if (this.stage === 'converged') {
      progress = 1;
    } else if (this.stage === 'refining') {
      const sampleFraction = this.refineIter / Math.max(1, this.currentTargetIter);
      progress = Math.min(0.995, (this.completedSamples + sampleFraction) / targetSSAA);
    }

    return {
      stage: this.stage,
      fps: this.fpsDisplay,
      frameMs: this.emaFrameMs,
      activeWidth: this.activeWidth,
      activeHeight: this.activeHeight,
      fullWidth: this.fullWidth,
      fullHeight: this.fullHeight,
      interactiveScale: this.getEffectiveInteractiveScale(),
      completedSamples: this.completedSamples,
      targetSamples: targetSSAA,
      currentIter:
        this.stage === 'interactive'
          ? this.currentInteractiveIter
          : this.stage === 'converged'
            ? this.currentTargetIter
            : this.completedSamples > 0
              ? this.currentTargetIter
              : this.refineIter,
      targetIter: this.currentTargetIter,
      progress,
      precisionMode: precMode,
      precisionBits: precMode === 'perturb' ? camera.precisionBits : precMode === 'df64' ? 48 : 32,
      gpuName: this.gpuName,
    };
  }

  private encodeDisplayPass(
    encoder: GPUCommandEncoder,
    gradientState: GradientState,
    evaluatedGradient: EvaluatedGradient,
  ): void {
    if (!this.displayBindGroup) return;

    this.writeDisplayUniforms(gradientState, evaluatedGradient);

    const textureView = this.context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      label: 'display-render',
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.004, g: 0.005, b: 0.012, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    renderPass.setPipeline(this.displayPipeline);
    renderPass.setBindGroup(0, this.displayBindGroup);
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();
  }

  /** Synchronously re-render the current swapchain frame and capture a PNG blob */
  public async captureSnapshot(): Promise<Blob | null> {
    if (!this.lastGradientState || !this.lastEvaluatedGradient) return null;

    const encoder = this.device.createCommandEncoder({ label: 'snapshot-frame' });
    this.encodeDisplayPass(encoder, this.lastGradientState, this.lastEvaluatedGradient);
    this.device.queue.submit([encoder.finish()]);

    return new Promise((resolve) => {
      this.canvas.toBlob((blob) => resolve(blob), 'image/png');
    });
  }
}
