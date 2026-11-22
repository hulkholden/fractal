// Shared uniforms, bindings, df64 (double-float) arithmetic, and smooth escape math

struct ComputeUniforms {
  // Viewport center in df64: (cx_hi, cx_lo, cy_hi, cy_lo)
  center: vec4<f32>,
  // Complex step per pixel in df64: (step_hi, step_lo, bailout2, reserved)
  scale: vec4<f32>,
  // (width, height, jitter_x, jitter_y)
  viewport: vec4<f32>,
  // (start_iter, chunk_iter, max_iter, sample_index)
  iter_params: vec4<u32>,
  // (use_df64, is_first_chunk, active_stride, reserved)
  mode_params: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u_comp: ComputeUniforms;
@group(0) @binding(1) var<storage, read_write> samples0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> samples1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> state_z: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> state_meta: array<vec2<u32>>;

// --- Emulated Double-Float (df64 = vec2<f32>(hi, lo)) ---
// Provides ~48 bits of mantissa precision for deep zooms past 10^12.
// Uses hardware fma() with runtime uniform u_comp.scale.w (= 1.0) so GPU
// fast-math compilers (notably Apple Metal) cannot reassociate or collapse
// Knuth/Dekker error-free transformations.

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let one = u_comp.scale.w;
  let s = fma(a, one, b);
  let v = fma(s, one, -a);
  let e = fma(b, one, -v);
  return vec2<f32>(s, e);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let one = u_comp.scale.w;
  let s = fma(a, one, b);
  let v = fma(s, one, -a);
  let sv = fma(s, one, -v);
  let e = fma(a, one, -sv) + fma(b, one, -v);
  return vec2<f32>(s, e);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let one = u_comp.scale.w;
  let p = fma(a * b, one, 0.0);
  let e = fma(a, b, -p);
  return vec2<f32>(p, e);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let c = s.y + t.x;
  let v = quick_two_sum(s.x, c);
  let w = t.y + v.y;
  return quick_two_sum(v.x, w);
}

fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return df_add(a, vec2<f32>(-b.x, -b.y));
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p = two_prod(a.x, b.x);
  let t = p.y + (a.x * b.y + a.y * b.x);
  return quick_two_sum(p.x, t);
}

fn df_mul_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
  let p = two_prod(a.x, b);
  let t = p.y + a.y * b;
  return quick_two_sum(p.x, t);
}

fn df_sqr(a: vec2<f32>) -> vec2<f32> {
  let p = two_prod(a.x, a.x);
  let t = p.y + 2.0 * a.x * a.y;
  return quick_two_sum(p.x, t);
}

fn df_mul2(a: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * 2.0, a.y * 2.0);
}

// Smooth renormalized escape count (continuous across iteration boundaries with bailout >= 256)
fn compute_smooth_iter(iter: u32, mag2: f32) -> f32 {
  let log_zn = 0.5 * log2(max(mag2, 4.0));
  let nu = log2(max(log_zn, 1.0));
  return max(0.0, f32(iter) + 1.0 - nu);
}

fn write_sample(idx: u32, sample_idx: u32, val: f32) {
  switch sample_idx {
    case 0u: { samples0[idx].x = val; }
    case 1u: { samples0[idx].y = val; }
    case 2u: { samples0[idx].z = val; }
    case 3u: { samples0[idx].w = val; }
    case 4u: { samples1[idx].x = val; }
    case 5u: { samples1[idx].y = val; }
    case 6u: { samples1[idx].z = val; }
    default: { samples1[idx].w = val; }
  }
}
