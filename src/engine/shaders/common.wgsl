// Shared uniforms, bindings, df64 (double-float) math, rescaled perturbation + Zhuoran rebasing, and smooth escape math

struct ComputeUniforms {
  // 0: Viewport center in df64: (cx_hi, cx_lo, cy_hi, cy_lo)
  center: vec4<f32>,
  // 1: Complex step per pixel in df64: (step_hi, step_lo, bailout2, df_one)
  scale: vec4<f32>,
  // 2: (width, height, jitter_x, jitter_y)
  viewport: vec4<f32>,
  // 3: (start_iter, chunk_iter, max_iter, sample_index)
  iter_params: vec4<u32>,
  // 4: (precision_mode: 0=fp32, 1=df64, 2=perturb; is_first_chunk; active_stride; is_julia)
  mode_params: vec4<u32>,
  // 5: Julia parameter c in df64: (c_re_hi, c_re_lo, c_im_hi, c_im_lo)
  julia_c: vec4<f32>,
  // 6: Perturbation rescaled coordinates at 2^scale_exp: (offset_x, offset_y, step_scaled, 0)
  perturb_offset: vec4<f32>,
  // 7: Perturbation metadata: (ref_len, scale_exp, reserved, reserved)
  perturb_meta: vec4<i32>,
};

@group(0) @binding(0) var<uniform> u_comp: ComputeUniforms;
@group(0) @binding(1) var<storage, read_write> samples0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> samples1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> state_z: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> state_meta: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> ref_orbit: array<vec2<f32>>;

// --- Emulated Double-Float (df64 = vec2<f32>(hi, lo)) ---
// Provides ~48 bits of mantissa precision for deep zooms up to ~10^11.
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

// --- Rescaled Perturbation Theory + Zhuoran Rebasing ---
// Supports Mandelbrot and Julia ultra-deep zooms past 10^140 without f32 underflow.

struct PerturbResult {
  px: f32,
  py: f32,
  ref_idx: u32,
  pexp: i32,
  iter: u32,
  escaped: u32,
  mag2: f32,
};

fn fractal_iterate_perturb(
  dcx: f32,
  dcy: f32,
  px_init: f32,
  py_init: f32,
  ref_idx_init: u32,
  pexp_init: i32,
  start_iter: u32,
  steps: u32,
  bailout2: f32,
  is_first: u32
) -> PerturbResult {
  let ref_len = u32(max(u_comp.perturb_meta.x, 1));
  let scale_exp = u_comp.perturb_meta.y;
  let is_julia = u_comp.mode_params.w;
  let z0 = ref_orbit[0];

  var px = px_init;
  var py = py_init;
  var ref_idx = ref_idx_init;
  var pexp = pexp_init;

  if (is_first == 1u) {
    ref_idx = 0u;
    pexp = scale_exp;
    if (is_julia == 1u) {
      px = dcx;
      py = dcy;
    } else {
      px = 0.0;
      py = 0.0;
    }
  }

  var iter = start_iter;
  var escaped = 0u;
  var mag2 = 0.0;

  for (var i = 0u; i < steps; i = i + 1u) {
    let Z = ref_orbit[min(ref_idx, ref_len)];

    // Linear term: 2 * Z * delta (kept at exponent pexp)
    let lin_x = 2.0 * (Z.x * px - Z.y * py);
    let lin_y = 2.0 * (Z.x * py + Z.y * px);

    // Quadratic term: delta^2 * 2^pexp
    var sq_x = 0.0;
    var sq_y = 0.0;
    if (pexp >= -110) {
      let s = ldexp(1.0, min(pexp, 60));
      sq_x = (px * px - py * py) * s;
      sq_y = (2.0 * px * py) * s;
    }

    var nx = lin_x + sq_x;
    var ny = lin_y + sq_y;

    // Constant pixel delta term (Mandelbrot only): + dc * 2^(scale_exp - pexp)
    if (is_julia == 0u) {
      let exp_diff = scale_exp - pexp;
      if (exp_diff > 30) {
        // |delta| < 2^-30 * |dc|, so next delta is dc at scale_exp
        nx = dcx;
        ny = dcy;
        pexp = scale_exp;
      } else if (exp_diff >= -30) {
        let sc = ldexp(1.0, exp_diff);
        nx = nx + dcx * sc;
        ny = ny + dcy * sc;
      }
    }

    px = nx;
    py = ny;
    ref_idx = ref_idx + 1u;
    iter = iter + 1u;

    // Exact power-of-two renormalization so mantissa stays centered in f32 range
    let amax = max(abs(px), abs(py));
    if (amax > 0.0 && (amax > 16.0 || amax < 0.0625)) {
      let fr = frexp(amax);
      px = ldexp(px, -fr.exp);
      py = ldexp(py, -fr.exp);
      pexp = pexp + fr.exp;
    }

    let Zn = ref_orbit[min(ref_idx, ref_len)];

    // Bailout test & Zhuoran rebasing
    if (pexp >= -50) {
      let sc = ldexp(1.0, min(pexp, 60));
      let dx_real = px * sc;
      let dy_real = py * sc;
      let fx = Zn.x + dx_real;
      let fy = Zn.y + dy_real;
      let full_mag2 = fx * fx + fy * fy;
      if (full_mag2 > bailout2) {
        escaped = 1u;
        mag2 = full_mag2;
        break;
      }
      let d_mag2 = dx_real * dx_real + dy_real * dy_real;
      let rx = fx - z0.x;
      let ry = fy - z0.y;
      let r_mag2 = rx * rx + ry * ry;
      let should_rebase = select(
        (r_mag2 <= d_mag2 || ref_idx >= ref_len),
        (ref_idx >= ref_len || (pexp >= -8 && r_mag2 < d_mag2 * 0.25)),
        is_julia == 1u
      );
      if (should_rebase) {
        px = rx;
        py = ry;
        pexp = 0;
        ref_idx = 0u;
        let fa = max(abs(px), abs(py));
        if (fa > 0.0) {
          let fr = frexp(fa);
          px = ldexp(px, -fr.exp);
          py = ldexp(py, -fr.exp);
          pexp = fr.exp;
        }
      }
    } else {
      let zn_mag2 = Zn.x * Zn.x + Zn.y * Zn.y;
      if (zn_mag2 > bailout2) {
        escaped = 1u;
        mag2 = zn_mag2;
        break;
      }
      let zn_max = max(abs(Zn.x), abs(Zn.y));
      if (ref_idx >= ref_len) {
        let rx = Zn.x - z0.x;
        let ry = Zn.y - z0.y;
        let ra = max(abs(rx), abs(ry));
        if (ra > 0.0) {
          let fr = frexp(ra);
          px = ldexp(rx, -fr.exp);
          py = ldexp(ry, -fr.exp);
          pexp = fr.exp;
        }
        ref_idx = 0u;
      } else if (is_julia == 0u && zn_max == 0.0) {
        ref_idx = 0u;
      } else if (is_julia == 0u && zn_max < 1e-14 && -pexp <= 120) {
        let zn_sx = ldexp(Zn.x, -pexp);
        let zn_sy = ldexp(Zn.y, -pexp);
        let fx = zn_sx + px;
        let fy = zn_sy + py;
        if (fx * fx + fy * fy <= px * px + py * py) {
          px = fx;
          py = fy;
          ref_idx = 0u;
          let fa = max(abs(px), abs(py));
          if (fa > 0.0) {
            let fr = frexp(fa);
            px = ldexp(px, -fr.exp);
            py = ldexp(py, -fr.exp);
            pexp = pexp + fr.exp;
          }
        }
      }
    }
  }

  return PerturbResult(px, py, ref_idx, pexp, iter, escaped, mag2);
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
