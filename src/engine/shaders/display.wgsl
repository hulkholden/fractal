// Fullscreen display shader:
// - Runs at 60Hz independent of compute refinement
// - Periodic Oklab Catmull-Rom gradient interpolation + live palette morphing & hue drift
// - Linear-light supersampled resolve (1x..8x SSAA) and edge-clean bilinear interactive upscale

struct DisplayUniforms {
  // (canvas_w, canvas_h, active_w, active_h)
  dims: vec4<f32>,
  // (color_phase, color_density, morph_blend, hue_drift)
  color_params: vec4<f32>,
  // (sample_count, upscale_bilinear, reserved0, reserved1)
  render_flags: vec4<u32>,
  // 8 periodic Oklab color stops (L, a, b, 0) for Palette A and Palette B
  palette_a: array<vec4<f32>, 8>,
  palette_b: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> u_disp: DisplayUniforms;
@group(0) @binding(1) var<storage, read> samples0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> samples1: array<vec4<f32>>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  // Fullscreen triangle covering NDC [-1, 1] x [-1, 1]
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, 1.0)
  );
  let p = positions[vid];
  var out: VsOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  // UV (0,0) top-left -> (1,1) bottom-right matching compute pixel coords
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// Convert Oklab (L, a, b) to linear sRGB (Björn Ottosson)
fn oklab_to_linear_srgb(lab: vec3<f32>) -> vec3<f32> {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;

  let r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn blended_stop(idx: u32, morph: f32) -> vec3<f32> {
  let a = u_disp.palette_a[idx].xyz;
  let b = u_disp.palette_b[idx].xyz;
  return mix(a, b, morph);
}

// Smooth periodic Catmull-Rom spline in Oklab space
fn catmull_rom(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (
    (2.0 * p1) +
    (-p0 + p2) * t +
    (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
    (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
  );
}

fn eval_palette_linear(mu: f32) -> vec3<f32> {
  // Interior of the set: deep velvety obsidian
  if (mu < 0.0) {
    return vec3<f32>(0.0025, 0.0035, 0.009);
  }

  let phase = u_disp.color_params.x;
  let density = u_disp.color_params.y;
  let morph = u_disp.color_params.z;
  let hue_drift = u_disp.color_params.w;

  // Perceptually balanced log + gentle sqrt mapping:
  // keeps wide sweeping ribbons outside AND crisp color detail in deep spirals
  let mapped = log(mu + 1.0) * 0.42 + sqrt(mu) * 0.045;
  let t = fract(mapped * density - phase);

  let scaled = t * 8.0;
  let i1 = u32(floor(scaled)) & 7u;
  let f = fract(scaled);

  let p0 = blended_stop((i1 + 7u) & 7u, morph);
  let p1 = blended_stop(i1, morph);
  let p2 = blended_stop((i1 + 1u) & 7u, morph);
  let p3 = blended_stop((i1 + 2u) & 7u, morph);

  var lab = catmull_rom(p0, p1, p2, p3, f);

  // Subtle perceptual hue rotation in Oklab (a, b) plane when Living Morph is active
  if (abs(hue_drift) > 0.0001) {
    let ch = cos(hue_drift);
    let sh = sin(hue_drift);
    let a_rot = lab.y * ch - lab.z * sh;
    let b_rot = lab.y * sh + lab.z * ch;
    lab.y = a_rot;
    lab.z = b_rot;
  }

  // Softly prevent Catmull-Rom lightness overshoot
  lab.x = clamp(lab.x, 0.0, 0.985);
  return oklab_to_linear_srgb(lab);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(1e-6)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let active_w = u_disp.dims.z;
  let active_h = u_disp.dims.w;
  let stride = u32(active_w);
  let max_x = u32(max(active_w - 1.0, 0.0));
  let max_y = u32(max(active_h - 1.0, 0.0));

  var linear_rgb = vec3<f32>(0.0);

  if (u_disp.render_flags.y == 1u) {
    // Interactive downscaled mode: evaluate palette per texel, then bilinearly blend linear RGB
    // (avoids interior/exterior boundary halo artifacts!)
    let fx = clamp(in.uv.x * active_w - 0.5, 0.0, active_w - 1.0);
    let fy = clamp(in.uv.y * active_h - 0.5, 0.0, active_h - 1.0);

    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, max_x);
    let y1 = min(y0 + 1u, max_y);
    let tx = fract(fx);
    let ty = fract(fy);

    let c00 = eval_palette_linear(samples0[y0 * stride + x0].x);
    let c10 = eval_palette_linear(samples0[y0 * stride + x1].x);
    let c01 = eval_palette_linear(samples0[y1 * stride + x0].x);
    let c11 = eval_palette_linear(samples0[y1 * stride + x1].x);

    let top = mix(c00, c10, tx);
    let bot = mix(c01, c11, tx);
    linear_rgb = mix(top, bot, ty);
  } else {
    // Native 1:1 resolution with 1x..8x supersampled linear-light resolve
    let px = min(u32(floor(in.uv.x * active_w)), max_x);
    let py = min(u32(floor(in.uv.y * active_h)), max_y);
    let idx = py * stride + px;
    let count = max(u_disp.render_flags.x, 1u);

    let s0 = samples0[idx];
    var acc = eval_palette_linear(s0.x);
    if (count >= 2u) { acc += eval_palette_linear(s0.y); }
    if (count >= 3u) { acc += eval_palette_linear(s0.z); }
    if (count >= 4u) { acc += eval_palette_linear(s0.w); }

    if (count >= 5u) {
      let s1 = samples1[idx];
      acc += eval_palette_linear(s1.x);
      if (count >= 6u) { acc += eval_palette_linear(s1.y); }
      if (count >= 7u) { acc += eval_palette_linear(s1.z); }
      if (count >= 8u) { acc += eval_palette_linear(s1.w); }
    }

    linear_rgb = acc / f32(count);
  }

  let srgb = linear_to_srgb(clamp(linear_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(srgb, 1.0);
}
