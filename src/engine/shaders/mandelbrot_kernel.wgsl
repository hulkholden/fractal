// Mandelbrot fractal kernel: z_{n+1} = z_n^2 + c, z_0 = 0

struct IterResult {
  zx: vec2<f32>,
  zy: vec2<f32>,
  iter: u32,
  escaped: u32,
  mag2: f32,
};

// Fast analytical check for the main cardioid and period-2 bulb.
// Skips the entire iteration loop for interior points in these regions.
fn fractal_early_interior(cx: f32, cy: f32) -> bool {
  let xm = cx - 0.25;
  let y2 = cy * cy;
  let q = xm * xm + y2;
  if (q * (q + xm) <= 0.25 * y2) {
    return true;
  }
  let xp1 = cx + 1.0;
  if (xp1 * xp1 + y2 <= 0.0625) {
    return true;
  }
  return false;
}

fn fractal_iterate_f32(
  px: f32,
  py: f32,
  zx_init: f32,
  zy_init: f32,
  start_iter: u32,
  steps: u32,
  bailout2: f32,
  is_first: u32
) -> IterResult {
  var zx = select(zx_init, 0.0, is_first == 1u);
  var zy = select(zy_init, 0.0, is_first == 1u);
  var iter = start_iter;
  let end_iter = start_iter + steps;
  var mag2 = zx * zx + zy * zy;
  var escaped = 0u;

  for (var i = 0u; i < steps; i = i + 1u) {
    let zx2 = zx * zx;
    let zy2 = zy * zy;
    mag2 = zx2 + zy2;
    if (mag2 > bailout2) {
      escaped = 1u;
      break;
    }
    let next_zy = 2.0 * zx * zy + py;
    zx = zx2 - zy2 + px;
    zy = next_zy;
    iter = iter + 1u;
    if (iter >= end_iter) {
      break;
    }
  }

  return IterResult(vec2<f32>(zx, 0.0), vec2<f32>(zy, 0.0), iter, escaped, mag2);
}

fn fractal_iterate_df64(
  px: vec2<f32>,
  py: vec2<f32>,
  zx_init: vec2<f32>,
  zy_init: vec2<f32>,
  start_iter: u32,
  steps: u32,
  bailout2: f32,
  is_first: u32
) -> IterResult {
  var zx = select(zx_init, vec2<f32>(0.0, 0.0), is_first == 1u);
  var zy = select(zy_init, vec2<f32>(0.0, 0.0), is_first == 1u);
  var iter = start_iter;
  var mag2 = zx.x * zx.x + zy.x * zy.x;
  var escaped = 0u;

  for (var i = 0u; i < steps; i = i + 1u) {
    let zx2 = df_sqr(zx);
    let zy2 = df_sqr(zy);
    mag2 = zx2.x + zy2.x;
    if (mag2 > bailout2) {
      escaped = 1u;
      break;
    }
    let zxy = df_mul(zx, zy);
    let next_zy = df_add(df_mul2(zxy), py);
    zx = df_add(df_sub(zx2, zy2), px);
    zy = next_zy;
    iter = iter + 1u;
  }

  return IterResult(zx, zy, iter, escaped, mag2);
}
