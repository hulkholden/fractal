// Quadratic Julia fractal kernel: z_{n+1} = z_n^2 + c, z_0 = pixel coordinate

struct IterResult {
  zx: vec2<f32>,
  zy: vec2<f32>,
  iter: u32,
  escaped: u32,
  mag2: f32,
};

fn fractal_early_interior(_px: f32, _py: f32) -> bool {
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
  var zx = select(zx_init, px, is_first == 1u);
  var zy = select(zy_init, py, is_first == 1u);
  let cx = u_comp.julia_c.x;
  let cy = u_comp.julia_c.z;
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
    let next_zy = 2.0 * zx * zy + cy;
    zx = zx2 - zy2 + cx;
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
  var zx = select(zx_init, px, is_first == 1u);
  var zy = select(zy_init, py, is_first == 1u);
  let cx = vec2<f32>(u_comp.julia_c.x, u_comp.julia_c.y);
  let cy = vec2<f32>(u_comp.julia_c.z, u_comp.julia_c.w);
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
    let next_zy = df_add(df_mul2(zxy), cy);
    zx = df_add(df_sub(zx2, zy2), cx);
    zy = next_zy;
    iter = iter + 1u;
  }

  return IterResult(zx, zy, iter, escaped, mag2);
}
