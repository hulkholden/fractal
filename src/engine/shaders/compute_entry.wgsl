// Compute entry points:
// - main_interactive: zero state-buffer overhead, optimized for locked 60Hz pan/zoom
// - main_refine: stateful chunked iteration + multi-sample SSAA when stationary

@compute @workgroup_size(8, 8)
fn main_interactive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(u_comp.viewport.x);
  let height = u32(u_comp.viewport.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let dx = (f32(gid.x) + 0.5 + u_comp.viewport.z) - 0.5 * u_comp.viewport.x;
  let dy = 0.5 * u_comp.viewport.y - (f32(gid.y) + 0.5 + u_comp.viewport.w);
  let bailout2 = u_comp.scale.z;
  let max_iter = u_comp.iter_params.z;
  let use_df64 = u_comp.mode_params.x;
  let stride = u_comp.mode_params.z;
  let idx = gid.y * stride + gid.x;

  if (use_df64 == 0u) {
    let cx = u_comp.center.x + dx * u_comp.scale.x;
    let cy = u_comp.center.z + dy * u_comp.scale.x;

    if (fractal_early_interior(cx, cy)) {
      samples0[idx].x = -1.0;
      return;
    }

    let res = fractal_iterate_f32(cx, cy, 0.0, 0.0, 0u, max_iter, bailout2);
    if (res.escaped == 1u) {
      samples0[idx].x = compute_smooth_iter(res.iter, res.mag2);
    } else {
      samples0[idx].x = -1.0;
    }
  } else {
    let step_df = vec2<f32>(u_comp.scale.x, u_comp.scale.y);
    let cx = df_add(vec2<f32>(u_comp.center.x, u_comp.center.y), df_mul_f32(step_df, dx));
    let cy = df_add(vec2<f32>(u_comp.center.z, u_comp.center.w), df_mul_f32(step_df, dy));

    if (fractal_early_interior(cx.x, cy.x)) {
      samples0[idx].x = -1.0;
      return;
    }

    let res = fractal_iterate_df64(
      cx,
      cy,
      vec2<f32>(0.0, 0.0),
      vec2<f32>(0.0, 0.0),
      0u,
      max_iter,
      bailout2
    );
    if (res.escaped == 1u) {
      samples0[idx].x = compute_smooth_iter(res.iter, res.mag2);
    } else {
      samples0[idx].x = -1.0;
    }
  }
}

@compute @workgroup_size(8, 8)
fn main_refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(u_comp.viewport.x);
  let height = u32(u_comp.viewport.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let stride = u_comp.mode_params.z;
  let idx = gid.y * stride + gid.x;
  let is_first = u_comp.mode_params.y;
  let sample_idx = u_comp.iter_params.w;
  let chunk_iter = u_comp.iter_params.y;
  let max_iter = u_comp.iter_params.z;
  let bailout2 = u_comp.scale.z;
  let use_df64 = u_comp.mode_params.x;

  var zx = vec2<f32>(0.0, 0.0);
  var zy = vec2<f32>(0.0, 0.0);
  var iter = 0u;

  if (is_first == 0u) {
    let cell_meta = state_meta[idx];
    if (cell_meta.y != 0u || cell_meta.x >= max_iter) {
      return;
    }
    let z = state_z[idx];
    zx = vec2<f32>(z.x, z.y);
    zy = vec2<f32>(z.z, z.w);
    iter = cell_meta.x;
  }

  let dx = (f32(gid.x) + 0.5 + u_comp.viewport.z) - 0.5 * u_comp.viewport.x;
  let dy = 0.5 * u_comp.viewport.y - (f32(gid.y) + 0.5 + u_comp.viewport.w);
  let remaining = max_iter - iter;
  let steps = min(chunk_iter, remaining);

  var res: IterResult;

  if (use_df64 == 0u) {
    let cx = u_comp.center.x + dx * u_comp.scale.x;
    let cy = u_comp.center.z + dy * u_comp.scale.x;

    if (is_first == 1u && fractal_early_interior(cx, cy)) {
      state_meta[idx] = vec2<u32>(max_iter, 2u);
      write_sample(idx, sample_idx, -1.0);
      return;
    }

    res = fractal_iterate_f32(cx, cy, zx.x, zy.x, iter, steps, bailout2);
  } else {
    let step_df = vec2<f32>(u_comp.scale.x, u_comp.scale.y);
    let cx = df_add(vec2<f32>(u_comp.center.x, u_comp.center.y), df_mul_f32(step_df, dx));
    let cy = df_add(vec2<f32>(u_comp.center.z, u_comp.center.w), df_mul_f32(step_df, dy));

    if (is_first == 1u && fractal_early_interior(cx.x, cy.x)) {
      state_meta[idx] = vec2<u32>(max_iter, 2u);
      write_sample(idx, sample_idx, -1.0);
      return;
    }

    res = fractal_iterate_df64(cx, cy, zx, zy, iter, steps, bailout2);
  }

  if (res.escaped == 1u) {
    state_meta[idx] = vec2<u32>(res.iter, 1u);
    write_sample(idx, sample_idx, compute_smooth_iter(res.iter, res.mag2));
  } else {
    state_z[idx] = vec4<f32>(res.zx.x, res.zx.y, res.zy.x, res.zy.y);
    state_meta[idx] = vec2<u32>(res.iter, 0u);
    write_sample(idx, sample_idx, -1.0);
  }
}
