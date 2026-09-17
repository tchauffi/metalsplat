// Tile-based front-to-back alpha-compositing rasterizer for 2D gaussian
// splats ("surfels"), forward + backward. Mirrors
// metalsplat.reference.rasterize_2dgs_ref.rasterize_gaussians_2dgs exactly
// in the math; that module is the oracle these kernels are tested against.
// Same dispatch model as rasterize.metal (threadgroup-per-tile,
// thread-per-pixel), but per-pixel alpha comes from the exact ray-splat
// intersection instead of an analytic screen-space conic, and the
// rasterizer additionally produces differentiable depth/normal/distortion
// outputs (3DGS's rasterize.metal only differentiates image).
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Far plane used only to normalize depth for the distortion regularizer.
// Matches diff-surfel-rasterization's `__device__ const float far_n = 100.0`;
// see rasterize_2dgs_ref's module docstring for why the distortion term is
// accumulated on normalized depth `m` rather than raw metric `z_hit`.
constant float DISTORTION_FAR = 100.0;

// m = far/(far - near) * (1 - near/z), mapping z in [near, far] to [0, 1].
// Callers only ever evaluate this where alpha > 0, which already implies
// z > near (see ray_splat_alpha), so `near/z` is always finite here.
inline float normalized_depth(float z, float near) {
    return DISTORTION_FAR / (DISTORTION_FAR - near) * (1.0 - near / z);
}

// d(normalized_depth)/dz, for chaining the distortion gradient back onto
// the metric intersection depth in the backward pass.
inline float d_normalized_depth_dz(float z, float near) {
    return DISTORTION_FAR / (DISTORTION_FAR - near) * near / (z * z);
}

// Resolves the ray-splat intersection for one (gaussian, pixel) pair and
// returns its alpha, mirroring rasterize_2dgs_ref's per-gaussian loop body
// exactly (see project_2dgs_ref's module docstring for the transform/M
// derivation). Used by both forward and backward so the two can never
// silently diverge. `hu`/`hv`/`wloc`/`u`/`v`/`degenerate`/`uv_active`/
// `z_hit`/`raw` are handed back via out-params (backward needs them to
// build the exact same gradients forward used to build the output).
inline float ray_splat_alpha(
    float3 row0, float3 row1, float3 row2,
    float2 mean2d, float px, float py,
    float opacity, float near, float filter_size, float depth_fallback,
    thread float3& hu_out, thread float3& hv_out, thread float& wloc_out,
    thread float& u_out, thread float& v_out, thread bool& degenerate_out,
    thread bool& uv_active_out, thread float& z_hit_out, thread float& raw_out,
    thread float& rho_out)
{
    float3 hu = row0 - px * row2;
    float3 hv = row1 - py * row2;
    float3 cr = cross(hu, hv);
    float wloc = cr.z;
    bool degenerate = fabs(wloc) < 1e-9;

    float u = 0.0, v = 0.0;
    if (!degenerate) {
        u = cr.x / wloc;
        v = cr.y / wloc;
    }

    // Screen-space low-pass fallback, the 2DGS paper's own anti-aliasing
    // compromise for near-edge-on splats. `filter_size` is the filter's
    // standard deviation in pixels, matching diff-surfel-rasterization's
    // `FilterSize = 0.707106 // sqrt(2)/2` / `FilterInvSquare = 2.0f`.
    // Sub-pixel by design -- and since `rho` below is a min(), it is a
    // lower bound on every splat's footprint, so widening it stops the
    // optimizer sharpening detail by shrinking a splat. Deliberately
    // *not* the projection's covariance dilation, a px^2 quantity for the
    // EWA/tile-culling bound: sharing one number made neither tunable
    // without breaking the other. See rasterize_2dgs_ref's
    // DEFAULT_FILTER_SIZE.
    float dx = px - mean2d.x, dy = py - mean2d.y;
    float filter_var = filter_size * filter_size;
    float rho_screen = (dx * dx + dy * dy) / filter_var;
    float rho_uv = degenerate ? (rho_screen + 1.0) : (u * u + v * v);
    bool uv_active = rho_uv <= rho_screen;
    float rho = uv_active ? rho_uv : rho_screen;

    // Only trust the ray-splat intersection depth when it actually won the
    // rho min() above -- near that boundary `wloc` can be small-but-not-
    // quite-degenerate, so u/v (scaling as 1/wloc) can be enormous and
    // numerically unstable even though they don't affect alpha there.
    // Using them for z_hit anyway would amplify ordinary GPU/CPU float32
    // differences into large, spurious depth disagreements. See
    // rasterize_2dgs_ref's matching comment.
    float z_hit = (degenerate || !uv_active) ? depth_fallback
                                              : (row2.x * u + row2.y * v + row2.z);

    float raw = opacity * exp(-0.5 * rho);
    float alpha = min(0.99, raw);
    if (z_hit <= near) alpha = 0.0;

    hu_out = hu; hv_out = hv; wloc_out = wloc;
    u_out = u; v_out = v; degenerate_out = degenerate;
    uv_active_out = uv_active; z_hit_out = z_hit; raw_out = raw; rho_out = rho;
    return alpha;
}

kernel void rasterize_2dgs_forward(
    device const float* means2d,             // (N,2)
    device const float* transform,            // (N,9) row-major 3x3
    device const float* normal,                // (N,3)
    device const float* opacities,              // (N,)
    device const float* colors,                  // (N,3)
    device const float* depths,                   // (N,) gaussian mean depth (degenerate fallback)
    device const int* sorted_ids,                  // (M,)
    device const int* tile_bins,                    // (num_tiles,2)
    constant int& tiles_x,
    constant int& img_width,
    constant int& img_height,
    constant int& tile_size,
    constant float& near,
    constant float& filter_size,
    constant float* background,
    device float* out_image,                          // (H,W,3)
    device float* out_depth,                            // (H,W)
    device float* out_normal,                            // (H,W,3)
    device float* out_distortion,                         // (H,W)
    device float* out_dist_m1,                             // (H,W) sum of weight*m, backward-only
    device float* out_dist_m2,                              // (H,W) sum of weight*m*m, backward-only
    device float* out_final_T,                               // (H,W)
    device int* out_last_contributor,                        // (H,W)
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]])
{
    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    if (px >= uint(img_width) || py >= uint(img_height)) return;

    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    int end = tile_bins[tile_id * 2 + 1];

    float pcx = float(px) + 0.5, pcy = float(py) + 0.5;
    float3 accum = float3(0.0);
    float accum_depth = 0.0;
    float3 accum_normal = float3(0.0);
    float T = 1.0;
    int last_contributor = -1;

    // Running zeroth/first/second moments of the per-pixel weight
    // distribution over normalized depth, for the distortion loss -- the
    // exact same accumulation this pixel's backward pass will need to
    // invert (see rasterize_2dgs_backward's module-level derivation).
    float A_dist = 0.0, M1_dist = 0.0, M2_dist = 0.0, L_dist = 0.0;

    for (int idx = start; idx < end; idx++) {
        int gid = sorted_ids[idx];
        float3 row0 = float3(transform[gid * 9 + 0], transform[gid * 9 + 1], transform[gid * 9 + 2]);
        float3 row1 = float3(transform[gid * 9 + 3], transform[gid * 9 + 4], transform[gid * 9 + 5]);
        float3 row2 = float3(transform[gid * 9 + 6], transform[gid * 9 + 7], transform[gid * 9 + 8]);
        float2 mean2d = float2(means2d[gid * 2 + 0], means2d[gid * 2 + 1]);

        float3 hu, hv; float wloc, u, v, z_hit, raw, rho; bool degenerate, uv_active;
        float alpha = ray_splat_alpha(
            row0, row1, row2, mean2d, pcx, pcy, opacities[gid], near, filter_size, depths[gid],
            hu, hv, wloc, u, v, degenerate, uv_active, z_hit, raw, rho);

        if (alpha < (1.0 / 255.0)) continue;
        float test_T = T * (1.0 - alpha);
        if (test_T < 1e-4) break;

        float3 color = float3(colors[gid * 3 + 0], colors[gid * 3 + 1], colors[gid * 3 + 2]);
        float3 normal_g = float3(normal[gid * 3 + 0], normal[gid * 3 + 1], normal[gid * 3 + 2]);
        float weight = T * alpha;

        accum += weight * color;
        accum_depth += weight * z_hit;
        accum_normal += weight * normal_g;

        // Mip-NeRF-360/2DGS distortion regularizer
        // sum_{i<j} w_i*w_j*(m_i - m_j)^2, accumulated in one front-to-back
        // pass via the prefix-moment expansion (see rasterize_2dgs_ref's
        // module docstring, including why the *squared* difference is
        // load-bearing and the cheaper signed variant is not a regularizer
        // at all). Runs on normalized depth `m`, not metric z_hit --
        // `accum_depth` above keeps the metric units.
        float m = normalized_depth(z_hit, near);
        L_dist += weight * (m * m * A_dist - 2.0 * m * M1_dist + M2_dist);
        A_dist += weight;
        M1_dist += weight * m;
        M2_dist += weight * m * m;

        T = test_T;
        last_contributor = idx;
    }

    accum += T * float3(background[0], background[1], background[2]);

    uint pixel_idx = py * uint(img_width) + px;
    out_image[pixel_idx * 3 + 0] = accum.x;
    out_image[pixel_idx * 3 + 1] = accum.y;
    out_image[pixel_idx * 3 + 2] = accum.z;
    out_depth[pixel_idx] = accum_depth;
    out_normal[pixel_idx * 3 + 0] = accum_normal.x;
    out_normal[pixel_idx * 3 + 1] = accum_normal.y;
    out_normal[pixel_idx * 3 + 2] = accum_normal.z;
    out_distortion[pixel_idx] = L_dist;
    out_dist_m1[pixel_idx] = M1_dist;
    out_dist_m2[pixel_idx] = M2_dist;
    out_final_T[pixel_idx] = T;
    out_last_contributor[pixel_idx] = last_contributor;
}

// Backward.
//
// image/depth/normal all use the standard alpha-compositing identity
// dQ/dalpha_i = T_i*(q_i - A_suffix_i) for a linear-in-weights per-pixel
// quantity Q = sum_k w_k*q_k -- same pattern rasterize.metal already uses
// for color, just with q_k = z_hit_k (depth) or q_k = normal_k, each with
// its own running suffix accumulator (A_depth/A_normal), both seeded at 0
// (unlike color's `A`, which seeds at `background` because the forward
// pass adds a T*background residual that depth/normal don't have).
//
// The distortion loss L = sum_{i<j} w_i*w_j*(mu_i - mu_j)^2 (accumulated
// forward via prefix moments A/M1/M2 of w / w*mu / w*mu^2 -- see
// rasterize_2dgs_ref's module docstring) is NOT of that linear form (each
// term depends on *other* gaussians' weights), so it needs its own
// derivation. Differentiating w.r.t. alpha_m through the full
// transmittance chain (T_k depends on alpha_m for every k > m too, not
// just k = m) and collecting every pair (i,j) that alpha_m's perturbation
// touches gives:
//
//   dL/dalpha_m = T_m*Fpre_m
//               + T_m*Gsuf_m*(1-2*alpha_m)/(1-alpha_m)
//               - 2*Lsuf_m/(1-alpha_m)
//               - Cross_m/(1-alpha_m)
//
// where, splitting the contributors at m into a prefix (i < m) and a
// suffix (k > m):
//
//   Fpre_m = sum_{i<m} w_i*(mu_m - mu_i)^2
//          = mu_m^2*Apre - 2*mu_m*M1pre + M2pre
//   Gsuf_m = sum_{k>m} w_k*(mu_m - mu_k)^2
//          = mu_m^2*Asuf - 2*mu_m*M1suf + M2suf
//   Cross_m = sum_{i<m} sum_{k>m} w_i*w_k*(mu_i - mu_k)^2
//           = M2pre*Asuf - 2*M1pre*M1suf + Apre*M2suf
//   Lsuf_m  = L itself computed using only gaussians *after* m
//
// Every piece is an O(1)-update running scalar in the SAME single
// back-to-front pass already used for color/opacity/mean: the prefix
// moments Apre/M1pre/M2pre are decremented from running totals seeded at
// the forward pass's finals (A_total = 1-final_T, M1_total = dist_m1,
// M2_total = dist_m2), and the suffix moments Asuf/M1suf/M2suf plus
// Lsuf (via Lsuf += w_m*Gsuf_m) are incremented as we walk.
//
// `mu_k` here is the *normalized* depth m = far/(far-near)*(1 - near/z_k),
// not the metric z_k (see normalized_depth() above and
// rasterize_2dgs_ref's module docstring for why). That only changes the
// depth-gradient branch: dL/dmu_m = 2*w_m*(mu_m*(Apre+Asuf) -
// (M1pre+M1suf)) (which needs no alpha-chain correction, since depths
// don't affect weights) is chained onto z by dm/dz =
// far/(far-near)*near/z^2. Note the moment running totals must therefore
// be seeded from the forward's `dist_m1`/`dist_m2` (sums of w*m and
// w*m*m), NOT from `out_depth` (sum of w*z, the metric depth map) --
// those are different accumulations.
//
// This closed form was derived and numerically verified against
// torch.autograd on rasterize_2dgs_ref's plain-torch recursion before being
// written here -- do not "simplify" this without re-deriving against that
// oracle.
//
// z_hit/u/v's dependence on the projection stage's per-gaussian transform
// (M, i.e. `row0/row1/row2`) is resolved via the implicit function theorem
// on the two ray-plane equations, rather than differentiating through the
// forward pass's cross-product solve directly (equivalent, but much less
// error-prone to derive and verify): u, v solve
//   F1 = row0.(u,v,1) - px*row2.(u,v,1) = 0
//   F2 = row1.(u,v,1) - py*row2.(u,v,1) = 0
// with Jacobian d(F1,F2)/d(u,v) = [[hu.x,hu.y],[hv.x,hv.y]] (hu, hv as
// already computed for the forward solve; its determinant is exactly
// `wloc`, the same cross-product z-component). The reverse-mode adjoint
// lambda solves J^T @ lambda = (d_u, d_v), then every parameter's gradient
// is -lambda . dF/dparam (dF1/d(row0) = dF2/d(row1) = (u,v,1),
// dF1/d(row2) = -px*(u,v,1), dF2/d(row2) = -py*(u,v,1)).
#define BACKWARD_BATCH 256

kernel void rasterize_2dgs_backward(
    device const float* means2d,             // (N,2)
    device const float* transform,            // (N,9)
    device const float* normal,                // (N,3)
    device const float* opacities,              // (N,)
    device const float* colors,                  // (N,3)
    device const float* depths,                   // (N,)
    device const int* sorted_ids,                  // (M,)
    device const int* tile_bins,                    // (num_tiles,2)
    constant int& tiles_x,
    constant int& img_width,
    constant int& img_height,
    constant int& tile_size,
    constant float& near,
    constant float& filter_size,
    constant float* background,
    device const float* final_T,                       // (H,W), from forward
    device const float* dist_m1,                         // (H,W), from forward, sum of weight*m
    device const float* dist_m2,                          // (H,W), from forward, sum of weight*m*m
    device const int* last_contributor,                    // (H,W), from forward
    device const float* d_out_image,                       // (H,W,3)
    device const float* d_out_depth,                         // (H,W)
    device const float* d_out_normal,                         // (H,W,3)
    device const float* d_out_distortion,                      // (H,W)
    device atomic_float* d_means2d,                              // (N,2)
    device atomic_float* d_transform,                             // (N,9)
    device atomic_float* d_normal,                                 // (N,3)
    device atomic_float* d_opacities,                               // (N,)
    device atomic_float* d_colors,                                   // (N,3)
    device atomic_float* d_means2d_abs,                               // (N,) AbsGS-style densification signal
    device atomic_float* d_pixel_count,                                // (N,) pixels this gaussian contributed to
    device atomic_float* d_depths,                                      // (N,) via the degenerate/screen-space fallback only
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint local_idx [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float3 sh_row0[BACKWARD_BATCH];
    threadgroup float3 sh_row1[BACKWARD_BATCH];
    threadgroup float3 sh_row2[BACKWARD_BATCH];
    threadgroup float2 sh_mean2d[BACKWARD_BATCH];
    threadgroup float sh_opacity[BACKWARD_BATCH];
    threadgroup float3 sh_color[BACKWARD_BATCH];
    threadgroup float3 sh_normal[BACKWARD_BATCH];
    threadgroup float sh_depth_fallback[BACKWARD_BATCH];
    threadgroup int sh_gid[BACKWARD_BATCH];

    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    bool inside = (px < uint(img_width)) && (py < uint(img_height));

    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    int end = tile_bins[tile_id * 2 + 1];

    uint pixel_idx = inside ? (py * uint(img_width) + px) : 0;
    int last = inside ? last_contributor[pixel_idx] : -1;
    bool active = inside && (last >= 0);

    float pcx = float(px) + 0.5, pcy = float(py) + 0.5;

    float3 d_C = active ? float3(d_out_image[pixel_idx*3+0], d_out_image[pixel_idx*3+1], d_out_image[pixel_idx*3+2]) : float3(0.0);
    float d_Depth = active ? d_out_depth[pixel_idx] : 0.0;
    float3 d_Normal = active ? float3(d_out_normal[pixel_idx*3+0], d_out_normal[pixel_idx*3+1], d_out_normal[pixel_idx*3+2]) : float3(0.0);
    float d_Dist = active ? d_out_distortion[pixel_idx] : 0.0;

    float T = active ? final_T[pixel_idx] : 0.0;
    float3 A_color = float3(background[0], background[1], background[2]);
    float A_depth = 0.0;
    float3 A_normal = float3(0.0);

    // Distortion running state: A_run/M1_run/M2_run = prefix moments
    // through the current index inclusive (seeded at the forward totals,
    // decremented as we walk back-to-front); Asuf/M1suf/M2suf/Lsuf =
    // suffix strictly after the current index (seeded at 0, incremented
    // as we walk).
    float A_run = active ? (1.0 - T) : 0.0;
    float M1_run = active ? dist_m1[pixel_idx] : 0.0;
    float M2_run = active ? dist_m2[pixel_idx] : 0.0;
    float Asuf = 0.0, M1suf = 0.0, M2suf = 0.0, Lsuf = 0.0;

    for (int batch_end = end; batch_end > start; batch_end -= BACKWARD_BATCH) {
        int batch_start = max(start, batch_end - BACKWARD_BATCH);
        int count = batch_end - batch_start;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)local_idx < count) {
            int gid = sorted_ids[batch_start + int(local_idx)];
            sh_gid[local_idx] = gid;
            sh_row0[local_idx] = float3(transform[gid*9+0], transform[gid*9+1], transform[gid*9+2]);
            sh_row1[local_idx] = float3(transform[gid*9+3], transform[gid*9+4], transform[gid*9+5]);
            sh_row2[local_idx] = float3(transform[gid*9+6], transform[gid*9+7], transform[gid*9+8]);
            sh_mean2d[local_idx] = float2(means2d[gid*2+0], means2d[gid*2+1]);
            sh_opacity[local_idx] = opacities[gid];
            sh_color[local_idx] = float3(colors[gid*3+0], colors[gid*3+1], colors[gid*3+2]);
            sh_normal[local_idx] = float3(normal[gid*3+0], normal[gid*3+1], normal[gid*3+2]);
            sh_depth_fallback[local_idx] = depths[gid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j = count - 1; j >= 0; j--) {
            float3 g_row0 = float3(0.0), g_row1 = float3(0.0), g_row2 = float3(0.0);
            float3 g_normal = float3(0.0), g_color = float3(0.0);
            float2 g_mean2d = float2(0.0);
            float g_opacity = 0.0;
            float g_abs = 0.0;
            // The mean's own depth is a real input to the per-pixel math,
            // not just a sort key: `ray_splat_alpha` falls back to it for
            // z_hit wherever the ray-splat intersection is unusable (a
            // degenerate transform, or a pixel where the screen-space
            // gaussian wins). On those pixels -- and only those -- the
            // depth map's gradient flows here instead of into row2.
            float g_depth = 0.0;
            // 1.0 for each pixel this gaussian actually composited into, so
            // the host can turn the summed |screen gradient| into a *per
            // pixel* mean. Counted on exactly the same condition that adds
            // to g_abs, so the two are always consistent denominators.
            float g_pixels = 0.0;

            if (active && (batch_start + j) <= last) {
                float3 row0 = sh_row0[j], row1 = sh_row1[j], row2 = sh_row2[j];
                float2 mean2d = sh_mean2d[j];
                float opacity = sh_opacity[j];

                float3 hu, hv; float wloc, u, v, z_hit, raw, rho; bool degenerate, uv_active;
                float alpha = ray_splat_alpha(
                    row0, row1, row2, mean2d, pcx, pcy, opacity, near, filter_size, sh_depth_fallback[j],
                    hu, hv, wloc, u, v, degenerate, uv_active, z_hit, raw, rho);

                if (alpha >= (1.0 / 255.0)) {
                    T = T / (1.0 - alpha);
                    float weight = T * alpha;
                    float3 color = sh_color[j];
                    float3 normal_g = sh_normal[j];

                    float d_alpha_c = dot(T * (color - A_color), d_C);
                    float d_alpha_z = T * (z_hit - A_depth) * d_Depth;
                    float d_alpha_n = dot(T * (normal_g - A_normal), d_Normal);

                    // Distortion runs on normalized depth `m`, so every
                    // prefix/suffix moment here is over weight*m^p, and the
                    // resulting depth gradient is chained back onto metric
                    // z_hit via dm/dz. See the derivation note above.
                    float m_norm = normalized_depth(z_hit, near);
                    float m_sq = m_norm * m_norm;
                    float A_excl = A_run - weight;
                    float M1_excl = M1_run - weight * m_norm;
                    float M2_excl = M2_run - weight * m_sq;
                    float inv_1ma = 1.0 / (1.0 - alpha);  // alpha <= 0.99, never singular
                    float Fpre = m_sq * A_excl - 2.0 * m_norm * M1_excl + M2_excl;
                    float Gsuf = m_sq * Asuf - 2.0 * m_norm * M1suf + M2suf;
                    float cross = M2_excl * Asuf - 2.0 * M1_excl * M1suf + A_excl * M2suf;
                    float contrib1 = T * Fpre;
                    float contrib2 = T * Gsuf * (1.0 - 2.0 * alpha) * inv_1ma;
                    float contrib3 = -2.0 * Lsuf * inv_1ma;
                    float contrib4 = -cross * inv_1ma;
                    float d_alpha_dist = (contrib1 + contrib2 + contrib3 + contrib4) * d_Dist;
                    float g_zhit_dist = 2.0 * weight
                                        * (m_norm * (A_excl + Asuf) - (M1_excl + M1suf))
                                        * d_Dist * d_normalized_depth_dz(z_hit, near);

                    float d_alpha = d_alpha_c + d_alpha_z + d_alpha_n + d_alpha_dist;
                    if (raw >= 0.99) d_alpha = 0.0;

                    g_opacity = d_alpha * exp(-0.5 * rho);
                    float d_rho = d_alpha * raw * (-0.5);

                    g_color = weight * d_C;
                    float g_zhit_total = weight * d_Depth + g_zhit_dist;
                    g_normal = weight * d_Normal;

                    float d_rho_uv = uv_active ? d_rho : 0.0;
                    float d_rho_screen = uv_active ? 0.0 : d_rho;

                    float dx = pcx - mean2d.x, dy = pcy - mean2d.y;
                    float filter_var = filter_size * filter_size;
                    float d_dx = d_rho_screen * 2.0 * dx / filter_var;
                    float d_dy = d_rho_screen * 2.0 * dy / filter_var;
                    g_mean2d = float2(-d_dx, -d_dy);

                    // z_hit only depends on (u, v, row2) when the ray-splat
                    // branch actually won (see ray_splat_alpha) -- gate the
                    // whole implicit-adjoint block on that, not just on
                    // non-degeneracy, or a near-boundary z_hit gradient
                    // would leak into row0/row1/row2 via an unstable u/v
                    // that forward's output never actually depended on.
                    g_depth = (degenerate || !uv_active) ? g_zhit_total : 0.0;

                    if (!degenerate && uv_active) {
                        float d_u = d_rho_uv * 2.0 * u + g_zhit_total * row2.x;
                        float d_v = d_rho_uv * 2.0 * v + g_zhit_total * row2.y;
                        float3 d_row2_direct = g_zhit_total * float3(u, v, 1.0);

                        float lam1 = (d_u * hv.y - hv.x * d_v) / wloc;
                        float lam2 = (hu.x * d_v - hu.y * d_u) / wloc;
                        float3 uv1 = float3(u, v, 1.0);
                        g_row0 = -lam1 * uv1;
                        g_row1 = -lam2 * uv1;
                        g_row2 = (lam1 * pcx + lam2 * pcy) * uv1 + d_row2_direct;
                    }

                    // AbsGS-style densification signal, in the same
                    // screen-space (pixel) units as 3DGS's rasterize.metal
                    // uses. Two mutually-exclusive-per-pixel sources:
                    // the screen-space-fallback branch's direct d_mean2d
                    // (g_mean2d, already computed above), and -- the
                    // dominant one in practice, since most well-resolved
                    // pixels take the ray-splat branch, not the fallback
                    // -- the ray-splat branch's own sensitivity to the
                    // gaussian's *screen-projected* center, recovered from
                    // g_row0.z/g_row1.z (the mean-column entries of
                    // d_transform, i.e. d(row0[2])/d(row1[2])) via the
                    // exact identity row0[2] = means2d.x * z_mean, so
                    // d(means2d.x) = g_row0.z * z_mean at fixed depth.
                    //
                    // The scale factor is the *mean's* camera-space depth
                    // (sh_depth_fallback, the projection stage's `depths`
                    // output), not the per-pixel intersection depth
                    // z_hit. row0 is the pixel-x numerator evaluated at
                    // the mean column of M: row0[2] = fx*mean_cam.x +
                    // cx*mean_cam.z = means2d.x * mean_cam.z (see
                    // project_2dgs_ref), and mean_cam.z is the same for
                    // every pixel this gaussian covers. Using z_hit here
                    // scaled each pixel's contribution by z_hit/z_mean,
                    // which is 1 only for a splat parallel to the image
                    // plane and drifts with the intersection across a
                    // tilted one -- biasing the densification signal by
                    // splat orientation.
                    //
                    // This mirrors the official 2DGS CUDA rasterizer's
                    // dL_dmean2D, which is likewise derived from
                    // dL_dtransMat's mean-column entries scaled by depth
                    // (diff-surfel-rasterization's backward.cu), not from
                    // a screen-fallback-only term.
                    float z_mean = sh_depth_fallback[j];
                    float2 g_mean2d_equiv = g_mean2d + float2(g_row0.z, g_row1.z) * z_mean;
                    g_abs = length(g_mean2d_equiv);
                    g_pixels = 1.0;

                    A_color = alpha * color + (1.0 - alpha) * A_color;
                    A_depth = alpha * z_hit + (1.0 - alpha) * A_depth;
                    A_normal = alpha * normal_g + (1.0 - alpha) * A_normal;
                    Lsuf = Lsuf + weight * Gsuf;
                    Asuf = Asuf + weight;
                    M1suf = M1suf + weight * m_norm;
                    M2suf = M2suf + weight * m_sq;
                    A_run = A_excl;
                    M1_run = M1_excl;
                    M2_run = M2_excl;
                }
            }

            float r_row0x = simd_sum(g_row0.x), r_row0y = simd_sum(g_row0.y), r_row0z = simd_sum(g_row0.z);
            float r_row1x = simd_sum(g_row1.x), r_row1y = simd_sum(g_row1.y), r_row1z = simd_sum(g_row1.z);
            float r_row2x = simd_sum(g_row2.x), r_row2y = simd_sum(g_row2.y), r_row2z = simd_sum(g_row2.z);
            float r_nx = simd_sum(g_normal.x), r_ny = simd_sum(g_normal.y), r_nz = simd_sum(g_normal.z);
            float r_colx = simd_sum(g_color.x), r_coly = simd_sum(g_color.y), r_colz = simd_sum(g_color.z);
            float r_mx = simd_sum(g_mean2d.x), r_my = simd_sum(g_mean2d.y);
            float r_op = simd_sum(g_opacity);
            float r_abs = simd_sum(g_abs);
            float r_pixels = simd_sum(g_pixels);
            float r_depth = simd_sum(g_depth);

            if (lane == 0) {
                bool any = (r_row0x!=0.0)||(r_row0y!=0.0)||(r_row0z!=0.0)
                        || (r_row1x!=0.0)||(r_row1y!=0.0)||(r_row1z!=0.0)
                        || (r_row2x!=0.0)||(r_row2y!=0.0)||(r_row2z!=0.0)
                        || (r_abs!=0.0) || (r_pixels!=0.0)
                        || (r_nx!=0.0)||(r_ny!=0.0)||(r_nz!=0.0)
                        || (r_colx!=0.0)||(r_coly!=0.0)||(r_colz!=0.0)
                        || (r_mx!=0.0)||(r_my!=0.0)||(r_op!=0.0)||(r_depth!=0.0);
                if (any) {
                    int gid = sh_gid[j];
                    atomic_fetch_add_explicit(&d_transform[gid*9+0], r_row0x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+1], r_row0y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+2], r_row0z, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+3], r_row1x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+4], r_row1y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+5], r_row1z, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+6], r_row2x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+7], r_row2y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_transform[gid*9+8], r_row2z, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_normal[gid*3+0], r_nx, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_normal[gid*3+1], r_ny, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_normal[gid*3+2], r_nz, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_colors[gid*3+0], r_colx, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_colors[gid*3+1], r_coly, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_colors[gid*3+2], r_colz, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d[gid*2+0], r_mx, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d[gid*2+1], r_my, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_opacities[gid], r_op, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d_abs[gid], r_abs, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_pixel_count[gid], r_pixels, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_depths[gid], r_depth, memory_order_relaxed);
                }
            }
        }
    }
}
