// Tile-based front-to-back alpha-compositing rasterizer, forward + backward.
// Mirrors metalsplat.reference.rasterize_ref.rasterize_gaussians (a brute
// force per-pixel-over-all-gaussians implementation) exactly in the math;
// that module is the oracle these kernels are tested against. Dispatch is
// one thread per pixel, one threadgroup per tile (threads=(tiles_x*T,
// tiles_y*T), group_size=(T,T)), matching gsplat's tiled layout.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

inline float gaussian_alpha(float2 d, float a, float b, float c, float opacity, thread float& power_out) {
    float power = -0.5 * (a * d.x * d.x + 2.0 * b * d.x * d.y + c * d.y * d.y);
    power_out = power;
    if (power > 0.0) return 0.0;
    return min(0.99, opacity * exp(power));
}

kernel void rasterize_forward(
    device const float* means2d,             // (N,2)
    device const float* conics,               // (N,3) a,b,c
    device const float* opacities,             // (N,)
    device const float* colors,                 // (N,3)
    device const float* depths,                  // (N,) camera-space z
    device const int* sorted_ids,                 // (M,)
    device const int* tile_bins,                   // (num_tiles,2)
    constant int& tiles_x,
    constant int& img_width,
    constant int& img_height,
    constant int& tile_size,
    constant float* background,
    device float* out_image,                         // (H,W,3)
    device float* out_depth,                           // (H,W) alpha-weighted expected depth
    device float* out_final_T,                          // (H,W)
    device int* out_last_contributor,                     // (H,W)
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]])
{
    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    if (px >= uint(img_width) || py >= uint(img_height)) return;

    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    int end = tile_bins[tile_id * 2 + 1];

    float2 pixel_center = float2(float(px) + 0.5, float(py) + 0.5);
    float3 accum = float3(0.0, 0.0, 0.0);
    float accum_depth = 0.0;
    float T = 1.0;
    int last_contributor = -1;

    for (int idx = start; idx < end; idx++) {
        int gid = sorted_ids[idx];
        float2 mean = float2(means2d[gid * 2 + 0], means2d[gid * 2 + 1]);
        float2 d = pixel_center - mean;
        float a = conics[gid * 3 + 0], b = conics[gid * 3 + 1], c = conics[gid * 3 + 2];
        float power;
        float alpha = gaussian_alpha(d, a, b, c, opacities[gid], power);
        if (alpha < (1.0 / 255.0)) continue;

        float test_T = T * (1.0 - alpha);
        if (test_T < 1e-4) break;

        float3 color = float3(colors[gid * 3 + 0], colors[gid * 3 + 1], colors[gid * 3 + 2]);
        float weight = T * alpha;
        accum += weight * color;
        accum_depth += weight * depths[gid];
        T = test_T;
        last_contributor = idx;
    }

    accum += T * float3(background[0], background[1], background[2]);

    uint pixel_idx = py * uint(img_width) + px;
    out_image[pixel_idx * 3 + 0] = accum.x;
    out_image[pixel_idx * 3 + 1] = accum.y;
    out_image[pixel_idx * 3 + 2] = accum.z;
    out_depth[pixel_idx] = accum_depth;
    out_final_T[pixel_idx] = T;
    out_last_contributor[pixel_idx] = last_contributor;
}

// Backward. Two things dominate cost here, and both are addressed by having
// the whole threadgroup walk the tile's gaussian list in lockstep rather
// than each pixel walking it alone:
//
//  1. Atomics. Every contributing (pixel, gaussian) pair used to issue 11
//     global atomic adds, and a 16x16 tile means 256 threads hammering the
//     same gaussian's gradient slots. Now each SIMD group reduces its 32
//     lanes with simd_sum and one lane issues the atomic, cutting atomic
//     traffic by up to 32x.
//  2. Loads. All 256 threads used to read the same gaussian's mean, conic,
//     opacity and colour from device memory. They are now staged once into
//     threadgroup memory per batch.
//
// The cost of lockstep is that a thread whose pixel saturated early still
// loops over the rest of the tile doing nothing. That is the same trade the
// reference CUDA implementation makes.
#define BACKWARD_BATCH 256

kernel void rasterize_backward(
    device const float* means2d,             // (N,2)
    device const float* conics,               // (N,3)
    device const float* opacities,             // (N,)
    device const float* colors,                 // (N,3)
    device const int* sorted_ids,                // (M,)
    device const int* tile_bins,                  // (num_tiles,2)
    constant int& tiles_x,
    constant int& img_width,
    constant int& img_height,
    constant int& tile_size,
    constant float* background,
    device const float* final_T,                     // (H,W), from forward
    device const int* last_contributor,                // (H,W), from forward
    device const float* d_out_image,                     // (H,W,3)
    device atomic_float* d_means2d,                        // (N,2)
    device atomic_float* d_conics,                          // (N,3)
    device atomic_float* d_opacities,                        // (N,)
    device atomic_float* d_colors,                            // (N,3)
    device atomic_float* d_means2d_abs,                        // (N,) sum of |per-pixel contribution|
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint local_idx [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float2 sh_mean[BACKWARD_BATCH];
    threadgroup float3 sh_conic[BACKWARD_BATCH];
    threadgroup float sh_opacity[BACKWARD_BATCH];
    threadgroup float3 sh_color[BACKWARD_BATCH];
    threadgroup int sh_gid[BACKWARD_BATCH];

    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    // No early return: simd_sum below requires every lane of the SIMD group
    // to reach it, so out-of-bounds threads stay alive and contribute zero.
    bool inside = (px < uint(img_width)) && (py < uint(img_height));

    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    int end = tile_bins[tile_id * 2 + 1];

    uint pixel_idx = inside ? (py * uint(img_width) + px) : 0;
    int last = inside ? last_contributor[pixel_idx] : -1;
    bool active = inside && (last >= 0);

    float2 pixel_center = float2(float(px) + 0.5, float(py) + 0.5);
    float3 d_C = active
        ? float3(d_out_image[pixel_idx * 3 + 0], d_out_image[pixel_idx * 3 + 1], d_out_image[pixel_idx * 3 + 2])
        : float3(0.0);
    float T = active ? final_T[pixel_idx] : 0.0;
    float3 A = float3(background[0], background[1], background[2]);  // suffix colour, starts at A_k = background

    for (int batch_end = end; batch_end > start; batch_end -= BACKWARD_BATCH) {
        int batch_start = max(start, batch_end - BACKWARD_BATCH);
        int count = batch_end - batch_start;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)local_idx < count) {
            int gid = sorted_ids[batch_start + int(local_idx)];
            sh_gid[local_idx] = gid;
            sh_mean[local_idx] = float2(means2d[gid * 2 + 0], means2d[gid * 2 + 1]);
            sh_conic[local_idx] = float3(conics[gid * 3 + 0], conics[gid * 3 + 1], conics[gid * 3 + 2]);
            sh_opacity[local_idx] = opacities[gid];
            sh_color[local_idx] = float3(colors[gid * 3 + 0], colors[gid * 3 + 1], colors[gid * 3 + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j = count - 1; j >= 0; j--) {
            float3 g_color = float3(0.0);
            float3 g_conic = float3(0.0);
            float2 g_mean = float2(0.0);
            float g_opacity = 0.0;
            float g_abs = 0.0;

            // Gaussians past this pixel's last contributor never affected it.
            if (active && (batch_start + j) <= last) {
                float2 d = pixel_center - sh_mean[j];
                float3 cn = sh_conic[j];
                float opacity = sh_opacity[j];
                float power;
                float alpha = gaussian_alpha(d, cn.x, cn.y, cn.z, opacity, power);
                if (alpha >= (1.0 / 255.0)) {
                    T = T / (1.0 - alpha);  // recover T_i from T_{i+1}
                    float3 color = sh_color[j];

                    float3 dC_dalpha = T * (color - A);
                    float d_alpha_raw = dot(dC_dalpha, d_C);
                    float raw = opacity * exp(power);
                    // if alpha was clamped at 0.99, gradient doesn't flow through the clamp
                    if (raw >= 0.99) d_alpha_raw = 0.0;

                    g_color = (T * alpha) * d_C;
                    g_opacity = d_alpha_raw * exp(power);

                    float d_power = d_alpha_raw * raw;
                    g_conic = float3(d_power * (-0.5 * d.x * d.x),
                                     d_power * (-1.0 * d.x * d.y),
                                     d_power * (-0.5 * d.y * d.y));

                    float d_dx = d_power * (-(cn.x * d.x + cn.y * d.y));
                    float d_dy = d_power * (-(cn.y * d.x + cn.z * d.y));
                    g_mean = float2(-d_dx, -d_dy);

                    // AbsGS-style densification signal: sum of |per-pixel
                    // contribution| magnitudes rather than the (signed)
                    // gradient of the summed loss. Contributions from
                    // different pixels can have opposite signs and cancel out
                    // in d_means2d, hiding gaussians that are being pulled in
                    // conflicting directions by different parts of the image
                    // -- exactly the over-reconstructed/blurry case
                    // densification is supposed to catch.
                    g_abs = sqrt(d_dx * d_dx + d_dy * d_dy);

                    A = alpha * color + (1.0 - alpha) * A;
                }
            }

            // One atomic per SIMD group instead of one per contributing pixel.
            float r_col_x = simd_sum(g_color.x);
            float r_col_y = simd_sum(g_color.y);
            float r_col_z = simd_sum(g_color.z);
            float r_op = simd_sum(g_opacity);
            float r_con_x = simd_sum(g_conic.x);
            float r_con_y = simd_sum(g_conic.y);
            float r_con_z = simd_sum(g_conic.z);
            float r_mean_x = simd_sum(g_mean.x);
            float r_mean_y = simd_sum(g_mean.y);
            float r_abs = simd_sum(g_abs);

            if (lane == 0) {
                bool any = (r_col_x != 0.0) || (r_col_y != 0.0) || (r_col_z != 0.0)
                        || (r_op != 0.0) || (r_con_x != 0.0) || (r_con_y != 0.0)
                        || (r_con_z != 0.0) || (r_mean_x != 0.0) || (r_mean_y != 0.0)
                        || (r_abs != 0.0);
                if (any) {
                    int gid = sh_gid[j];
                    atomic_fetch_add_explicit(&d_colors[gid * 3 + 0], r_col_x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_colors[gid * 3 + 1], r_col_y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_colors[gid * 3 + 2], r_col_z, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_opacities[gid], r_op, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_conics[gid * 3 + 0], r_con_x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_conics[gid * 3 + 1], r_con_y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_conics[gid * 3 + 2], r_con_z, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d[gid * 2 + 0], r_mean_x, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d[gid * 2 + 1], r_mean_y, memory_order_relaxed);
                    atomic_fetch_add_explicit(&d_means2d_abs[gid], r_abs, memory_order_relaxed);
                }
            }
        }
    }
}
