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
    device const int* sorted_ids,                // (M,)
    device const int* tile_bins,                  // (num_tiles,2)
    constant int& tiles_x,
    constant int& img_width,
    constant int& img_height,
    constant int& tile_size,
    constant float* background,
    device float* out_image,                        // (H,W,3)
    device float* out_final_T,                        // (H,W)
    device int* out_last_contributor,                   // (H,W)
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
        accum += T * alpha * color;
        T = test_T;
        last_contributor = idx;
    }

    accum += T * float3(background[0], background[1], background[2]);

    uint pixel_idx = py * uint(img_width) + px;
    out_image[pixel_idx * 3 + 0] = accum.x;
    out_image[pixel_idx * 3 + 1] = accum.y;
    out_image[pixel_idx * 3 + 2] = accum.z;
    out_final_T[pixel_idx] = T;
    out_last_contributor[pixel_idx] = last_contributor;
}

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
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]])
{
    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    if (px >= uint(img_width) || py >= uint(img_height)) return;

    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];

    uint pixel_idx = py * uint(img_width) + px;
    int last = last_contributor[pixel_idx];
    if (last < 0) return;  // no gaussian contributed to this pixel

    float2 pixel_center = float2(float(px) + 0.5, float(py) + 0.5);
    float3 d_C = float3(d_out_image[pixel_idx * 3 + 0], d_out_image[pixel_idx * 3 + 1], d_out_image[pixel_idx * 3 + 2]);

    float T = final_T[pixel_idx];
    float3 A = float3(background[0], background[1], background[2]);  // suffix color accumulator, starts as A_k = background

    for (int idx = last; idx >= start; idx--) {
        int gid = sorted_ids[idx];
        float2 mean = float2(means2d[gid * 2 + 0], means2d[gid * 2 + 1]);
        float2 d = pixel_center - mean;
        float a = conics[gid * 3 + 0], b = conics[gid * 3 + 1], c = conics[gid * 3 + 2];
        float power;
        float alpha = gaussian_alpha(d, a, b, c, opacities[gid], power);
        if (alpha < (1.0 / 255.0)) continue;

        T = T / (1.0 - alpha);  // recover T_i from T_{i+1}
        float3 color = float3(colors[gid * 3 + 0], colors[gid * 3 + 1], colors[gid * 3 + 2]);

        float3 dC_dalpha = T * (color - A);
        float d_alpha_raw = dot(dC_dalpha, d_C);  // gradient w.r.t. raw = opacity*exp(power)
        // if alpha was clamped at 0.99, gradient doesn't flow through the clamp
        float raw = opacities[gid] * exp(power);
        if (raw >= 0.99) d_alpha_raw = 0.0;

        float3 d_color = (T * alpha) * d_C;
        atomic_fetch_add_explicit(&d_colors[gid * 3 + 0], d_color.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_colors[gid * 3 + 1], d_color.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_colors[gid * 3 + 2], d_color.z, memory_order_relaxed);

        float d_opacity = d_alpha_raw * exp(power);
        atomic_fetch_add_explicit(&d_opacities[gid], d_opacity, memory_order_relaxed);

        float d_power = d_alpha_raw * raw;
        float d_a = d_power * (-0.5 * d.x * d.x);
        float d_b = d_power * (-1.0 * d.x * d.y);
        float d_cc = d_power * (-0.5 * d.y * d.y);
        atomic_fetch_add_explicit(&d_conics[gid * 3 + 0], d_a, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_conics[gid * 3 + 1], d_b, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_conics[gid * 3 + 2], d_cc, memory_order_relaxed);

        float d_dx = d_power * (-(a * d.x + b * d.y));
        float d_dy = d_power * (-(b * d.x + c * d.y));
        atomic_fetch_add_explicit(&d_means2d[gid * 2 + 0], -d_dx, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_means2d[gid * 2 + 1], -d_dy, memory_order_relaxed);

        A = alpha * color + (1.0 - alpha) * A;
    }
}
