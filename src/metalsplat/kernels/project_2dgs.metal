// Forward + backward projection of 2D gaussian splats ("surfels") to 2D
// screen space. Mirrors metalsplat.reference.project_2dgs_ref.
// project_gaussians_2dgs exactly; that module is the spec and the
// numerical oracle these kernels are tested against. See its docstring
// for the M/H/W ray-splat-intersection math and conventions.
//
// Produces `transform` (M's 9 independent entries) and `normal` for the
// rasterizer's exact per-pixel ray-splat intersection, plus each splat's
// exact screen-space bounding rectangle for tile binning and culling.
#include <metal_stdlib>
using namespace metal;

// sqrt(2 ln 255): the farthest (in splat sigmas) any splat composites
// before its alpha falls below the rasterizer's 1/255 cutoff. Matches
// reference/tiling_ref.py MAX_SIGMA_EXTENT.
constant float MAX_SIGMA_EXTENT = 3.3290429;

inline float3x3 mat3_from_rowmajor(constant float* flat) {
    return float3x3(float3(flat[0], flat[3], flat[6]),
                     float3(flat[1], flat[4], flat[7]),
                     float3(flat[2], flat[5], flat[8]));
}

inline float3x3 quat_to_rotmat(float w, float x, float y, float z) {
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    float3 col0 = float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
    float3 col1 = float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
    float3 col2 = float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
    return float3x3(col0, col1, col2);
}

// Exact screen rectangle (xmin, ymin, xmax, ymax) of everywhere the splat
// can composite at alpha cutoff radius `c` (in splat sigmas). See
// project_2dgs_ref.surfel_rects for the derivation; this mirrors it.
//
// The rasterizer's rho is min(rho_uv, rho_screen), so the footprint is the
// union of (a) the perspective image of the local ellipse u^2 + v^2 <= c^2
// and (b) the screen-space filter's disk of radius c * filter_size around
// the projected center. (a) is found from the dual conic: in screen
// coordinates relative to the projected center (rows r0' = r0 - mx*r2,
// r1' = r1 - my*r2, which keeps the numbers small), C*_ij =
// dot((c^2, c^2, -1), ri' * rj'); the ellipse is bounded iff C*_22 < 0,
// with center C*_02/C*_22 and half-extent sqrt(center^2 - C*_00/C*_22)
// per axis. C*_22 >= 0 means the local ellipse reaches the camera plane,
// so its image is unbounded: the whole frame is the only safe bound.
inline float4 surfel_rect(float3 row0, float3 row1, float3 row2, float2 m2d,
                          float c, float filter_size, float img_width, float img_height)
{
    float3 r0 = row0 - m2d.x * row2;
    float3 r1 = row1 - m2d.y * row2;
    float3 d = float3(c * c, c * c, -1.0);
    float c22 = dot(d, row2 * row2);
    float4 rect;
    if (c22 < 0.0) {
        float ox = dot(d, r0 * row2) / c22;
        float oy = dot(d, r1 * row2) / c22;
        float hx = sqrt(max(ox * ox - dot(d, r0 * r0) / c22, 0.0));
        float hy = sqrt(max(oy * oy - dot(d, r1 * r1) / c22, 0.0));
        rect = float4(m2d.x + ox - hx, m2d.y + oy - hy, m2d.x + ox + hx, m2d.y + oy + hy);
    } else {
        rect = float4(0.0, 0.0, img_width, img_height);
    }
    float s = c * filter_size;
    return float4(min(rect.x, m2d.x - s), min(rect.y, m2d.y - s),
                  max(rect.z, m2d.x + s), max(rect.w, m2d.y + s));
}

kernel void project_2dgs_forward(
    device const float* means,        // (N,3)
    device const float* scales,       // (N,2) s_u, s_v
    device const float* quats,        // (N,4) w,x,y,z
    device const float* cutoffs,      // (N,) binning alpha cutoff in sigmas, 0 = cull
    constant float* rwc,               // (9,) row-major world-to-camera rotation
    constant float* twc,                // (3,) world-to-camera translation
    constant float& fx,
    constant float& fy,
    constant float& cx,
    constant float& cy,
    constant float& img_width,
    constant float& img_height,
    constant float& near,
    constant float& filter_size,
    device float* out_means2d,          // (N,2)
    device float* out_depths,            // (N,)
    device float* out_rects,              // (N,4) xmin, ymin, xmax, ymax
    device float* out_valid,               // (N,) 1.0 / 0.0
    device float* out_transform,            // (N,9) row-major 3x3
    device float* out_normal,                // (N,3)
    uint gid [[thread_position_in_grid]])
{
    float3 mean = float3(means[gid * 3 + 0], means[gid * 3 + 1], means[gid * 3 + 2]);
    float su = scales[gid * 2 + 0], sv = scales[gid * 2 + 1];
    float4 q = float4(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]);

    float3x3 Rwc = mat3_from_rowmajor(rwc);
    float3x3 RwcT = transpose(Rwc);
    float3 twc_v = float3(twc[0], twc[1], twc[2]);

    float3 mean_cam = Rwc * mean + twc_v;
    float x = mean_cam.x, y = mean_cam.y, z = mean_cam.z;
    out_depths[gid] = z;
    float z_safe = max(z, near);

    float u = fx * x / z_safe + cx;
    float v = fy * y / z_safe + cy;
    out_means2d[gid * 2 + 0] = u;
    out_means2d[gid * 2 + 1] = v;

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3 t_u = Rq[0] * su;
    float3 t_v = Rq[1] * sv;
    float3 normal_raw = Rq[2];

    float3 cam_pos = -(RwcT * twc_v);
    float3 view_dir = cam_pos - mean;
    float flip_sign = (dot(normal_raw, view_dir) < 0.0) ? -1.0 : 1.0;
    float3 normal = flip_sign * normal_raw;
    out_normal[gid * 3 + 0] = normal.x;
    out_normal[gid * 3 + 1] = normal.y;
    out_normal[gid * 3 + 2] = normal.z;

    float3 tu_cam = Rwc * t_u;   // direction: no translation
    float3 tv_cam = Rwc * t_v;

    float3 row0 = float3(fx * tu_cam.x + cx * tu_cam.z, fx * tv_cam.x + cx * tv_cam.z, fx * mean_cam.x + cx * mean_cam.z);
    float3 row1 = float3(fy * tu_cam.y + cy * tu_cam.z, fy * tv_cam.y + cy * tv_cam.z, fy * mean_cam.y + cy * mean_cam.z);
    float3 row2 = float3(tu_cam.z, tv_cam.z, mean_cam.z);
    for (int k = 0; k < 3; ++k) {
        out_transform[gid * 9 + 0 * 3 + k] = row0[k];
        out_transform[gid * 9 + 1 * 3 + k] = row1[k];
        out_transform[gid * 9 + 2 * 3 + k] = row2[k];
    }

    // Visibility is judged at the largest cutoff any opacity can reach, so
    // it does not depend on opacity (as in 3DGS's projection); the binning
    // rectangle uses the gaussian's own, opacity-aware cutoff.
    float2 m2d = float2(u, v);
    float4 vis = surfel_rect(row0, row1, row2, m2d, MAX_SIGMA_EXTENT, filter_size, img_width, img_height);
    bool in_front = z > near;
    bool in_bounds = (vis.z >= 0.0) && (vis.x < img_width) &&
                     (vis.w >= 0.0) && (vis.y < img_height);
    bool valid = in_front && in_bounds;
    out_valid[gid] = valid ? 1.0 : 0.0;

    float c = cutoffs[gid];
    float4 rect = (valid && c > 0.0)
        ? surfel_rect(row0, row1, row2, m2d, c, filter_size, img_width, img_height)
        : float4(0.0, 0.0, -1.0, -1.0);
    out_rects[gid * 4 + 0] = rect.x;
    out_rects[gid * 4 + 1] = rect.y;
    out_rects[gid * 4 + 2] = rect.z;
    out_rects[gid * 4 + 3] = rect.w;
}

kernel void project_2dgs_backward(
    device const float* means,        // (N,3)
    device const float* scales,       // (N,2)
    device const float* quats,        // (N,4) w,x,y,z
    constant float* rwc,               // (9,) row-major
    constant float* twc,                // (3,)
    constant float& fx,
    constant float& fy,
    constant float& cx,
    constant float& cy,
    constant float& near,
    device const float* valid_in,       // (N,) from forward
    device const float* d_means2d,       // (N,2)
    device const float* d_depths,         // (N,) camera-space z of the mean
    device const float* d_transform,        // (N,9) row-major 3x3
    device const float* d_normal,            // (N,3)
    device float* d_means,                    // (N,3)
    device float* d_scales,                    // (N,2)
    device float* d_quats,                      // (N,4)
    uint gid [[thread_position_in_grid]])
{
    if (valid_in[gid] < 0.5) {
        d_means[gid * 3 + 0] = 0.0; d_means[gid * 3 + 1] = 0.0; d_means[gid * 3 + 2] = 0.0;
        d_scales[gid * 2 + 0] = 0.0; d_scales[gid * 2 + 1] = 0.0;
        d_quats[gid * 4 + 0] = 0.0; d_quats[gid * 4 + 1] = 0.0;
        d_quats[gid * 4 + 2] = 0.0; d_quats[gid * 4 + 3] = 0.0;
        return;
    }

    // ---- Recompute the forward pass (see project_2dgs_forward) ----
    float3 mean = float3(means[gid * 3 + 0], means[gid * 3 + 1], means[gid * 3 + 2]);
    float su = scales[gid * 2 + 0], sv = scales[gid * 2 + 1];
    float4 q = float4(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]);

    float3x3 Rwc = mat3_from_rowmajor(rwc);
    float3 twc_v = float3(twc[0], twc[1], twc[2]);
    float3x3 RwcT = transpose(Rwc);

    float3 mean_cam = Rwc * mean + twc_v;
    float x = mean_cam.x, y = mean_cam.y;
    // A valid gaussian has z > near, so z_safe == z here.
    float z_safe = max(mean_cam.z, near);

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3 normal_raw = Rq[2];
    float3 cam_pos = -(RwcT * twc_v);
    float3 view_dir = cam_pos - mean;
    float flip_sign = (dot(normal_raw, view_dir) < 0.0) ? -1.0 : 1.0;

    // ---- means2d = (fx*x/z + cx, fy*y/z + cy) backward ----
    float d_u = d_means2d[gid * 2 + 0];
    float d_v = d_means2d[gid * 2 + 1];
    float3 d_mean_cam = float3(
        (fx / z_safe) * d_u,
        (fy / z_safe) * d_v,
        (-fx * x / (z_safe * z_safe)) * d_u + (-fy * y / (z_safe * z_safe)) * d_v);

    // ---- transform (M) backward ----
    float g00 = d_transform[gid * 9 + 0], g01 = d_transform[gid * 9 + 1], g02 = d_transform[gid * 9 + 2];
    float g10 = d_transform[gid * 9 + 3], g11 = d_transform[gid * 9 + 4], g12 = d_transform[gid * 9 + 5];
    float g20 = d_transform[gid * 9 + 6], g21 = d_transform[gid * 9 + 7], g22 = d_transform[gid * 9 + 8];

    float3 d_tu_cam = float3(fx * g00, fy * g10, cx * g00 + cy * g10 + g20);
    float3 d_tv_cam = float3(fx * g01, fy * g11, cx * g01 + cy * g11 + g21);
    d_mean_cam += float3(fx * g02, fy * g12, cx * g02 + cy * g12 + g22);

    // out_depths = mean_cam.z straight out of the forward (the *raw* z, not
    // z_safe -- a culled-by-`near` gaussian returns early above), so its
    // upstream gradient lands on that component and nowhere else.
    d_mean_cam.z += d_depths[gid];

    float3 d_mean = RwcT * d_mean_cam;
    d_means[gid * 3 + 0] = d_mean.x;
    d_means[gid * 3 + 1] = d_mean.y;
    d_means[gid * 3 + 2] = d_mean.z;

    float3 d_t_u = RwcT * d_tu_cam;   // direction, no translation
    float3 d_t_v = RwcT * d_tv_cam;

    // ---- t_u = Rq[0] * su, t_v = Rq[1] * sv, normal = flip * Rq[2] ----
    d_scales[gid * 2 + 0] = dot(d_t_u, Rq[0]);
    d_scales[gid * 2 + 1] = dot(d_t_v, Rq[1]);

    float3 d_normal_up = float3(d_normal[gid * 3 + 0], d_normal[gid * 3 + 1], d_normal[gid * 3 + 2]);
    float3 D_Rq0 = d_t_u * su;
    float3 D_Rq1 = d_t_v * sv;
    float3 D_Rq2 = flip_sign * d_normal_up;

    float d_R00 = D_Rq0[0], d_R10 = D_Rq0[1], d_R20 = D_Rq0[2];
    float d_R01 = D_Rq1[0], d_R11 = D_Rq1[1], d_R21 = D_Rq1[2];
    float d_R02 = D_Rq2[0], d_R12 = D_Rq2[1], d_R22 = D_Rq2[2];

    float w = q.x, qx = q.y, qy = q.z, qz = q.w;

    float d_w = 2.0 * qz * (d_R10 - d_R01) + 2.0 * qy * (d_R02 - d_R20) + 2.0 * qx * (d_R21 - d_R12);
    float d_qx = 2.0 * qy * (d_R10 + d_R01) + 2.0 * qz * (d_R20 + d_R02) + 2.0 * w * (d_R21 - d_R12) - 4.0 * qx * (d_R11 + d_R22);
    float d_qy = 2.0 * qx * (d_R10 + d_R01) + 2.0 * qz * (d_R21 + d_R12) + 2.0 * w * (d_R02 - d_R20) - 4.0 * qy * (d_R00 + d_R22);
    float d_qz = 2.0 * w * (d_R10 - d_R01) + 2.0 * qx * (d_R20 + d_R02) + 2.0 * qy * (d_R21 + d_R12) - 4.0 * qz * (d_R00 + d_R11);

    d_quats[gid * 4 + 0] = d_w;
    d_quats[gid * 4 + 1] = d_qx;
    d_quats[gid * 4 + 2] = d_qy;
    d_quats[gid * 4 + 3] = d_qz;
}
