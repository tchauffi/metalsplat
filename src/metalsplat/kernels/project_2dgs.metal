// Forward + backward projection of 2D gaussian splats ("surfels") to 2D
// screen space. Mirrors metalsplat.reference.project_2dgs_ref.
// project_gaussians_2dgs exactly; that module is the spec and the
// numerical oracle these kernels are tested against. See its docstring
// for the M/H/W ray-splat-intersection math and conventions.
//
// Reuses project.metal's exact EWA/conic math (treating the missing 3rd
// scale axis as EPS_3RD_AXIS) for the tile-culling outputs
// (means2d/depths/conics/radii/valid/compensation), and additionally
// produces `transform` (M's 9 independent entries) and `normal` for the
// rasterizer's exact per-pixel ray-splat intersection.
#include <metal_stdlib>
using namespace metal;

constant float EPS_3RD_AXIS = 1e-6;       // stand-in for the missing depth-axis scale
constant float RADIUS_SAFETY_MARGIN = 2.5; // tile-culling radius inflation, see project_2dgs_ref

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

kernel void project_2dgs_forward(
    device const float* means,        // (N,3)
    device const float* scales,       // (N,2) s_u, s_v
    device const float* quats,        // (N,4) w,x,y,z
    constant float* rwc,               // (9,) row-major world-to-camera rotation
    constant float* twc,                // (3,) world-to-camera translation
    constant float& fx,
    constant float& fy,
    constant float& cx,
    constant float& cy,
    constant float& img_width,
    constant float& img_height,
    constant float& near,
    constant float& eps2d,
    device float* out_means2d,          // (N,2)
    device float* out_depths,            // (N,)
    device float* out_conics,             // (N,3) a,b,c -- tile-culling only
    device float* out_radii,               // (N,)
    device float* out_valid,                // (N,) 1.0 / 0.0
    device float* out_compensations,         // (N,)
    device float* out_transform,              // (N,9) row-major 3x3
    device float* out_normal,                  // (N,3)
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

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3x3 M = float3x3(Rq[0] * su, Rq[1] * sv, Rq[2] * EPS_3RD_AXIS);
    float3x3 SigmaWorld = M * transpose(M);
    float3x3 SigmaCam = Rwc * SigmaWorld * RwcT;

    bool in_front = z > near;
    float z_safe = max(z, near);

    float S00 = SigmaCam[0][0], S11 = SigmaCam[1][1], S22 = SigmaCam[2][2];
    float S01 = SigmaCam[1][0], S02 = SigmaCam[2][0], S12 = SigmaCam[2][1];

    float lim_x = 1.3 * (0.5 * img_width) / fx;
    float lim_y = 1.3 * (0.5 * img_height) / fy;
    float tx = clamp(x / z_safe, -lim_x, lim_x) * z_safe;
    float ty = clamp(y / z_safe, -lim_y, lim_y) * z_safe;

    float j0 = fx / z_safe;
    float j1 = fy / z_safe;
    float j2 = -fx * tx / (z_safe * z_safe);
    float j3 = -fy * ty / (z_safe * z_safe);

    float a_raw = j0 * j0 * S00 + 2.0 * j0 * j2 * S02 + j2 * j2 * S22;
    float c_raw = j1 * j1 * S11 + 2.0 * j1 * j3 * S12 + j3 * j3 * S22;
    float b_raw = j0 * j1 * S01 + j0 * j3 * S02 + j1 * j2 * S12 + j2 * j3 * S22;

    float a = a_raw + eps2d;
    float c = c_raw + eps2d;
    float b = b_raw;

    float det = a * c - b * b;
    float det_safe = max(det, 1e-12);
    // ops.tiling/kernels/tiling.metal derive the per-tile bounding box from
    // this conic (not from out_radii below), so the tile-culling safety
    // margin has to inflate the covariance it represents: scaling Sigma2d
    // by margin^2 is scaling its inverse (the conic) by 1/margin^2. See
    // project_2dgs_ref's matching comment.
    float margin_sq = RADIUS_SAFETY_MARGIN * RADIUS_SAFETY_MARGIN;
    out_conics[gid * 3 + 0] = (c / det_safe) / margin_sq;
    out_conics[gid * 3 + 1] = (-b / det_safe) / margin_sq;
    out_conics[gid * 3 + 2] = (a / det_safe) / margin_sq;

    float det_orig = max(a_raw * c_raw - b_raw * b_raw, 0.0);
    out_compensations[gid] = sqrt(clamp(det_orig / det_safe, 0.0, 1.0));

    float mid = 0.5 * (a + c);
    float disc = max(mid * mid - det, 0.0);
    float lambda_max = mid + sqrt(disc);
    // Matches project_2dgs_ref exactly: ceil the raw 3DGS-style radius
    // first, then apply the tile-culling safety margin and ceil again
    // (rather than folding the margin in before the first ceil).
    float radius0 = ceil(3.0 * sqrt(max(lambda_max, 0.0)));
    float radius = ceil(radius0 * RADIUS_SAFETY_MARGIN);

    float u = fx * x / z_safe + cx;
    float v = fy * y / z_safe + cy;
    out_means2d[gid * 2 + 0] = u;
    out_means2d[gid * 2 + 1] = v;

    bool positive_det = det > 0.0;
    bool nonzero_radius = radius > 0.0;
    bool in_bounds = (u + radius >= 0.0) && (u - radius < img_width) &&
                      (v + radius >= 0.0) && (v - radius < img_height);
    bool valid = in_front && positive_det && nonzero_radius && in_bounds;

    out_radii[gid] = valid ? radius : 0.0;
    out_valid[gid] = valid ? 1.0 : 0.0;

    // ---- Exact ray-splat data: transform (M) and camera-facing normal ----
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

    float row0[3] = {fx * tu_cam.x + cx * tu_cam.z, fx * tv_cam.x + cx * tv_cam.z, fx * mean_cam.x + cx * mean_cam.z};
    float row1[3] = {fy * tu_cam.y + cy * tu_cam.z, fy * tv_cam.y + cy * tv_cam.z, fy * mean_cam.y + cy * mean_cam.z};
    float row2[3] = {tu_cam.z, tv_cam.z, mean_cam.z};
    for (int k = 0; k < 3; ++k) {
        out_transform[gid * 9 + 0 * 3 + k] = row0[k];
        out_transform[gid * 9 + 1 * 3 + k] = row1[k];
        out_transform[gid * 9 + 2 * 3 + k] = row2[k];
    }
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
    constant float& img_width,
    constant float& img_height,
    constant float& near,
    constant float& eps2d,
    device const float* valid_in,       // (N,) from forward
    device const float* d_means2d,       // (N,2)
    device const float* d_conics,         // (N,3) a,b,c
    device const float* d_compensations,   // (N,)
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
    float x = mean_cam.x, y = mean_cam.y, z = mean_cam.z;
    float z_safe = max(z, near);

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3x3 M = float3x3(Rq[0] * su, Rq[1] * sv, Rq[2] * EPS_3RD_AXIS);
    float3x3 SigmaWorld = M * transpose(M);
    float3x3 SigmaCam = Rwc * SigmaWorld * RwcT;

    float S00 = SigmaCam[0][0], S11 = SigmaCam[1][1], S22 = SigmaCam[2][2];
    float S01 = SigmaCam[1][0], S02 = SigmaCam[2][0], S12 = SigmaCam[2][1];

    float lim_x = 1.3 * (0.5 * img_width) / fx;
    float lim_y = 1.3 * (0.5 * img_height) / fy;
    float rx = clamp(x / z_safe, -lim_x, lim_x);
    float ry = clamp(y / z_safe, -lim_y, lim_y);
    float free_x = (fabs(x / z_safe) < lim_x) ? 1.0 : 0.0;
    float free_y = (fabs(y / z_safe) < lim_y) ? 1.0 : 0.0;

    float j0 = fx / z_safe;
    float j1 = fy / z_safe;
    float j2 = -fx * rx / z_safe;
    float j3 = -fy * ry / z_safe;

    float a_raw = j0 * j0 * S00 + 2.0 * j0 * j2 * S02 + j2 * j2 * S22;
    float c_raw = j1 * j1 * S11 + 2.0 * j1 * j3 * S12 + j3 * j3 * S22;
    float b_raw = j0 * j1 * S01 + j0 * j3 * S02 + j1 * j2 * S12 + j2 * j3 * S22;

    float a = a_raw + eps2d;
    float c = c_raw + eps2d;
    float b = b_raw;
    float det = a * c - b * b;
    float det2 = det * det;

    // ---- Upstream gradients ----
    float d_u = d_means2d[gid * 2 + 0];
    float d_v = d_means2d[gid * 2 + 1];
    // out_conic = raw_conic / margin^2 (see project_2dgs_forward), so the
    // same constant factor divides its gradient back out.
    float margin_sq = RADIUS_SAFETY_MARGIN * RADIUS_SAFETY_MARGIN;
    float d_conic_a = d_conics[gid * 3 + 0] / margin_sq;
    float d_conic_b = d_conics[gid * 3 + 1] / margin_sq;
    float d_conic_c = d_conics[gid * 3 + 2] / margin_sq;

    // ---- conic = inv([[a,b],[b,c]]) backward (closed-form Jacobian) ----
    float d_a = (-c * c / det2) * d_conic_a + (b * c / det2) * d_conic_b + (-b * b / det2) * d_conic_c;
    float d_b = (2.0 * b * c / det2) * d_conic_a + (-(det + 2.0 * b * b) / det2) * d_conic_b + (2.0 * a * b / det2) * d_conic_c;
    float d_c = (-b * b / det2) * d_conic_a + (a * b / det2) * d_conic_b + (-a * a / det2) * d_conic_c;

    float d_a_raw = d_a;
    float d_c_raw = d_c;
    float d_b_raw = d_b;

    float det_orig = max(a_raw * c_raw - b_raw * b_raw, 0.0);
    float comp = sqrt(clamp(det_orig / max(det, 1e-12), 0.0, 1.0));
    if (comp > 1e-9) {
        float d_comp = d_compensations[gid];
        float d_det_orig = d_comp / (2.0 * comp * det);
        float d_det_blur = -d_comp * comp / (2.0 * det);
        d_a_raw += d_det_orig * c_raw + d_det_blur * c;
        d_c_raw += d_det_orig * a_raw + d_det_blur * a;
        d_b_raw += d_det_orig * (-2.0 * b_raw) + d_det_blur * (-2.0 * b);
    }

    float d_S00 = j0 * j0 * d_a_raw;
    float d_S11 = j1 * j1 * d_c_raw;
    float d_S22 = j2 * j2 * d_a_raw + j3 * j3 * d_c_raw + j2 * j3 * d_b_raw;
    float d_S02 = 2.0 * j0 * j2 * d_a_raw + j0 * j3 * d_b_raw;
    float d_S12 = 2.0 * j1 * j3 * d_c_raw + j1 * j2 * d_b_raw;
    float d_S01 = j0 * j1 * d_b_raw;

    float d_j0 = (2.0 * j0 * S00 + 2.0 * j2 * S02) * d_a_raw + (j1 * S01 + j3 * S02) * d_b_raw;
    float d_j2 = (2.0 * j0 * S02 + 2.0 * j2 * S22) * d_a_raw + (j1 * S12 + j3 * S22) * d_b_raw;
    float d_j1 = (2.0 * j1 * S11 + 2.0 * j3 * S12) * d_c_raw + (j0 * S01 + j2 * S12) * d_b_raw;
    float d_j3 = (2.0 * j1 * S12 + 2.0 * j3 * S22) * d_c_raw + (j0 * S02 + j2 * S22) * d_b_raw;

    float inv_z2 = 1.0 / (z_safe * z_safe);
    float d_x = (-fx * inv_z2 * free_x) * d_j2;
    float d_y = (-fy * inv_z2 * free_y) * d_j3;
    float d_z = (-fx * inv_z2) * d_j0 + (-fy * inv_z2) * d_j1 +
                ((fx * x * inv_z2 / z_safe) * free_x + fx * rx * inv_z2) * d_j2 +
                ((fy * y * inv_z2 / z_safe) * free_y + fy * ry * inv_z2) * d_j3;

    d_x += (fx / z_safe) * d_u;
    d_y += (fy / z_safe) * d_v;
    d_z += (-fx * x / (z_safe * z_safe)) * d_u + (-fy * y / (z_safe * z_safe)) * d_v;

    float3 d_mean_cam_cov = float3(d_x, d_y, d_z);

    float3x3 D = float3x3(float3(d_S00, 0.0, 0.0),
                           float3(d_S01, d_S11, 0.0),
                           float3(d_S02, d_S12, d_S22));
    float3x3 D_world = RwcT * D * Rwc;
    float3x3 D_M = (D_world + transpose(D_world)) * M;

    // ---- Exact ray-splat path: transform (M)/normal backward ----
    float3 t_u = Rq[0] * su;
    float3 t_v = Rq[1] * sv;
    float3 normal_raw = Rq[2];
    float3 cam_pos = -(RwcT * twc_v);
    float3 view_dir = cam_pos - mean;
    float flip_sign = (dot(normal_raw, view_dir) < 0.0) ? -1.0 : 1.0;

    float g00 = d_transform[gid * 9 + 0], g01 = d_transform[gid * 9 + 1], g02 = d_transform[gid * 9 + 2];
    float g10 = d_transform[gid * 9 + 3], g11 = d_transform[gid * 9 + 4], g12 = d_transform[gid * 9 + 5];
    float g20 = d_transform[gid * 9 + 6], g21 = d_transform[gid * 9 + 7], g22 = d_transform[gid * 9 + 8];

    float3 d_tu_cam = float3(fx * g00, fy * g10, cx * g00 + cy * g10 + g20);
    float3 d_tv_cam = float3(fx * g01, fy * g11, cx * g01 + cy * g11 + g21);
    float3 d_meancam_from_transform = float3(fx * g02, fy * g12, cx * g02 + cy * g12 + g22);

    float3 d_t_u = RwcT * d_tu_cam;   // direction, no translation
    float3 d_t_v = RwcT * d_tv_cam;

    float3 d_normal_up = float3(d_normal[gid * 3 + 0], d_normal[gid * 3 + 1], d_normal[gid * 3 + 2]);

    float3 d_mean_cam = d_mean_cam_cov + d_meancam_from_transform;
    float3 d_mean = RwcT * d_mean_cam;
    d_means[gid * 3 + 0] = d_mean.x;
    d_means[gid * 3 + 1] = d_mean.y;
    d_means[gid * 3 + 2] = d_mean.z;

    // ---- Combine covariance-path and ray-splat-path gradients into
    // Rq's columns before applying the shared quat closed-form backward ----
    d_scales[gid * 2 + 0] = dot(D_M[0], Rq[0]) + dot(d_t_u, Rq[0]);
    d_scales[gid * 2 + 1] = dot(D_M[1], Rq[1]) + dot(d_t_v, Rq[1]);

    float3 D_Rq0 = (D_M[0] + d_t_u) * su;
    float3 D_Rq1 = (D_M[1] + d_t_v) * sv;
    float3 D_Rq2 = D_M[2] * EPS_3RD_AXIS + flip_sign * d_normal_up;

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
