// Forward + backward projection of 3D gaussians to 2D screen space.
// Mirrors metalsplat.reference.project_ref.project_gaussians exactly; that
// module is the spec and the numerical oracle these kernels are tested
// against. See its docstring for the math and conventions.
#include <metal_stdlib>
using namespace metal;

// Builds a float3x3 from a row-major flat 9-element buffer, i.e. flat[i*3+j]
// is (row i, col j). Metal's float3x3(c0, c1, c2) constructor takes column
// vectors, so column k is (flat[k], flat[3+k], flat[6+k]).
inline float3x3 mat3_from_rowmajor(constant float* flat) {
    return float3x3(float3(flat[0], flat[3], flat[6]),
                     float3(flat[1], flat[4], flat[7]),
                     float3(flat[2], flat[5], flat[8]));
}

// Rotation matrix from a unit quaternion (w, x, y, z), matching
// metalsplat.utils.quaternion.quat_to_rotmat.
inline float3x3 quat_to_rotmat(float w, float x, float y, float z) {
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    float3 col0 = float3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
    float3 col1 = float3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
    float3 col2 = float3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
    return float3x3(col0, col1, col2);
}

kernel void project_forward(
    device const float* means,        // (N,3)
    device const float* scales,       // (N,3)
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
    device float* out_conics,             // (N,3) a,b,c
    device float* out_radii,               // (N,)
    device float* out_valid,                // (N,) 1.0 / 0.0
    uint gid [[thread_position_in_grid]])
{
    float3 mean = float3(means[gid * 3 + 0], means[gid * 3 + 1], means[gid * 3 + 2]);
    float3 scale = float3(scales[gid * 3 + 0], scales[gid * 3 + 1], scales[gid * 3 + 2]);
    float4 q = float4(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]);

    float3x3 Rwc = mat3_from_rowmajor(rwc);
    float3 twc_v = float3(twc[0], twc[1], twc[2]);

    float3 mean_cam = Rwc * mean + twc_v;
    float x = mean_cam.x, y = mean_cam.y, z = mean_cam.z;
    out_depths[gid] = z;

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3x3 M = float3x3(Rq[0] * scale.x, Rq[1] * scale.y, Rq[2] * scale.z);
    float3x3 SigmaWorld = M * transpose(M);
    float3x3 SigmaCam = Rwc * SigmaWorld * transpose(Rwc);

    bool in_front = z > near;
    float z_safe = max(z, near);

    float S00 = SigmaCam[0][0], S11 = SigmaCam[1][1], S22 = SigmaCam[2][2];
    float S01 = SigmaCam[1][0], S02 = SigmaCam[2][0], S12 = SigmaCam[2][1];

    float j0 = fx / z_safe;
    float j1 = fy / z_safe;
    float j2 = -fx * x / (z_safe * z_safe);
    float j3 = -fy * y / (z_safe * z_safe);

    float a_raw = j0 * j0 * S00 + 2.0 * j0 * j2 * S02 + j2 * j2 * S22;
    float c_raw = j1 * j1 * S11 + 2.0 * j1 * j3 * S12 + j3 * j3 * S22;
    float b_raw = j0 * j1 * S01 + j0 * j3 * S02 + j1 * j2 * S12 + j2 * j3 * S22;

    float a = a_raw + eps2d;
    float c = c_raw + eps2d;
    float b = b_raw;

    float det = a * c - b * b;
    float det_safe = max(det, 1e-12);
    out_conics[gid * 3 + 0] = c / det_safe;
    out_conics[gid * 3 + 1] = -b / det_safe;
    out_conics[gid * 3 + 2] = a / det_safe;

    float mid = 0.5 * (a + c);
    float disc = max(mid * mid - det, 0.0);
    float lambda_max = mid + sqrt(disc);
    float radius = ceil(3.0 * sqrt(max(lambda_max, 0.0)));

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
}

kernel void project_backward(
    device const float* means,        // (N,3)
    device const float* scales,       // (N,3)
    device const float* quats,        // (N,4) w,x,y,z
    constant float* rwc,               // (9,) row-major
    constant float* twc,                // (3,)
    constant float& fx,
    constant float& fy,
    constant float& cx,
    constant float& cy,
    constant float& near,
    constant float& eps2d,
    device const float* valid_in,       // (N,) from forward
    device const float* d_means2d,       // (N,2)
    device const float* d_conics,         // (N,3) a,b,c
    device float* d_means,                 // (N,3)
    device float* d_scales,                 // (N,3)
    device float* d_quats,                   // (N,4)
    uint gid [[thread_position_in_grid]])
{
    if (valid_in[gid] < 0.5) {
        d_means[gid * 3 + 0] = 0.0; d_means[gid * 3 + 1] = 0.0; d_means[gid * 3 + 2] = 0.0;
        d_scales[gid * 3 + 0] = 0.0; d_scales[gid * 3 + 1] = 0.0; d_scales[gid * 3 + 2] = 0.0;
        d_quats[gid * 4 + 0] = 0.0; d_quats[gid * 4 + 1] = 0.0;
        d_quats[gid * 4 + 2] = 0.0; d_quats[gid * 4 + 3] = 0.0;
        return;
    }

    // ---- Recompute the forward pass (see project_forward) ----
    float3 mean = float3(means[gid * 3 + 0], means[gid * 3 + 1], means[gid * 3 + 2]);
    float3 scale = float3(scales[gid * 3 + 0], scales[gid * 3 + 1], scales[gid * 3 + 2]);
    float4 q = float4(quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]);

    float3x3 Rwc = mat3_from_rowmajor(rwc);
    float3 twc_v = float3(twc[0], twc[1], twc[2]);
    float3x3 RwcT = transpose(Rwc);

    float3 mean_cam = Rwc * mean + twc_v;
    float x = mean_cam.x, y = mean_cam.y, z = mean_cam.z;
    float z_safe = max(z, near);

    float3x3 Rq = quat_to_rotmat(q.x, q.y, q.z, q.w);
    float3x3 M = float3x3(Rq[0] * scale.x, Rq[1] * scale.y, Rq[2] * scale.z);
    float3x3 SigmaWorld = M * transpose(M);
    float3x3 SigmaCam = Rwc * SigmaWorld * RwcT;

    float S00 = SigmaCam[0][0], S11 = SigmaCam[1][1], S22 = SigmaCam[2][2];
    float S01 = SigmaCam[1][0], S02 = SigmaCam[2][0], S12 = SigmaCam[2][1];

    float j0 = fx / z_safe;
    float j1 = fy / z_safe;
    float j2 = -fx * x / (z_safe * z_safe);
    float j3 = -fy * y / (z_safe * z_safe);

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
    float d_conic_a = d_conics[gid * 3 + 0];
    float d_conic_b = d_conics[gid * 3 + 1];
    float d_conic_c = d_conics[gid * 3 + 2];

    // ---- conic = inv([[a,b],[b,c]]) backward (closed-form Jacobian) ----
    float d_a = (-c * c / det2) * d_conic_a + (b * c / det2) * d_conic_b + (-b * b / det2) * d_conic_c;
    float d_b = (2.0 * b * c / det2) * d_conic_a + (-(det + 2.0 * b * b) / det2) * d_conic_b + (2.0 * a * b / det2) * d_conic_c;
    float d_c = (-b * b / det2) * d_conic_a + (a * b / det2) * d_conic_b + (-a * a / det2) * d_conic_c;

    float d_a_raw = d_a;
    float d_c_raw = d_c;
    float d_b_raw = d_b;

    // ---- a_raw/b_raw/c_raw = J Sigma_cam J^T backward ----
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

    // ---- Jacobian entries -> d_x, d_y, d_z (partial) ----
    float d_x = (-fx / (z_safe * z_safe)) * d_j2;
    float d_y = (-fy / (z_safe * z_safe)) * d_j3;
    float d_z = (-fx / (z_safe * z_safe)) * d_j0 + (-fy / (z_safe * z_safe)) * d_j1 +
                (2.0 * fx * x / (z_safe * z_safe * z_safe)) * d_j2 +
                (2.0 * fy * y / (z_safe * z_safe * z_safe)) * d_j3;

    // ---- means2d = (fx*x/z + cx, fy*y/z + cy) backward ----
    d_x += (fx / z_safe) * d_u;
    d_y += (fy / z_safe) * d_v;
    d_z += (-fx * x / (z_safe * z_safe)) * d_u + (-fy * y / (z_safe * z_safe)) * d_v;

    float3 d_mean_cam = float3(d_x, d_y, d_z);

    // ---- SigmaCam = Rwc SigmaWorld Rwc^T backward: dL/dSigmaWorld = Rwc^T D Rwc ----
    // D must be non-symmetric here: the forward pass only ever reads each
    // off-diagonal entry of SigmaCam from one specific (row, col) slot
    // (S01/S02/S12 above), never its mirror, so gradient only flows back
    // into that same slot -- mirroring it too would double-count.
    float3x3 D = float3x3(float3(d_S00, 0.0, 0.0),
                           float3(d_S01, d_S11, 0.0),
                           float3(d_S02, d_S12, d_S22));
    float3x3 D_world = RwcT * D * Rwc;

    // ---- SigmaWorld = M M^T backward: dL/dM = (G + G^T) M, G = D_world ----
    // (D_world is not symmetric here since D isn't, so the "2*G*M" shortcut
    // that holds for symmetric G doesn't apply -- use the general form.)
    float3x3 D_M = (D_world + transpose(D_world)) * M;

    // ---- M columns = Rq columns * scale backward ----
    d_scales[gid * 3 + 0] = dot(D_M[0], Rq[0]);
    d_scales[gid * 3 + 1] = dot(D_M[1], Rq[1]);
    d_scales[gid * 3 + 2] = dot(D_M[2], Rq[2]);

    float3x3 D_Rq = float3x3(D_M[0] * scale.x, D_M[1] * scale.y, D_M[2] * scale.z);

    // ---- Rq(quat) backward ----
    float d_R00 = D_Rq[0][0], d_R10 = D_Rq[0][1], d_R20 = D_Rq[0][2];
    float d_R01 = D_Rq[1][0], d_R11 = D_Rq[1][1], d_R21 = D_Rq[1][2];
    float d_R02 = D_Rq[2][0], d_R12 = D_Rq[2][1], d_R22 = D_Rq[2][2];

    float w = q.x, qx = q.y, qy = q.z, qz = q.w;

    float d_w = 2.0 * qz * (d_R10 - d_R01) + 2.0 * qy * (d_R02 - d_R20) + 2.0 * qx * (d_R21 - d_R12);
    float d_qx = 2.0 * qy * (d_R10 + d_R01) + 2.0 * qz * (d_R20 + d_R02) + 2.0 * w * (d_R21 - d_R12) - 4.0 * qx * (d_R11 + d_R22);
    float d_qy = 2.0 * qx * (d_R10 + d_R01) + 2.0 * qz * (d_R21 + d_R12) + 2.0 * w * (d_R02 - d_R20) - 4.0 * qy * (d_R00 + d_R22);
    float d_qz = 2.0 * w * (d_R10 - d_R01) + 2.0 * qx * (d_R20 + d_R02) + 2.0 * qy * (d_R21 + d_R12) - 4.0 * qz * (d_R00 + d_R11);

    d_quats[gid * 4 + 0] = d_w;
    d_quats[gid * 4 + 1] = d_qx;
    d_quats[gid * 4 + 2] = d_qy;
    d_quats[gid * 4 + 3] = d_qz;

    // ---- mean_cam = Rwc*mean + twc backward: dL/dmean = Rwc^T d_mean_cam ----
    float3 d_mean = RwcT * d_mean_cam;
    d_means[gid * 3 + 0] = d_mean.x;
    d_means[gid * 3 + 1] = d_mean.y;
    d_means[gid * 3 + 2] = d_mean.z;
}
