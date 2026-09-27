// Helpers shared by several kernels. Not a library of its own: kernels
// pull it in with `#include "common.metal"`, which kernels/_loader.py
// expands textually (torch.mps.compile_shader compiles one source string,
// so the Metal compiler cannot resolve local includes itself).

// sqrt(2 ln 255): the farthest, in sigmas, any gaussian composites before
// its alpha falls below the rasterizers' 1/255 cutoff (at opacity 1).
// Matches reference/tiling_ref.py MAX_SIGMA_EXTENT.
constant float MAX_SIGMA_EXTENT = 3.3290429;

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

// Backward of quat_to_rotmat: dL/d(w, x, y, z) given dL/dR, both in the
// same column-major float3x3 layout quat_to_rotmat returns (dR[k] is the
// gradient of column k).
inline float4 quat_to_rotmat_backward(float4 q, float3x3 dR) {
    float d_R00 = dR[0][0], d_R10 = dR[0][1], d_R20 = dR[0][2];
    float d_R01 = dR[1][0], d_R11 = dR[1][1], d_R21 = dR[1][2];
    float d_R02 = dR[2][0], d_R12 = dR[2][1], d_R22 = dR[2][2];

    float w = q.x, qx = q.y, qy = q.z, qz = q.w;

    float d_w = 2.0 * qz * (d_R10 - d_R01) + 2.0 * qy * (d_R02 - d_R20) + 2.0 * qx * (d_R21 - d_R12);
    float d_qx = 2.0 * qy * (d_R10 + d_R01) + 2.0 * qz * (d_R20 + d_R02) + 2.0 * w * (d_R21 - d_R12) - 4.0 * qx * (d_R11 + d_R22);
    float d_qy = 2.0 * qx * (d_R10 + d_R01) + 2.0 * qz * (d_R21 + d_R12) + 2.0 * w * (d_R02 - d_R20) - 4.0 * qy * (d_R00 + d_R22);
    float d_qz = 2.0 * w * (d_R10 - d_R01) + 2.0 * qx * (d_R20 + d_R02) + 2.0 * qy * (d_R21 + d_R12) - 4.0 * qz * (d_R00 + d_R11);
    return float4(d_w, d_qx, d_qy, d_qz);
}
