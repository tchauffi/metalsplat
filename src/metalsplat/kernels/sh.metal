// Spherical-harmonics color evaluation, degree <= 3 (up to 16 coefficients
// per channel), forward + backward. Mirrors metalsplat.reference.sh_ref.eval_sh
// exactly -- that module is the oracle these kernels are tested against.
//
// `num_coeffs` is the per-channel coefficient count of the buffer, passed in
// rather than hard-coded, so a degree-2 model still stores 9 per channel
// instead of padding to 16. `active_degree` gates which bands are evaluated
// and must satisfy (active_degree+1)^2 <= num_coeffs.
#include <metal_stdlib>
using namespace metal;

constant float SH_C0 = 0.28209479177387814;
constant float SH_C1 = 0.4886025119029199;
constant float SH_C2_0 = 1.0925484305920792;
constant float SH_C2_1 = -1.0925484305920792;
constant float SH_C2_2 = 0.31539156525252005;
constant float SH_C2_3 = -1.0925484305920792;
constant float SH_C2_4 = 0.5462742152960396;
constant float SH_C3_0 = -0.5900435899266435;
constant float SH_C3_1 = 2.890611442640554;
constant float SH_C3_2 = -0.4570457994644658;
constant float SH_C3_3 = 0.3731763325901154;
constant float SH_C3_4 = -0.4570457994644658;
constant float SH_C3_5 = 1.445305721320277;
constant float SH_C3_6 = -0.5900435899266435;

kernel void sh_forward(
    device const float* sh,      // (N, num_coeffs, 3)
    device const float* dirs,     // (N, 3) unit vectors
    constant int& active_degree,   // evaluate only up to this degree (0..3)
    constant int& num_coeffs,       // per-channel coefficients stored
    device float* out_color,         // (N, 3)
    uint gid [[thread_position_in_grid]])
{
    float x = dirs[gid * 3 + 0], y = dirs[gid * 3 + 1], z = dirs[gid * 3 + 2];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    device const float* c = sh + gid * num_coeffs * 3;

    // Terms above active_degree are skipped entirely, so their
    // coefficients receive exactly zero gradient in the backward pass
    // rather than merely being initialised to zero -- that's what makes
    // progressive SH growth an actual constraint during training.
    for (int ch = 0; ch < 3; ch++) {
        float result = SH_C0 * c[0 * 3 + ch];
        if (active_degree >= 1) {
            result += -SH_C1 * y * c[1 * 3 + ch] + SH_C1 * z * c[2 * 3 + ch] - SH_C1 * x * c[3 * 3 + ch];
        }
        if (active_degree >= 2) {
            result += SH_C2_0 * xy * c[4 * 3 + ch]
                     + SH_C2_1 * yz * c[5 * 3 + ch]
                     + SH_C2_2 * (2.0 * zz - xx - yy) * c[6 * 3 + ch]
                     + SH_C2_3 * xz * c[7 * 3 + ch]
                     + SH_C2_4 * (xx - yy) * c[8 * 3 + ch];
        }
        if (active_degree >= 3) {
            result += SH_C3_0 * y * (3.0 * xx - yy) * c[9 * 3 + ch]
                     + SH_C3_1 * xy * z * c[10 * 3 + ch]
                     + SH_C3_2 * y * (4.0 * zz - xx - yy) * c[11 * 3 + ch]
                     + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c[12 * 3 + ch]
                     + SH_C3_4 * x * (4.0 * zz - xx - yy) * c[13 * 3 + ch]
                     + SH_C3_5 * z * (xx - yy) * c[14 * 3 + ch]
                     + SH_C3_6 * x * (xx - 3.0 * yy) * c[15 * 3 + ch];
        }
        out_color[gid * 3 + ch] = result;
    }
}

kernel void sh_backward(
    device const float* sh,      // (N, num_coeffs, 3)
    device const float* dirs,     // (N, 3)
    device const float* d_color,   // (N, 3)
    constant int& active_degree,    // must match the forward pass
    constant int& num_coeffs,        // must match the forward pass
    device float* d_sh,               // (N, num_coeffs, 3)
    device float* d_dirs,             // (N, 3)
    uint gid [[thread_position_in_grid]])
{
    float x = dirs[gid * 3 + 0], y = dirs[gid * 3 + 1], z = dirs[gid * 3 + 2];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    device const float* c = sh + gid * num_coeffs * 3;
    device const float* dc = d_color + gid * 3;
    device float* d_c = d_sh + gid * num_coeffs * 3;

    float d_x = 0.0, d_y = 0.0, d_z = 0.0;

    for (int ch = 0; ch < 3; ch++) {
        float g = dc[ch];

        d_c[0 * 3 + ch] = SH_C0 * g;

        // Inactive degrees contribute nothing to the forward, so both
        // their own gradient and their share of d_dirs must be zero.
        float d1 = (active_degree >= 1) ? 1.0 : 0.0;
        float d2 = (active_degree >= 2) ? 1.0 : 0.0;
        float d3 = (active_degree >= 3) ? 1.0 : 0.0;

        if (num_coeffs > 1) {
            d_c[1 * 3 + ch] = d1 * -SH_C1 * y * g;
            d_c[2 * 3 + ch] = d1 * SH_C1 * z * g;
            d_c[3 * 3 + ch] = d1 * -SH_C1 * x * g;
        }
        if (num_coeffs > 4) {
            d_c[4 * 3 + ch] = d2 * SH_C2_0 * xy * g;
            d_c[5 * 3 + ch] = d2 * SH_C2_1 * yz * g;
            d_c[6 * 3 + ch] = d2 * SH_C2_2 * (2.0 * zz - xx - yy) * g;
            d_c[7 * 3 + ch] = d2 * SH_C2_3 * xz * g;
            d_c[8 * 3 + ch] = d2 * SH_C2_4 * (xx - yy) * g;
        }
        if (num_coeffs > 9) {
            d_c[9 * 3 + ch] = d3 * SH_C3_0 * y * (3.0 * xx - yy) * g;
            d_c[10 * 3 + ch] = d3 * SH_C3_1 * xy * z * g;
            d_c[11 * 3 + ch] = d3 * SH_C3_2 * y * (4.0 * zz - xx - yy) * g;
            d_c[12 * 3 + ch] = d3 * SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * g;
            d_c[13 * 3 + ch] = d3 * SH_C3_4 * x * (4.0 * zz - xx - yy) * g;
            d_c[14 * 3 + ch] = d3 * SH_C3_5 * z * (xx - yy) * g;
            d_c[15 * 3 + ch] = d3 * SH_C3_6 * x * (xx - 3.0 * yy) * g;
        }

        float sh1 = 0.0, sh2 = 0.0, sh3 = 0.0;
        if (num_coeffs > 1) {
            sh1 = d1 * c[1 * 3 + ch]; sh2 = d1 * c[2 * 3 + ch]; sh3 = d1 * c[3 * 3 + ch];
        }
        float sh4 = 0.0, sh5 = 0.0, sh6 = 0.0, sh7 = 0.0, sh8 = 0.0;
        if (num_coeffs > 4) {
            sh4 = d2 * c[4 * 3 + ch]; sh5 = d2 * c[5 * 3 + ch]; sh6 = d2 * c[6 * 3 + ch];
            sh7 = d2 * c[7 * 3 + ch]; sh8 = d2 * c[8 * 3 + ch];
        }
        float s9 = 0.0, s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0, s14 = 0.0, s15 = 0.0;
        if (num_coeffs > 9) {
            s9 = d3 * c[9 * 3 + ch];   s10 = d3 * c[10 * 3 + ch]; s11 = d3 * c[11 * 3 + ch];
            s12 = d3 * c[12 * 3 + ch]; s13 = d3 * c[13 * 3 + ch]; s14 = d3 * c[14 * 3 + ch];
            s15 = d3 * c[15 * 3 + ch];
        }

        float df_dx = -SH_C1 * sh3
                     + SH_C2_0 * y * sh4 + SH_C2_2 * (-2.0 * x) * sh6 + SH_C2_3 * z * sh7 + SH_C2_4 * (2.0 * x) * sh8
                     + SH_C3_0 * (6.0 * xy) * s9
                     + SH_C3_1 * yz * s10
                     + SH_C3_2 * (-2.0 * xy) * s11
                     + SH_C3_3 * (-6.0 * xz) * s12
                     + SH_C3_4 * (4.0 * zz - 3.0 * xx - yy) * s13
                     + SH_C3_5 * (2.0 * xz) * s14
                     + SH_C3_6 * (3.0 * xx - 3.0 * yy) * s15;
        float df_dy = -SH_C1 * sh1
                     + SH_C2_0 * x * sh4 + SH_C2_1 * z * sh5 + SH_C2_2 * (-2.0 * y) * sh6 + SH_C2_4 * (-2.0 * y) * sh8
                     + SH_C3_0 * (3.0 * xx - 3.0 * yy) * s9
                     + SH_C3_1 * xz * s10
                     + SH_C3_2 * (4.0 * zz - xx - 3.0 * yy) * s11
                     + SH_C3_3 * (-6.0 * yz) * s12
                     + SH_C3_4 * (-2.0 * xy) * s13
                     + SH_C3_5 * (-2.0 * yz) * s14
                     + SH_C3_6 * (-6.0 * xy) * s15;
        float df_dz = SH_C1 * sh2
                     + SH_C2_1 * y * sh5 + SH_C2_2 * (4.0 * z) * sh6 + SH_C2_3 * x * sh7
                     + SH_C3_1 * xy * s10
                     + SH_C3_2 * (8.0 * yz) * s11
                     + SH_C3_3 * (6.0 * zz - 3.0 * xx - 3.0 * yy) * s12
                     + SH_C3_4 * (8.0 * xz) * s13
                     + SH_C3_5 * (xx - yy) * s14;

        d_x += df_dx * g;
        d_y += df_dy * g;
        d_z += df_dz * g;
    }

    d_dirs[gid * 3 + 0] = d_x;
    d_dirs[gid * 3 + 1] = d_y;
    d_dirs[gid * 3 + 2] = d_z;
}
