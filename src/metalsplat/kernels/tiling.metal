#include <metal_stdlib>
using namespace metal;

// Tile binning: turn projected gaussians into (gaussian, tile) pairs.
//
// Split into two passes over the same geometry because the output length is
// data-dependent -- pass 1 counts the tiles each gaussian touches, the host
// prefix-sums those counts to get each gaussian's write offset and the total
// buffer size, and pass 2 writes the pairs. Recomputing the bounding box in
// pass 2 is cheaper than storing and reloading four int arrays.
//
// Pass 2 emits a sort key rather than a tile id: (tile_id << 32) | depth_bits,
// so one ordinary sort of the keys orders pairs by tile and, within a tile,
// front-to-back by depth. The raw float32 bit pattern of a positive float
// already compares correctly as an integer, and every gaussian that gets here
// has depth > near > 0.

// Per-axis 3-sigma half-extents of the projected gaussian, from the conic.
// The conic is the inverse of the 2D covariance, so inverting it back gives
// Sigma2d, whose diagonal is what bounds the ellipse along x and y.
//
// Bounding the ellipse by a circle of radius 3*sqrt(lambda_max) -- which is
// what this used to do, and what the reference 3DGS implementation does --
// is very loose for an elongated gaussian: a thin diagonal splat gets a box
// as wide as it is long. On the garden scene the tight box produces 43%
// fewer (gaussian, tile) pairs, which is less to sort and less for the
// rasterizer to walk per tile.
inline float2 ellipse_half_extents(float3 conic)
{
    float det = conic.x * conic.z - conic.y * conic.y;
    if (det <= 0.0) return float2(0.0);
    // Sigma2d = inv(conic): diagonal entries are conic.z/det and conic.x/det.
    return float2(3.0 * sqrt(max(conic.z / det, 0.0)),
                  3.0 * sqrt(max(conic.x / det, 0.0)));
}

inline void tile_bbox(float mx, float my, float hw, float hh,
                      int tiles_x, int tiles_y, float tile_size,
                      thread int& min_tx, thread int& min_ty,
                      thread int& span_x, thread int& span_y)
{
    // Deliberately asymmetric clamping, matching reference/tiling_ref.py:
    // the low edge is clamped up to 0 and the high edge down to the last
    // tile, but neither is clamped at the other end. A gaussian entirely off
    // the left of the image therefore ends up with max < min, hence a
    // negative span, which the max(..., 0) below turns into "touches nothing"
    // -- that is what culls it.
    int lo_x = (int)floor((mx - hw) / tile_size);
    int hi_x = (int)floor((mx + hw) / tile_size);
    int lo_y = (int)floor((my - hh) / tile_size);
    int hi_y = (int)floor((my + hh) / tile_size);

    min_tx = max(lo_x, 0);
    min_ty = max(lo_y, 0);
    span_x = max(min(hi_x, tiles_x - 1) - min_tx + 1, 0);
    span_y = max(min(hi_y, tiles_y - 1) - min_ty + 1, 0);
}

kernel void tile_counts(
    device const float* means2d,        // (N,2)
    device const float* conics,          // (N,3) a,b,c
    device const float* radii,            // (N,)
    device const float* valid,             // (N,)
    constant int& tiles_x,
    constant int& tiles_y,
    constant float& tile_size,
    device int* counts,                    // (N,) out
    uint gid [[thread_position_in_grid]])
{
    float r = radii[gid];
    if (valid[gid] < 0.5 || r <= 0.0) {
        counts[gid] = 0;
        return;
    }
    float2 half_extent = ellipse_half_extents(
        float3(conics[gid * 3 + 0], conics[gid * 3 + 1], conics[gid * 3 + 2]));
    int min_tx, min_ty, span_x, span_y;
    tile_bbox(means2d[gid * 2 + 0], means2d[gid * 2 + 1], half_extent.x, half_extent.y,
              tiles_x, tiles_y, tile_size, min_tx, min_ty, span_x, span_y);
    counts[gid] = span_x * span_y;
}

kernel void tile_pairs(
    device const float* means2d,        // (N,2)
    device const float* conics,          // (N,3) a,b,c
    device const float* depths,           // (N,)
    device const float* radii,             // (N,)
    device const float* valid,              // (N,)
    device const int* offsets,              // (N,) exclusive prefix sum of counts
    constant int& tiles_x,
    constant int& tiles_y,
    constant float& tile_size,
    device long* out_keys,                   // (M,) out
    device int* out_gaussian_ids,             // (M,) out
    uint gid [[thread_position_in_grid]])
{
    float r = radii[gid];
    if (valid[gid] < 0.5 || r <= 0.0) return;

    float2 half_extent = ellipse_half_extents(
        float3(conics[gid * 3 + 0], conics[gid * 3 + 1], conics[gid * 3 + 2]));
    int min_tx, min_ty, span_x, span_y;
    tile_bbox(means2d[gid * 2 + 0], means2d[gid * 2 + 1], half_extent.x, half_extent.y,
              tiles_x, tiles_y, tile_size, min_tx, min_ty, span_x, span_y);
    if (span_x <= 0 || span_y <= 0) return;

    // Reinterpret, not convert: the bit pattern is the sortable part of the key.
    long depth_bits = (long)as_type<uint>(depths[gid]);
    int base = offsets[gid];
    int k = 0;

    // Row-major over the gaussian's tile block (y outer, x inner), matching
    // the reference's local_i / row_span decomposition. Pairs from one
    // gaussian always land in distinct tiles, so they never tie on the key.
    for (int ty = 0; ty < span_y; ++ty) {
        int row_base = (min_ty + ty) * tiles_x + min_tx;
        for (int tx = 0; tx < span_x; ++tx) {
            out_keys[base + k] = ((long)(row_base + tx) << 32) | depth_bits;
            out_gaussian_ids[base + k] = (int)gid;
            ++k;
        }
    }
}
