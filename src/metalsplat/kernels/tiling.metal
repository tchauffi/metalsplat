#include <metal_stdlib>
using namespace metal;

// Tile binning: turn per-gaussian screen rectangles into (gaussian, tile)
// pairs.
//
// Each gaussian arrives as an axis-aligned pixel-space rectangle
// (xmin, ymin, xmax, ymax) covering everywhere it can composite. How that
// rectangle is found is the projection's business, not binning's: 3DGS
// bounds its conic ellipse (ellipse_rects below), 2DGS bounds the exact
// perspective image of its disk (kernels/project_2dgs.metal). An empty
// rectangle (xmax < xmin, or NaN) touches no tile.
//
// Split into two passes over the same rectangles because the output length
// is data-dependent -- pass 1 counts the tiles each gaussian touches, the
// host prefix-sums those counts to get each gaussian's write offset and the
// total buffer size, and pass 2 writes the pairs.
//
// Pass 2 emits a sort key rather than a tile id: (tile_id << 32) | depth_bits,
// so one ordinary sort of the keys orders pairs by tile and, within a tile,
// front-to-back by depth. The raw float32 bit pattern of a positive float
// already compares correctly as an integer, and every gaussian that gets here
// has depth > near > 0.

// Per-axis half-extents of the projected gaussian, `k` sigmas out, from the
// conic. `k` comes from the host per gaussian: 3 by default, or the
// opacity-aware distance at which the rasterizer's alpha falls below 1/255
// (see reference/tiling_ref.py sigma_extent).
// The conic is the inverse of the 2D covariance, so inverting it back gives
// Sigma2d, whose diagonal is what bounds the ellipse along x and y.
//
// Bounding the ellipse by a circle of radius 3*sqrt(lambda_max) -- which is
// what this used to do, and what the reference 3DGS implementation does --
// is very loose for an elongated gaussian: a thin diagonal splat gets a box
// as wide as it is long. On the garden scene the tight box produces 43%
// fewer (gaussian, tile) pairs, which is less to sort and less for the
// rasterizer to walk per tile.
inline float2 ellipse_half_extents(float3 conic, float k)
{
    float det = conic.x * conic.z - conic.y * conic.y;
    if (det <= 0.0) return float2(0.0);
    // Sigma2d = inv(conic): diagonal entries are conic.z/det and conic.x/det.
    return float2(k * sqrt(max(conic.z / det, 0.0)),
                  k * sqrt(max(conic.x / det, 0.0)));
}

// 3DGS: the rectangle bounding each gaussian's ellipse `extent` sigmas out,
// or an empty one for a culled gaussian (invalid, zero radius, or too faint
// to reach the alpha cutoff anywhere).
kernel void ellipse_rects(
    device const float* means2d,        // (N,2)
    device const float* conics,          // (N,3) a,b,c
    device const float* radii,            // (N,)
    device const float* valid,             // (N,)
    device const float* extent,             // (N,) half-extent in sigmas, 0 = cull
    device float* rects,                     // (N,4) out
    uint gid [[thread_position_in_grid]])
{
    float r = radii[gid];
    float ext = extent[gid];
    if (valid[gid] < 0.5 || r <= 0.0 || ext <= 0.0) {
        rects[gid * 4 + 0] = 0.0; rects[gid * 4 + 1] = 0.0;
        rects[gid * 4 + 2] = -1.0; rects[gid * 4 + 3] = -1.0;
        return;
    }
    float2 h = ellipse_half_extents(
        float3(conics[gid * 3 + 0], conics[gid * 3 + 1], conics[gid * 3 + 2]), ext);
    float mx = means2d[gid * 2 + 0], my = means2d[gid * 2 + 1];
    rects[gid * 4 + 0] = mx - h.x;
    rects[gid * 4 + 1] = my - h.y;
    rects[gid * 4 + 2] = mx + h.x;
    rects[gid * 4 + 3] = my + h.y;
}

inline void tile_bbox(float4 rect,
                      int tiles_x, int tiles_y, float tile_size,
                      thread int& min_tx, thread int& min_ty,
                      thread int& span_x, thread int& span_y)
{
    min_tx = 0; min_ty = 0; span_x = 0; span_y = 0;
    // Written as !(>=) so a NaN coordinate also reads as empty.
    if (!(rect.z >= rect.x) || !(rect.w >= rect.y)) return;

    // Clamp to just outside the tile grid before converting to int, so an
    // unbounded or huge rectangle cannot overflow the conversion. Clamping
    // to [-1, grid] preserves which tiles a rectangle touches: anything
    // entirely off one side stays entirely off it.
    float gx = float(tiles_x) * tile_size, gy = float(tiles_y) * tile_size;
    float x0 = clamp(rect.x, -1.0, gx), x1 = clamp(rect.z, -1.0, gx);
    float y0 = clamp(rect.y, -1.0, gy), y1 = clamp(rect.w, -1.0, gy);

    // Deliberately asymmetric clamping, matching reference/tiling_ref.py:
    // the low edge is clamped up to 0 and the high edge down to the last
    // tile, but neither is clamped at the other end. A rectangle entirely
    // off the left of the image therefore ends up with max < min, hence a
    // negative span, which the max(..., 0) below turns into "touches nothing"
    // -- that is what culls it.
    int lo_x = (int)floor(x0 / tile_size);
    int hi_x = (int)floor(x1 / tile_size);
    int lo_y = (int)floor(y0 / tile_size);
    int hi_y = (int)floor(y1 / tile_size);

    min_tx = max(lo_x, 0);
    min_ty = max(lo_y, 0);
    span_x = max(min(hi_x, tiles_x - 1) - min_tx + 1, 0);
    span_y = max(min(hi_y, tiles_y - 1) - min_ty + 1, 0);
}

inline float4 load_rect(device const float* rects, uint gid)
{
    return float4(rects[gid * 4 + 0], rects[gid * 4 + 1],
                  rects[gid * 4 + 2], rects[gid * 4 + 3]);
}

kernel void tile_counts(
    device const float* rects,          // (N,4) xmin, ymin, xmax, ymax
    device const float* valid,           // (N,)
    constant int& tiles_x,
    constant int& tiles_y,
    constant float& tile_size,
    device int* counts,                    // (N,) out
    uint gid [[thread_position_in_grid]])
{
    if (valid[gid] < 0.5) {
        counts[gid] = 0;
        return;
    }
    int min_tx, min_ty, span_x, span_y;
    tile_bbox(load_rect(rects, gid), tiles_x, tiles_y, tile_size,
              min_tx, min_ty, span_x, span_y);
    counts[gid] = span_x * span_y;
}

kernel void tile_pairs(
    device const float* rects,          // (N,4) xmin, ymin, xmax, ymax
    device const float* depths,          // (N,)
    device const float* valid,            // (N,)
    device const int* offsets,             // (N,) exclusive prefix sum of counts
    constant int& tiles_x,
    constant int& tiles_y,
    constant float& tile_size,
    device long* out_keys,                   // (M,) out
    device int* out_gaussian_ids,             // (M,) out
    uint gid [[thread_position_in_grid]])
{
    if (valid[gid] < 0.5) return;

    int min_tx, min_ty, span_x, span_y;
    tile_bbox(load_rect(rects, gid), tiles_x, tiles_y, tile_size,
              min_tx, min_ty, span_x, span_y);
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
