// Screen tests of a box's projection that GPU presentation's culling and
// virtual geometry's cluster selection share: whether it covers no pixel
// center (sub-pixel culling), and whether a depth pyramid hides it
// (occlusion culling). Modules listing this file declare F32_MAX.

// Pixels the projected bounds are widened by before looking for pixel
// centers: covers the rasterizer's snapping of vertices to its subpixel grid
// and rounding differences between this projection and the vertex shaders'.
const SUBPIXEL_MARGIN: f32 = 0.0625;
// With reversed depth nearer is larger. A box counts as
// hidden when its nearest depth lies below the pyramid's farthest times this
// scale: a relative margin (2^-16) well beyond float depth steps and the
// projection's rounding at every distance, so boxes touching what is in
// front of them (a sign on a wall) keep drawing in phase one, in the same
// order as without occlusion culling.
const HIDDEN_SCALE: f32 = 0.9999847412109375;

// The screen rectangle (normalized device x and y, lowest then highest) and
// nearest reversed depth of a box; `valid` is false for boxes reaching the
// eye plane, or whose rectangle is not finite.
struct Projected {
    low: vec2<f32>,
    high: vec3<f32>,
    valid: bool,
}

// The projection of the box `low` to `high` under `clip_from_box`.
fn project_box(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>) -> Projected {
    var projected: Projected;
    projected.low = vec2<f32>(F32_MAX);
    projected.high = vec3<f32>(-F32_MAX);
    for (var corner = 0u; corner < 8u; corner++) {
        let position = vec3<f32>(select(low.x, high.x, (corner & 1u) != 0u),
            select(low.y, high.y, (corner & 2u) != 0u),
            select(low.z, high.z, (corner & 4u) != 0u));
        let clip = clip_from_box * vec4<f32>(position, 1.0);
        if !(clip.w > 1.0e-6) {
            return projected;
        }
        let ndc = clip.xyz / clip.w;
        projected.low = min(projected.low, ndc.xy);
        projected.high = max(projected.high, ndc);
    }
    // Non-finite corners fail these comparisons.
    projected.valid = all(projected.low >= vec2<f32>(-F32_MAX))
        && all(projected.high.xy <= vec2<f32>(F32_MAX));
    return projected;
}

// Whether the box `low` to `high` under `clip_from_box` covers no pixel
// center of a viewport `size` pixels large: its screen rectangle, widened
// by SUBPIXEL_MARGIN, holds none on some axis, so no triangle within it can
// produce a fragment. Boxes reaching the eye plane or far outside the
// viewport, and non-finite ones, cover some.
fn covers_no_pixel(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        size: vec2<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    // Within a few viewports of the screen, pixel coordinates keep ample
    // subpixel precision.
    if !projected.valid || !(all(projected.low >= vec2<f32>(-4.0))
            && all(projected.high.xy <= vec2<f32>(4.0))) {
        return false;
    }
    // Pixel coordinates, rows downward; centers sit at i + 0.5.
    let first = vec2<f32>(projected.low.x * 0.5 + 0.5, 0.5 - projected.high.y * 0.5) * size
        - SUBPIXEL_MARGIN;
    let last = vec2<f32>(projected.high.x * 0.5 + 0.5, 0.5 - projected.low.y * 0.5) * size
        + SUBPIXEL_MARGIN;
    let centers = floor(last - 0.5) - ceil(first - 0.5);
    return centers.x < 0.0 || centers.y < 0.0;
}

// Whether `pyramid` hides the box `low` to `high` under `clip_from_box`:
// the farthest depths of each 2x2 block of a viewport `viewport.xy` pixels
// large, then of each 2x2 block of the level below, `viewport.z` levels in
// all. A box is hidden when its nearest depth lies beyond the farthest depth
// of every pyramid texel its screen rectangle touches; boxes reaching the
// eye plane, or too large for the pyramid, are not.
fn hidden(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        pyramid: texture_2d<f32>, viewport: vec4<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    if !projected.valid || !(projected.high.z >= 0.0 && projected.high.z <= 1.0) {
        return false;
    }
    // Pixels the rectangle touches, rows downward, within the viewport.
    let size = viewport.xy;
    let first = clamp(floor(vec2<f32>(projected.low.x * 0.5 + 0.5,
        0.5 - projected.high.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
    let last = clamp(floor(vec2<f32>(projected.high.x * 0.5 + 0.5,
        0.5 - projected.low.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
    // Level-zero texels are 2x2 pixels; each level halves them again. The
    // first level where the rectangle spans at most 2x2 texels.
    var a = vec2<u32>(first) / 2u;
    var b = vec2<u32>(last) / 2u;
    let levels = u32(viewport.z);
    var mip = 0u;
    loop {
        if (b.x - a.x <= 1u && b.y - a.y <= 1u) || mip + 1u >= levels {
            break;
        }
        a = a / 2u;
        b = b / 2u;
        mip += 1u;
    }
    if b.x - a.x > 1u || b.y - a.y > 1u {
        return false;
    }
    let farthest = min(
        min(textureLoad(pyramid, a, mip).r, textureLoad(pyramid, vec2<u32>(b.x, a.y), mip).r),
        min(textureLoad(pyramid, vec2<u32>(a.x, b.y), mip).r, textureLoad(pyramid, b, mip).r));
    return projected.high.z < farthest * HIDDEN_SCALE;
}
