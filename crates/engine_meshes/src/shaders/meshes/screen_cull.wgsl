const F32_MAX: f32 = 3.4028235e38f;
const SUBPIXEL_MARGIN: f32 = 0.0625;
const HIDDEN_SCALE: f32 = 0.9999847412109375;

struct Projected {
    low: vec2<f32>,
    high: vec3<f32>,
    valid: bool,
}

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
    projected.valid = all(projected.low >= vec2<f32>(-F32_MAX))
        && all(projected.high.xy <= vec2<f32>(F32_MAX));
    return projected;
}

fn covers_no_pixel(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        size: vec2<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    if !projected.valid || !(all(projected.low >= vec2<f32>(-4.0))
            && all(projected.high.xy <= vec2<f32>(4.0))) {
        return false;
    }
    let first = vec2<f32>(projected.low.x * 0.5 + 0.5, 0.5 - projected.high.y * 0.5) * size
        - SUBPIXEL_MARGIN;
    let last = vec2<f32>(projected.high.x * 0.5 + 0.5, 0.5 - projected.low.y * 0.5) * size
        + SUBPIXEL_MARGIN;
    let centers = floor(last - 0.5) - ceil(first - 0.5);
    return centers.x < 0.0 || centers.y < 0.0;
}

fn hidden(clip_from_box: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>,
        pyramid: texture_2d<f32>, viewport: vec4<f32>) -> bool {
    let projected = project_box(clip_from_box, low, high);
    if !projected.valid || !(projected.high.z >= 0.0 && projected.high.z <= 1.0) {
        return false;
    }
    let size = viewport.xy;
    let first = clamp(floor(vec2<f32>(projected.low.x * 0.5 + 0.5,
        0.5 - projected.high.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
    let last = clamp(floor(vec2<f32>(projected.high.x * 0.5 + 0.5,
        0.5 - projected.low.y * 0.5) * size), vec2<f32>(0.0), size - 1.0);
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
