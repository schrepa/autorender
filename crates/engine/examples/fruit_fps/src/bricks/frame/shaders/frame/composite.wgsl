import package::engine::color::encode_srgb;
import package::engine::post::frame::post;
import package::generated::frame::FrameGrade;

@group(0) @binding(2) var scene: texture_2d<f32>;
@group(2) @binding(0) var<uniform> frame_grade: FrameGrade;
@group(2) @binding(1) var scene_sampler: sampler;

const GRADE_MIDDLE: f32 = 0.18;
const GRADE_LUMA: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);
const EDGE_SMALLEST: f32 = 0.0312;
const EDGE_SHARE: f32 = 0.125;
const REDUCE_SHARE: f32 = 0.125;
const REDUCE_SMALLEST: f32 = 0.0078125;
const SPAN_MOST: f32 = 8.0;

fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn graded(color: vec3<f32>) -> vec3<f32> {
    let grade = frame_grade.grade;
    let exposed = max(color, vec3<f32>(0.0)) * grade.x;
    let luma = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let saturated = max(mix(vec3<f32>(luma), exposed, grade.z), vec3<f32>(0.0));
    return GRADE_MIDDLE * pow(saturated / GRADE_MIDDLE, vec3<f32>(grade.y));
}

fn tonemap(exposed: vec3<f32>) -> vec3<f32> {
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let shown = filmic(vec3<f32>(peak)).r;
    let kept = exposed * (shown / max(peak, 1.0e-6));
    let burnt = 1.0 - 1.0 / (max(peak - shown, 0.0) + 1.0);
    return saturate(mix(kept, filmic(exposed), burnt));
}

fn shown_at(uv: vec2<f32>) -> vec3<f32> {
    return tonemap(graded(textureSampleLevel(scene, scene_sampler, uv, 0.0).rgb));
}

fn luma_at(uv: vec2<f32>) -> f32 {
    return dot(shown_at(uv), GRADE_LUMA);
}

fn smoothed(uv: vec2<f32>, texel: vec2<f32>, middle: vec3<f32>) -> vec3<f32> {
    let luma_middle = dot(middle, GRADE_LUMA);
    let north_west = luma_at(uv + vec2<f32>(-0.5, -0.5) * texel);
    let north_east = luma_at(uv + vec2<f32>(0.5, -0.5) * texel);
    let south_west = luma_at(uv + vec2<f32>(-0.5, 0.5) * texel);
    let south_east = luma_at(uv + vec2<f32>(0.5, 0.5) * texel);
    let lowest = min(luma_middle, min(min(north_west, north_east), min(south_west, south_east)));
    let highest = max(luma_middle, max(max(north_west, north_east), max(south_west, south_east)));
    if highest - lowest < max(EDGE_SMALLEST, highest * EDGE_SHARE) {
        return middle;
    }
    var direction = vec2<f32>(
        (south_west + south_east) - (north_west + north_east),
        (north_west + south_west) - (north_east + south_east));
    let reduce = max((north_west + north_east + south_west + south_east) * 0.25 * REDUCE_SHARE,
        REDUCE_SMALLEST);
    let scale = 1.0 / (min(abs(direction.x), abs(direction.y)) + reduce);
    direction = clamp(direction * scale, vec2<f32>(-SPAN_MOST), vec2<f32>(SPAN_MOST)) * texel;
    let near = 0.5 * (shown_at(uv + direction * (1.0 / 3.0 - 0.5))
        + shown_at(uv + direction * (2.0 / 3.0 - 0.5)));
    let far = near * 0.5 + 0.25 * (shown_at(uv - direction * 0.5) + shown_at(uv + direction * 0.5));
    let luma_far = dot(far, GRADE_LUMA);
    return select(far, near, luma_far < lowest || luma_far > highest);
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(scene));
    let uv = position.xy * post.size.zw;
    var middle: vec3<f32>;
    if all(vec2<f32>(size) == post.size.xy) {
        let pixel = min(vec2<i32>(position.xy), size - 1);
        middle = tonemap(graded(textureLoad(scene, pixel, 0).rgb));
    } else {
        middle = shown_at(uv);
    }
    var shown = middle;
    if frame_grade.grade.w > 0.5 {
        shown = smoothed(uv, 1.0 / vec2<f32>(size), middle);
    }
    return vec4<f32>(encode_srgb(shown), 1.0);
}
