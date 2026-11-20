import package::engine::post::frame::post;
import package::generated::present::PresentFrame;

@group(0) @binding(2) var presented: texture_2d<f32>;
@group(2) @binding(0) var<uniform> present_frame: PresentFrame;
@group(2) @binding(1) var presented_sampler: sampler;

const GRADE_LUMA: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);
const EDGE_SMALLEST: f32 = 0.0312;
const EDGE_SHARE: f32 = 0.125;
const REDUCE_SHARE: f32 = 0.125;
const REDUCE_SMALLEST: f32 = 0.0078125;
const SPAN_MOST: f32 = 8.0;

fn shown_at(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(presented, presented_sampler, uv, 0.0).rgb;
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
fn fs_present(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(presented));
    let uv = position.xy * post.size.zw;
    var middle: vec3<f32>;
    if all(vec2<f32>(size) == post.size.xy) {
        let pixel = min(vec2<i32>(position.xy), size - 1);
        middle = textureLoad(presented, pixel, 0).rgb;
    } else {
        middle = shown_at(uv);
    }
    var shown = middle;
    if present_frame.antialias.x > 0.5 {
        shown = smoothed(uv, 1.0 / vec2<f32>(size), middle);
    }
    return vec4<f32>(shown, 1.0);
}
