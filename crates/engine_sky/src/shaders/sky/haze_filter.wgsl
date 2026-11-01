import package::sky::haze_view::{post, world_at};
import package::sky::piece::sky_piece;

@group(1) @binding(9) var marched: texture_2d<f32>;
@group(1) @binding(10) var haze_history: texture_2d<f32>;

struct HazeTargets {
    @location(0) haze: vec4<f32>,
    @location(1) history: vec4<f32>,
}

const HAZE_CURRENT: f32 = 0.1;

const HAZE_PLANE: f32 = 0.1;

fn reprojected_haze(pixel: vec2<i32>, own: f32) -> vec4<f32> {
    let drawn = sky_piece.haze.yz;
    if drawn.x <= 0.0 {
        return vec4<f32>(0.0);
    }
    let point = world_at(pixel);
    let clip = post.previous_view_projection * vec4<f32>(point.xyz, 1.0);
    if !(clip.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    var expected = 0.0;
    if own > 0.0 {
        expected = 1.0 / max(length(point.xyz - post.previous_eye.xyz), 1.0e-4);
    }
    let tolerance = max(HAZE_PLANE * expected, 1.0e-7);
    let at = uv * drawn - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = vec2<i32>(drawn) - 1;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let tap = textureLoad(haze_history, clamp(first + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let weight = bilinear * saturate(1.25 - abs(tap.w - expected) / tolerance);
            sum += tap.rgb * weight;
            total += weight;
        }
    }
    if total < 1.0e-3 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum / total, total);
}

fn filtered_haze(position: vec4<f32>) -> vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(marched, pixel, 0);
    let own = center.w;
    let tolerance = max(HAZE_PLANE * own, 1.0e-7);
    var low = center.rgb;
    var high = center.rgb;
    var surface = center.rgb;
    var count = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 {
                continue;
            }
            let neighbor = textureLoad(marched,
                clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            low = min(low, neighbor.rgb);
            high = max(high, neighbor.rgb);
            let same = saturate(1.25 - abs(neighbor.w - own) / tolerance);
            surface += neighbor.rgb * same;
            count += same;
        }
    }
    let history = reprojected_haze(pixel, own);
    if history.w <= 0.0 {
        return vec4<f32>(surface / count, own);
    }
    return vec4<f32>(mix(clamp(history.rgb, low, high), center.rgb, HAZE_CURRENT), own);
}

@fragment
fn fs_haze_filter(@builtin(position) position: vec4<f32>) -> HazeTargets {
    let filtered = filtered_haze(position);
    return HazeTargets(filtered, filtered);
}
