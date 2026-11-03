import package::engine::color::encode_srgb;
import package::generated::ui::UiRect;
@group(1) @binding(0) var<storage, read> instances: array<UiRect>;
@group(1) @binding(1) var sheet: texture_2d<f32>;
@group(1) @binding(2) var sheet_sampler: sampler;
struct Corner {
    @builtin(position) clip: vec4<f32>,
    @location(0) pixel: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) bounds: vec4<f32>,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) border: vec4<f32>,
    @location(5) @interpolate(flat) shape: vec4<f32>,
    @location(6) @interpolate(flat) limit: vec4<f32>,
}
@vertex fn vs_rect(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Corner {
    let quad = instances[index];
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
    var out: Corner;
    out.clip = vec4<f32>(mix(quad.place.xy, quad.place.zw, unit), 0.0, 1.0);
    out.pixel = mix(quad.rect.xy, quad.rect.zw, unit);
    out.uv = quad.uv.xy + unit * quad.uv.zw;
    out.bounds = quad.rect;
    out.color = quad.color;
    out.border = quad.border;
    out.shape = quad.shape;
    out.limit = quad.clip;
    return out;
}
fn shaded(corner: Corner, fill: vec4<f32>) -> vec4<f32> {
    let half_size = (corner.bounds.zw - corner.bounds.xy) * 0.5;
    let radius = min(corner.shape.x, min(half_size.x, half_size.y));
    let near = abs(corner.pixel - corner.bounds.xy - half_size) - half_size + radius;
    let edge = length(max(near, vec2<f32>(0.0))) + min(max(near.x, near.y), 0.0) - radius;
    let inner = clamp(0.5 - edge - corner.shape.y, 0.0, 1.0);
    let mixed = select(mix(corner.border, fill, inner), fill, corner.shape.y <= 0.0);
    let kept = all(corner.pixel >= corner.limit.xy) && all(corner.pixel < corner.limit.zw);
    let shown = clamp(0.5 - edge, 0.0, 1.0) * f32(kept);
    return vec4<f32>(mixed.rgb, mixed.a * shown);
}
@fragment fn fs_rect(corner: Corner) -> @location(0) vec4<f32> {
    return shaded(corner, corner.color);
}
@fragment fn fs_image(corner: Corner) -> @location(0) vec4<f32> {
    let texel = textureSample(sheet, sheet_sampler, corner.uv);
    return shaded(corner, vec4<f32>(encode_srgb(texel.rgb), texel.a) * corner.color);
}
