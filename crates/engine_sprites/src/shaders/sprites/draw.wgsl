import package::engine::view::camera;
import package::engine::shading::{ViewPixel, flat};
import package::generated::sprites::SpriteInstance;
import package::generated::targets::{ViewTargets, view_targets};
@group(1) @binding(0) var<storage, read> instances: array<SpriteInstance>;
@group(1) @binding(1) var sheet: texture_2d<f32>;
@group(1) @binding(2) var sheet_sampler: sampler;
struct Corner { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) tint: vec4<f32> }
@vertex fn vs_sprite(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Corner {
    let sprite = instances[index];
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    let world = vec4<f32>(sprite.place.xy + (unit - 0.5) * sprite.place.w, sprite.place.z, 1.0);
    var out: Corner;
    out.clip = camera.view_projection * world;
    out.uv = sprite.rect.xy + unit * sprite.rect.zw;
    out.tint = sprite.tint;
    return out;
}
@fragment fn fs_sprite(corner: Corner) -> ViewTargets {
    let texel = textureSample(sheet, sheet_sampler, corner.uv) * corner.tint;
    if texel.a < 0.5 { discard; }
    return view_targets(flat(texel), ViewPixel(corner.clip, corner.clip));
}
