import package::engine::drawn::{drawn_records, drawn_shown};
import package::engine::shading::flat;
import package::engine::view::{camera, view_pixel};
import package::face::text::glyph_coverage;
import package::generated::targets::{ViewTargets, view_targets};
import package::generated::text::{GlyphInstance, LabelGlyph};
@group(1) @binding(0) var<storage, read> instances: array<GlyphInstance>;
@group(1) @binding(1) var<storage, read> labels: array<LabelGlyph>;
struct Corner { @builtin(position) clip: vec4<f32>, @location(0) texel: vec2<f32>, @location(1) color: vec4<f32> }
@vertex fn vs_text(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Corner {
    let glyph = instances[index];
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
    var out: Corner;
    out.clip = vec4<f32>(mix(glyph.place.xy, glyph.place.zw, unit), 0.0, 1.0);
    out.texel = glyph.rect.xy + unit * glyph.rect.zw;
    out.color = glyph.color;
    return out;
}
@fragment fn fs_text(corner: Corner) -> @location(0) vec4<f32> {
    let coverage = glyph_coverage(vec2<i32>(corner.texel));
    return vec4<f32>(corner.color.rgb, corner.color.a * coverage);
}
struct Placed { @builtin(position) clip: vec4<f32>, @location(0) texel: vec2<f32>, @location(1) color: vec4<f32>, @location(2) previous: vec4<f32> }
fn facing(lens: mat4x4<f32>, origin: vec4<f32>, at: vec2<f32>) -> vec4<f32> {
    let right = normalize(vec3<f32>(lens[0].x, lens[1].x, lens[2].x));
    let up = normalize(vec3<f32>(lens[0].y, lens[1].y, lens[2].y));
    return vec4<f32>(origin.xyz + (right * at.x - up * at.y) * origin.w, 1.0);
}
@vertex fn vs_label(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Placed {
    let glyph = labels[index];
    var out: Placed;
    if !drawn_shown(glyph.drawn.x) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
    let at = mix(glyph.place.xy, glyph.place.zw, unit);
    out.clip = camera.view_projection * facing(camera.view_projection, glyph.origin, at);
    out.previous = camera.previous_view_projection
        * facing(camera.previous_view_projection, glyph.previous, at);
    out.texel = glyph.rect.xy + unit * glyph.rect.zw;
    out.color = glyph.color * drawn_records[glyph.drawn.x].tint;
    return out;
}
@fragment fn fs_label(placed: Placed) -> ViewTargets {
    let coverage = glyph_coverage(vec2<i32>(placed.texel));
    if placed.color.a * coverage < 0.5 { discard; }
    return view_targets(flat(vec4<f32>(placed.color.rgb, 1.0)), view_pixel(placed.clip, placed.previous));
}
