import package::generated::text::GlyphInstance;
@group(1) @binding(0) var<storage, read> instances: array<GlyphInstance>;
@group(1) @binding(1) var atlas: texture_2d<f32>;
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
    let coverage = textureLoad(atlas, vec2<i32>(corner.texel), 0).r;
    return vec4<f32>(corner.color.rgb, corner.color.a * coverage);
}
