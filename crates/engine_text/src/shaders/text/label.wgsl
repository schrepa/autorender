import package::engine::drawn::{drawn_records, drawn_shown};
import package::engine::view::{camera, stable_normalize, view_clip};
import package::generated::text::LabelGlyph;

@group(2) @binding(0) var<storage, read> labels: array<LabelGlyph>;

const LABEL_WHITE: u32 = 0xffffffffu;
const LABEL_ROUGH: u32 = 0xffu;

struct LabelCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct ShownLabelCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn facing(lens: mat4x4<f32>, origin: vec4<f32>, at: vec2<f32>) -> vec3<f32> {
    let right = normalize(vec3<f32>(lens[0].x, lens[1].x, lens[2].x));
    let up = normalize(vec3<f32>(lens[0].y, lens[1].y, lens[2].y));
    return origin.xyz + (right * at.x - up * at.y) * origin.w;
}

fn label_corner(corner: u32, glyph: LabelGlyph) -> LabelCorner {
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
    let at = mix(glyph.place.xy, glyph.place.zw, unit);
    let world = facing(camera.view_projection, glyph.origin, at);
    let was = facing(camera.previous_view_projection, glyph.previous, at);
    let shown = camera.previous_view_projection * vec4<f32>(was, 1.0);
    let texel = glyph.rect.xy + unit * glyph.rect.zw;
    var out: LabelCorner;
    out.clip_position = view_clip(world);
    out.normal_u = vec4<f32>(stable_normalize(camera.position.xyz - world), texel.x);
    out.world_v = vec4<f32>(world, texel.y);
    out.material = vec4<u32>(LABEL_WHITE, LABEL_WHITE, LABEL_ROUGH, 0u);
    out.previous = shown.xyw;
    out.color = glyph.color * drawn_records[glyph.drawn.x].tint;
    out.uv1 = texel;
    return out;
}

@vertex fn vs_label(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> ShownLabelCorner {
    let glyph = labels[index];
    var out: ShownLabelCorner;
    if !drawn_shown(glyph.drawn.x) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let placed = label_corner(corner, glyph);
    out.clip_position = placed.clip_position;
    out.normal_u = placed.normal_u;
    out.world_v = placed.world_v;
    out.material = placed.material;
    out.previous = placed.previous;
    out.color = placed.color;
    out.uv1 = placed.uv1;
    out.object = vec4<u32>(0u, glyph.drawn.x, 0u, 0u);
    return out;
}

@vertex fn vs_label_view(@builtin(vertex_index) corner: u32,
        @builtin(instance_index) index: u32) -> LabelCorner {
    return label_corner(corner, labels[index]);
}
