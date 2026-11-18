import package::engine::drawn::{drawn_emission, drawn_floats, drawn_model, drawn_slot};
import package::engine::view::{camera, stable_normalize, view_clip};
import package::figures::figure::{SkinnedVertex, blended};
import package::figures::light::octahedral_normal;
import package::figures::material::material_texels;

struct RevealCorner {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) glow: vec3<f32>,
}

@vertex fn vs_revealed(vertex: SkinnedVertex, @builtin(instance_index) index: u32)
        -> RevealCorner {
    let slot = drawn_slot(index);
    let glow = drawn_emission(slot);
    var out: RevealCorner;
    if dot(glow, glow) == 0.0 {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let skin = blended(u32(drawn_floats(slot, 0u).x), vertex.joints, vertex.weights);
    let rest = vec4<f32>(vertex.position, 1.0);
    let posed = vec3<f32>(dot(skin.row_0, rest), dot(skin.row_1, rest), dot(skin.row_2, rest));
    let normal = octahedral_normal(vertex.normal);
    let turned = vec3<f32>(dot(skin.row_0.xyz, normal), dot(skin.row_1.xyz, normal),
        dot(skin.row_2.xyz, normal));
    let model = drawn_model(slot);
    let world = (model * vec4<f32>(posed, 1.0)).xyz;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(turned, 0.0)).xyz;
    out.uv = vertex.uv0;
    out.glow = glow;
    return out;
}

@fragment fn fs_revealed(corner: RevealCorner) -> @location(0) vec4<f32> {
    let texels = material_texels(corner.uv);
    let normal = stable_normalize(corner.normal);
    let view = stable_normalize(camera.position.xyz - corner.world);
    let edge = 1.0 - abs(dot(normal, view));
    let outline = edge * edge * edge;
    let fill = 0.12 + 0.05 * dot(texels.base_color, vec3<f32>(0.333));
    return vec4<f32>(corner.glow * (outline + fill), 0.0);
}
