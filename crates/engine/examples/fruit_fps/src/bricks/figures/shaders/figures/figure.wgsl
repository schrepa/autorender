import package::engine::drawn::{drawn_floats, drawn_model, drawn_slot};
import package::engine::view::view_clip;
import package::figures::light::{FigureCorner, octahedral_normal};

struct JointRow {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

@group(2) @binding(1) var<storage, read> figure_palette: array<JointRow>;

struct SkinnedVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(9) joints: vec4<u32>,
    @location(10) weights: vec4<f32>,
}

fn blended(base: u32, joints: vec4<u32>, weights: vec4<f32>) -> JointRow {
    let a = figure_palette[base + joints.x];
    let b = figure_palette[base + joints.y];
    let c = figure_palette[base + joints.z];
    let d = figure_palette[base + joints.w];
    return JointRow(
        a.row_0 * weights.x + b.row_0 * weights.y + c.row_0 * weights.z + d.row_0 * weights.w,
        a.row_1 * weights.x + b.row_1 * weights.y + c.row_1 * weights.z + d.row_1 * weights.w,
        a.row_2 * weights.x + b.row_2 * weights.y + c.row_2 * weights.z + d.row_2 * weights.w);
}

@vertex fn vs_figure(vertex: SkinnedVertex, @builtin(instance_index) index: u32) -> FigureCorner {
    let slot = drawn_slot(index);
    let skin = blended(u32(drawn_floats(slot, 0u).x), vertex.joints, vertex.weights);
    let rest = vec4<f32>(vertex.position, 1.0);
    let posed = vec3<f32>(dot(skin.row_0, rest), dot(skin.row_1, rest), dot(skin.row_2, rest));
    let normal = octahedral_normal(vertex.normal);
    let turned = vec3<f32>(dot(skin.row_0.xyz, normal), dot(skin.row_1.xyz, normal),
        dot(skin.row_2.xyz, normal));
    let model = drawn_model(slot);
    let world = (model * vec4<f32>(posed, 1.0)).xyz;
    var out: FigureCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(turned, 0.0)).xyz;
    out.uv = vertex.uv0;
    out.slot = slot;
    return out;
}
