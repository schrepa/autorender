import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::view_clip;
import package::figures::light::{FigureCorner, figure_field, octahedral_normal};

struct StaticVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(3) uv0: vec2<f32>,
}

@vertex fn vs_ground(vertex: StaticVertex, @builtin(instance_index) index: u32) -> FigureCorner {
    let slot = drawn_slot(index);
    let model = drawn_model(slot);
    let world = (model * vec4<f32>(vertex.position, 1.0)).xyz;
    var out: FigureCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(octahedral_normal(vertex.normal), 0.0)).xyz;
    out.uv = vertex.uv0;
    out.slot = slot;
    out.field = figure_field(world, slot);
    return out;
}
