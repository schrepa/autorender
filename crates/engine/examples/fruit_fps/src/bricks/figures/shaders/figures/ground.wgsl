import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::view_clip;
import package::figures::light::{FigureCorner, figure_field};
import package::generated::figures::{MeshBase, mesh_normal};

@vertex fn vs_ground(base: MeshBase, @builtin(instance_index) index: u32) -> FigureCorner {
    let slot = drawn_slot(index);
    let model = drawn_model(slot);
    let world = (model * vec4<f32>(base.position, 1.0)).xyz;
    var out: FigureCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(mesh_normal(base.normal), 0.0)).xyz;
    out.uv = base.uv0;
    out.slot = slot;
    out.field = figure_field(world, slot);
    return out;
}
