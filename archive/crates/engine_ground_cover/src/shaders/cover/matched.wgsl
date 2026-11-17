import package::cover::draw::CoverVaryings;
import package::engine::kinds::{drawn_pixel, extra_values, surface_input};
import package::engine::slots::{shade, slot_output};
import package::engine::surface::surface_point;
import package::generated::writes::{Outputs, outputs};

@fragment
fn fs_cover_matched(input: CoverVaryings, @builtin(front_facing) front: bool) -> Outputs {
    var values = extra_values(input.clip_position, input.normal_u, input.world_v, input.material,
        input.color, input.uv1);
    values.kind = input.object.w;
    values.object = input.object.y;
    let surface = surface_input(values, front);
    let point = surface_point(surface);
    return outputs(point, slot_output(surface, point, shade(point, values.pixel)),
        drawn_pixel(input.clip_position, input.previous));
}
