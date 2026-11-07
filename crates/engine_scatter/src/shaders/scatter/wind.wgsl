import package::engine::shading::PositionInput;
import package::face::view_wind::wind_sway;
import package::generated::scatter::WIND_REACH;
import package::scatter::values::placed_values_of;

fn position(input: PositionInput) -> vec3<f32> {
    let sway = placed_values_of(input.kind, input.object).sway;
    if !(sway > 0.0) {
        return input.world;
    }
    let offset = wind_sway(input.time, input.pixels, input.origin, input.world.y - input.origin.y,
        input.local, sway);
    return input.world + offset * min(1.0, WIND_REACH / max(length(offset), 1.0e-6));
}
