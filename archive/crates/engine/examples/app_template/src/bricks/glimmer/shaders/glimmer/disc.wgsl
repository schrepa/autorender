import package::engine::shading::{SurfaceInput, SurfacePoint};

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let unit = input.uv * 2.0 - 1.0;
    return select(1.0, 0.0, dot(unit, unit) > 1.0);
}
