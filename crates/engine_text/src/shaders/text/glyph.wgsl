import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::face::text::glyph_coverage;

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.color.a * glyph_coverage(vec2<i32>(input.uv));
}
