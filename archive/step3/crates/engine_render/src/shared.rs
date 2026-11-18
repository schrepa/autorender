pub(crate) const PASSES_MODULE: &str = "generated::passes";
pub(crate) const KINDS_MODULE: &str = "generated::kinds";
const SHADED: &str = "
fn point_shaded(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    return shade(point, pixel);
}
";
