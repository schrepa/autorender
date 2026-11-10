import package::engine::shading::{Shaded, SurfacePoint, unlit};

const GLIMMER_BANDS: f32 = 4.0;

fn lit(point: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    let band = ceil(saturate(point.n_v) * GLIMMER_BANDS) / GLIMMER_BANDS;
    var shaded = unlit(point);
    shaded.linear = point.base_color * band + point.emissive + point.inputs.added_light;
    return shaded;
}
