import package::surfaces::materials::{VALUE_GLIMMER, emissive_factor, material_value};
import package::surfaces::shading::{SurfaceInput, SurfacePoint};
import package::surfaces::mapped::emissive_texel;

fn emissive(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return point.base_color * material_value(VALUE_GLIMMER)
        + emissive_factor() * emissive_texel(input);
}
