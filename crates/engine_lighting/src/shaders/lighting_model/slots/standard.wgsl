import package::engine::materials::{VALUE_TRANSMISSION, material_value};
import package::engine::shading::{SurfaceInput, SurfacePoint};

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.x;
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.y;
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return 1.0;
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return material_value(VALUE_TRANSMISSION);
}
