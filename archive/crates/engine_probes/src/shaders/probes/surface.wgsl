import package::engine::color::encode_srgb;
import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::face::lighting::{sun_visibility, surface_sun};

fn folded(direction: vec3<f32>) -> vec2<f32> {
    let across = direction.xy
        / max(abs(direction.x) + abs(direction.y) + abs(direction.z), 1.0e-20);
    if direction.z >= 0.0 {
        return across;
    }
    let sides = select(vec2<f32>(-1.0), vec2<f32>(1.0), across >= vec2<f32>(0.0));
    return (1.0 - abs(across.yx)) * sides;
}

fn unfolded(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = select(vec2<f32>(-fold), vec2<f32>(fold), encoded >= vec2<f32>(0.0));
    return normalize(vec3<f32>(encoded - shift, z));
}

fn codes(values: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(round(saturate(values) * 255.0));
}

fn probe_base(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<u32> {
    return codes(vec4<f32>(encode_srgb(saturate(point.base_color)), point.occlusion));
}

fn probe_surface(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<u32> {
    let faced = point.thickness == 0.0 && !point.ignore_face;
    let shape = codes(vec4<f32>(point.roughness, point.metalness, 0.0, 0.0));
    return vec4<u32>(shape.xy, point.kind, select(0u, 1u, faced));
}

fn probe_normal(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    let skyward = normalize(point.normal + point.ambient_bend);
    return vec4<f32>(folded(point.normal), folded(skyward));
}

fn probe_emissive(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    return vec4<f32>(point.emissive, point.transmission);
}

fn probe_bounce(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    let open = 1.0 - point.direct_blocked;
    var visible = open;
    if open != 0.0 && !point.direct_known {
        visible *= sun_visibility(surface_sun(point));
    }
    return vec4<f32>(point.added_indirect, visible);
}
