import package::engine::shading::{Shaded, ViewPixel};
import package::engine::surfaces::SURFACE_THIN;
import package::face::trace_depths::{TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN, trace_depth};

const MIRRORED_BITS: u32 = 14u;

fn surface_depth(kind: u32) -> u32 {
    return trace_depth(select(TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN, kind == SURFACE_THIN), kind);
}

fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

fn surface(shaded: Shaded, at: ViewPixel) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(shaded.normal) * 0.5 + 0.5);
    let roughness = pack4x8unorm(vec4<f32>(shaded.roughness, shaded.outputs.across, 0.0, 0.0));
    let most = f32((1u << MIRRORED_BITS) - 1u);
    let mirrored = u32(floor(saturate(shaded.outputs.mirrored) * most + 0.5));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness,
        mirrored | (surface_depth(shaded.kind) << MIRRORED_BITS));
}

fn reflectivity(shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.outputs.reflect;
}
