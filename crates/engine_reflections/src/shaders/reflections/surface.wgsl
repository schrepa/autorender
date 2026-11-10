import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::face::lighting::{SURFACE_THIN, reflecting};
import package::face::trace_depths::{TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN, trace_depth};
import package::generated::pieces::reflections::piece;

fn surface_depth(kind: u32) -> u32 {
    return trace_depth(select(TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN, kind == SURFACE_THIN), kind);
}

fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

fn reflects(point: SurfacePoint) -> bool {
    return point.roughness < piece().surface.x;
}

fn surface(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(point.normal) * 0.5 + 0.5);
    var across = point.roughness;
    if reflects(point) {
        across = reflecting(point).across;
    }
    let roughness = pack4x8unorm(vec4<f32>(point.roughness, across, 0.0, 0.0));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness, surface_depth(point.kind));
}

fn reflected(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    if !reflects(point) {
        return vec4<f32>(0.0);
    }
    let environment = reflecting(point);
    return vec4<f32>(environment.light, environment.share) * point.opacity;
}
