import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::face::lighting::{Reflecting, SURFACE_THIN, reflecting};
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

fn traced(point: SurfacePoint, across: f32) -> f32 {
    let limit = piece().surface;
    let fine = saturate((1.0 - max(point.roughness, across) / limit.x) / limit.y);
    return point.mirrored * max(fine, point.mirrored_rough);
}

fn reflects(point: SurfacePoint) -> bool {
    if point.kind == SURFACE_THIN {
        return false;
    }
    let kept = traced(point, point.roughness - point.smoother_across);
    return pack4x8unorm(vec4<f32>(kept, 0.0, 0.0, 0.0)) != 0u;
}

struct Asked {
    reflects: bool,
    reflecting: Reflecting,
}

fn asked(point: SurfacePoint) -> Asked {
    var answer: Asked;
    answer.reflects = reflects(point);
    if answer.reflects {
        answer.reflecting = reflecting(point);
    }
    return answer;
}

fn surface(point: SurfacePoint, shaded: Shaded, at: ViewPixel, asked: Asked) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(point.normal) * 0.5 + 0.5);
    let rough = point.roughness;
    var across = rough;
    var kept = 0.0;
    if asked.reflects {
        across = asked.reflecting.across;
        kept = traced(point, across);
    }
    let roughness = pack4x8unorm(vec4<f32>(rough, across, 0.0, 0.0));
    let body = surface_depth(point.kind) | (pack4x8unorm(vec4<f32>(kept, 0.0, 0.0, 0.0)) << 8u);
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness, body);
}

fn reflected(point: SurfacePoint, shaded: Shaded, at: ViewPixel, asked: Asked) -> vec4<f32> {
    if !asked.reflects {
        return vec4<f32>(0.0);
    }
    let environment = asked.reflecting;
    return vec4<f32>(environment.light, environment.share) * point.opacity;
}
