import package::face::dither::stored;
import package::face::lighting::{MIRRORED_BITS, Shaded, SurfacePoint};
import package::frame::view::camera;
import package::meshes::declarations::{SceneTargets, TemporalTargets};
import package::meshes::surface::{SPECULAR_LIMIT, SPECULAR_VARIANCE, octahedral_coordinates};

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    let widening = min(2.0 * SPECULAR_VARIANCE * (dot(turn_dx, turn_dx) + dot(turn_dy, turn_dy)),
        SPECULAR_LIMIT);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

fn surface_texel(shaded: Shaded) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(shaded.normal) * 0.5 + 0.5);
    let roughness = pack4x8unorm(vec4<f32>(shaded.roughness, shaded.across, 0.0, 0.0));
    let most = f32((1u << MIRRORED_BITS) - 1u);
    let mirrored = u32(floor(saturate(shaded.mirrored) * most + 0.5));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness,
        mirrored | (shaded.body << MIRRORED_BITS));
}

fn scene_targets(shaded: Shaded, pixel: vec2<f32>) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(stored(shaded.color, pixel), shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(stored(shaded.color, clip_position.xy), shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    let current = vec2<f32>(clip_position.x * camera.jitter.z - 1.0,
        1.0 - clip_position.y * camera.jitter.w) - camera.jitter.xy;
    targets.motion = (previous.xy / previous.z - current) * vec2<f32>(0.5, -0.5);
    if !(previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
