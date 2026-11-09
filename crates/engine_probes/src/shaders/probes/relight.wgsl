import package::engine::color::decode_srgb;
import package::engine::shading::{SurfacePoint, derived};
import package::engine::view::{camera, stable_normalize};
import package::face::lit::finish;
import package::face::sky::environment_sample;
import package::probes::filter::FaceOutput;
import package::probes::surface::unfolded;

@group(1) @binding(0) var stored_base: texture_2d_array<u32>;
@group(1) @binding(1) var stored_surface: texture_2d_array<u32>;
@group(1) @binding(2) var stored_normal: texture_2d_array<f32>;
@group(1) @binding(3) var stored_emissive: texture_2d_array<f32>;
@group(1) @binding(4) var stored_bounce: texture_2d_array<f32>;
@group(1) @binding(5) var stored_depth: texture_depth_2d_array;

fn stored_point(texel: vec2<i32>, layer: i32, world: vec3<f32>) -> SurfacePoint {
    let base = textureLoad(stored_base, texel, layer, 0);
    let surface = textureLoad(stored_surface, texel, layer, 0);
    let directions = textureLoad(stored_normal, texel, layer, 0);
    let emissive = textureLoad(stored_emissive, texel, layer, 0);
    let bounce = textureLoad(stored_bounce, texel, layer, 0);
    let normal = unfolded(directions.xy);
    let faced = surface.w != 0u;
    let to_eye = camera.position.xyz - world;
    var point: SurfacePoint;
    point.base_color = decode_srgb(vec3<f32>(base.rgb) / 255.0);
    point.opacity = 1.0;
    point.emissive = emissive.rgb;
    point.occlusion = f32(base.a) / 255.0;
    point.roughness = f32(surface.x) / 255.0;
    point.metalness = f32(surface.y) / 255.0;
    point.transmission = emissive.a;
    point.normal = normal;
    point.normal_dx = dpdx(normal);
    point.normal_dy = dpdy(normal);
    point.geometric = normal;
    point.view = stable_normalize(to_eye);
    point.distance = length(to_eye);
    point.world = world;
    point.world_dx = select(vec3<f32>(0.0), dpdx(world), faced);
    point.world_dy = select(vec3<f32>(0.0), dpdy(world), faced);
    point.kind = surface.z;
    point.inputs.ignore_face = !faced;
    point.inputs.bounce = bounce.rgb;
    point.inputs.sky_bend = unfolded(directions.zw) - normal;
    point.inputs.sun_blocked = 1.0 - bounce.a;
    point.inputs.sun_known = true;
    return derived(point);
}

@fragment
fn fs_relight(input: FaceOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let layer = i32(input.item);
    let depth = textureLoad(stored_depth, texel, layer, 0);
    let placed = camera.inverse_view_projection * vec4<f32>(input.ndc, depth, 1.0);
    let world = placed.xyz / placed.w;
    let pixel = input.clip_position.xy;
    let lit = finish(stored_point(texel, layer, world), pixel, vec3<f32>(0.0)).linear;
    let sky = environment_sample(normalize(world - camera.position.xyz), 0.0);
    return vec4<f32>(select(lit, sky, depth <= 0.0), 1.0);
}
