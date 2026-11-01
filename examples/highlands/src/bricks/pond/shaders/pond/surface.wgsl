import package::engine::shading::{BODY_WATER, Shaded, SurfacePoint};
import package::engine::view::{camera, stable_normalize, view_pixel};
import package::face::ground_tiles::{GroundTerrain, ground_height, terrain_info};
import package::face::lighting::finish;
import package::generated::pond::{POND_MOST, PondShape};
import package::generated::targets::{ViewTargets, view_targets};

@group(1) @binding(0) var<uniform> ponds: array<PondShape, POND_MOST>;

const POND_BODY: vec3<f32> = vec3<f32>(0.02, 0.05, 0.06);
const POND_ROUGHNESS: f32 = 0.06;
const POND_F0: f32 = 0.02;
const POND_CLEAR: f32 = 0.8;
const POND_DEEP: f32 = 4.0;

struct PondOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) pond: u32,
}

@vertex
fn vs_pond(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> PondOutput {
    let shape = ponds[instance];
    let corner = vec2<f32>(f32((50u >> vertex) & 1u), f32((44u >> vertex) & 1u));
    let place = mix(shape.area.xy, shape.area.zw, corner);
    let world = vec3<f32>(place.x, shape.surface.x, place.y);
    var output: PondOutput;
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    output.world = world;
    output.pond = instance;
    return output;
}

fn pond_ground(shape: PondShape, world: vec3<f32>) -> f32 {
    var directory = 0u;
    if shape.ground.z != 0u {
        directory = terrain_info(shape.ground.x).levels.x;
    }
    let terrain = GroundTerrain(directory, shape.ground.y, shape.terrain.w);
    let local = world.xz - shape.terrain.xz;
    let ground = ground_height(terrain, local);
    if ground.found {
        return shape.terrain.y + ground.height;
    }
    return shape.surface.x - POND_DEEP;
}

struct PondShade {
    shaded: Shaded,
    cover: f32,
}

fn shade_pond(input: PondOutput) -> PondShade {
    let world = input.world;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let shape = ponds[input.pond];
    let depth = shape.surface.x - pond_ground(shape, world);
    if depth <= 0.0 {
        discard;
    }
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let toward = camera.position.xyz - world;
    let view = stable_normalize(toward);
    let n_v = max(dot(up, view), 1.0e-4);
    var point: SurfacePoint;
    point.albedo = POND_BODY;
    point.opacity = 1.0;
    point.occlusion = 1.0;
    point.roughness = POND_ROUGHNESS;
    point.alpha = POND_ROUGHNESS * POND_ROUGHNESS;
    point.f0 = vec3<f32>(POND_F0);
    point.diffuse_color = POND_BODY;
    point.normal = up;
    point.geometric = up;
    point.view = view;
    point.n_v = n_v;
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.distance = length(toward);
    var shade: PondShade;
    shade.shaded = finish(point, vec3<f32>(0.0));
    shade.shaded.body = BODY_WATER;
    shade.cover = 1.0 - exp(-depth / (POND_CLEAR * n_v));
    shade.shaded.reflect *= shade.cover;
    shade.shaded.mirrored *= shade.cover;
    return shade;
}

@fragment
fn fs_pond(input: PondOutput) -> ViewTargets {
    let shade = shade_pond(input);
    let previous = (camera.previous_view_projection * vec4<f32>(input.world, 1.0)).xyw;
    var targets = view_targets(shade.shaded,
        view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
    targets.color = vec4<f32>(shade.shaded.linear * shade.cover, shade.cover);
    return targets;
}
