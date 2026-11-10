import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::slots::{derived, set_metalness, set_occlusion, set_roughness,
    set_transmission, shade as shade_point, standard_point};
import package::engine::surfaces::SURFACE_POND;
import package::engine::view::{camera, stable_normalize, view_pixel};
import package::face::debug::{DebugTargets, debug_record, debug_targets};
import package::face::ground_tiles::{GroundTerrain, ground_height, terrain_info};
import package::face::lighting::given_reflectance;
import package::generated::kinds::KIND_POND;
import package::generated::pond::{POND_MOST, PondShape};
import package::generated::writes::{Outputs, outputs};

@group(1) @binding(0) var<uniform> ponds: array<PondShape, POND_MOST>;
@group(1) @binding(1) var pond_scene: texture_2d<f32>;

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
    point: SurfacePoint,
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
    var point = standard_point();
    point.base_color = POND_BODY;
    point.kind = SURFACE_POND;
    point.normal = up;
    point.geometric = up;
    point.view = view;
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.distance = length(toward);
    point = set_roughness(point, POND_ROUGHNESS);
    point = set_metalness(point, 0.0);
    point = set_occlusion(point, 1.0);
    point = set_transmission(point, 0.0);
    point = given_reflectance(derived(point), vec3<f32>(POND_F0), POND_BODY);
    var shade: PondShade;
    shade.shaded = shade_point(point, input.clip_position.xy);
    shade.cover = 1.0 - exp(-depth / (POND_CLEAR * n_v));
    shade.point = point;
    shade.point.opacity = shade.cover;
    return shade;
}

@fragment
fn fs_pond(input: PondOutput) -> Outputs {
    let shade = shade_pond(input);
    let previous = (camera.previous_view_projection * vec4<f32>(input.world, 1.0)).xyw;
    var targets = outputs(shade.point, shade.shaded,
        view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
    let last = vec2<i32>(textureDimensions(pond_scene)) - 1;
    let seen = textureLoad(pond_scene, min(vec2<i32>(input.clip_position.xy), last), 0).rgb;
    targets.color = vec4<f32>(shade.shaded.linear * shade.cover + seen * (1.0 - shade.cover), 1.0);
    return targets;
}

@fragment
fn fs_debug_pond(input: PondOutput) -> DebugTargets {
    let shape = ponds[input.pond];
    if shape.surface.x - pond_ground(shape, input.world) <= 0.0 {
        discard;
    }
    return debug_targets(debug_record(KIND_POND, 0u), input.pond, 0u, vec3<f32>(0.0, 1.0, 0.0),
        0u);
}
