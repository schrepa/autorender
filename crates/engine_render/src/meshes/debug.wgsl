const DEBUG_KIND_TERRAIN: u32 = 6u;
const DEBUG_KIND_COVER: u32 = 7u;
const DEBUG_KIND_WATER: u32 = 8u;
const DEBUG_KIND_SEA: u32 = 9u;
const DEBUG_KIND_FALLS: u32 = 10u;
const DEBUG_KIND_SKY: u32 = 11u;
const DEBUG_LEVEL_SHIFT: u32 = 4u;
const DEBUG_FADING: u32 = 0x100u;
const DEBUG_SOURCE_SHIFT: u32 = 16u;
const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_LEVEL: u32 = 1u;
const DEBUG_SUN_HORIZON: u32 = 21u;
const DEBUG_SUN_BEYOND: u32 = 22u;
const DEBUG_SHADOWED: u32 = 0x80u;
const DEBUG_TILE_EDGE: u32 = 0x800u;
const DEBUG_PATCH_EDGE: u32 = 0x1000u;
const DEBUG_PATCH_SHIFT: u32 = 24u;
const DEBUG_NO_MATERIAL: u32 = 0xffu;
const DEBUG_GRAVITY: f32 = 9.81;
const DEBUG_DRY: f32 = 1.0e-4;
const DEBUG_SPAN_STEPS: f32 = 254.0;
override DEBUG_WATER_VIEWS: bool = false;

struct DebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
}

fn debug_targets(record: u32, object: u32, detail: u32, point: SurfacePoint) -> DebugTargets {
    let sun = debug_sun(point.world + point.shadow_offset, point.geometric, point.world_dx,
        point.world_dy, point.on_terrain);
    var targets: DebugTargets;
    targets.ids = vec4<u32>(record | (sun << DEBUG_SOURCE_SHIFT), object, detail,
        pack2x16snorm(octahedral_coordinates(stable_normalize(point.normal))));
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}

fn debug_point(world: vec3<f32>, normal: vec3<f32>) -> SurfacePoint {
    var point: SurfacePoint;
    point.world = world;
    point.world_dx = dpdx(world);
    point.world_dy = dpdy(world);
    point.geometric = normal;
    point.normal = normal;
    point.distance = length(world - camera.position.xyz);
    return point;
}

fn debug_shadowed(source: u32, visible: f32) -> u32 {
    return source | select(0u, DEBUG_SHADOWED, visible < 0.5);
}

fn debug_sun(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        on_terrain: bool) -> u32 {
    if shadow.state.x == 0.0 {
        return DEBUG_SUN_NONE;
    }
    if terrain_visibility(world, on_terrain) <= 0.0 {
        return DEBUG_SUN_HORIZON | DEBUG_SHADOWED;
    }
    let found = map_lookup(world, normal, dx, dy);
    if found.y < 0.0 {
        return DEBUG_SUN_BEYOND;
    }
    return debug_shadowed(DEBUG_SUN_LEVEL + u32(found.y), found.x);
}

struct DebugOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) @interpolate(flat) object: vec4<u32>,
}

fn debug_output(vertex: VertexOutput, color: vec4<f32>, uv1: vec2<f32>, object: vec4<u32>)
        -> DebugOutput {
    var output: DebugOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.color = color;
    output.uv1 = uv1;
    output.object = object;
    return output;
}

@vertex
fn vs_debug_affine(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> DebugOutput {
    return debug_output(vertex_affine(input), vec4<f32>(1.0), input.uv,
        vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u));
}

@vertex
fn vs_debug_affine_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(14) draw: vec4<u32>) -> DebugOutput {
    return debug_output(vertex_affine(input), color, uv1,
        vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u));
}

@vertex
fn vs_debug_affine_skinned(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(11) joints: vec4<u32>,
        @location(12) weights: vec4<f32>, @location(14) draw: vec4<u32>) -> DebugOutput {
    return debug_output(vertex_affine_skinned(input, joints, weights), color, uv1,
        vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u));
}

@vertex
fn vs_debug_virtual(input: VirtualInput, @location(14) draw: vec4<u32>) -> DebugOutput {
    let vertex = vertex_virtual(input);
    return debug_output(vertex, vec4<f32>(1.0), vec2<f32>(vertex.normal_u.w, vertex.world_v.w),
        vec4<u32>(draw.x, slot_index(input.slot), input.cluster, 0u));
}

@fragment
fn fs_debug_mesh(input: DebugOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    var extended: ExtendedOutput;
    extended.clip_position = input.clip_position;
    extended.normal_u = input.normal_u;
    extended.world_v = input.world_v;
    extended.material = input.material;
    extended.color = input.color;
    extended.uv1 = input.uv1;
    let values = extended_fragment(extended);
    let point = surface_point(values.color, values.normal, values.world, values.uv, values.uv1,
        values.surface, front);
    return debug_targets(input.object.x, input.object.y, input.object.z, point);
}

struct DebugImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    @location(9) @interpolate(flat) object: vec2<u32>,
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
}

@vertex
fn vs_debug_impostor(input: ImpostorInput, @location(14) draw: vec4<u32>)
        -> DebugImpostorOutput {
    let card = impostor_card(input).output;
    var output: DebugImpostorOutput;
    output.clip_position = card.clip_position;
    output.frames_ab = card.frames_ab;
    output.frame_c = card.frame_c;
    output.world = card.world;
    output.normal_x = card.normal_x;
    output.normal_y = card.normal_y;
    output.normal_z = card.normal_z;
    output.depth_axis = card.depth_axis;
    output.material = card.material;
    output.batch = card.batch;
    output.object = vec2<u32>(draw.x, input.slot & ~MOVED_SLOT);
    output.crossings = card.crossings;
    output.slopes_ab = card.slopes_ab;
    output.slope_c = card.slope_c;
    return output;
}

struct DebugImpostorTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_debug_impostor(input: DebugImpostorOutput) -> DebugImpostorTargets {
    var card: ImpostorOutput;
    card.clip_position = input.clip_position;
    card.frames_ab = input.frames_ab;
    card.frame_c = input.frame_c;
    card.world = input.world;
    card.normal_x = input.normal_x;
    card.normal_y = input.normal_y;
    card.normal_z = input.normal_z;
    card.depth_axis = input.depth_axis;
    card.material = input.material;
    card.batch = input.batch;
    card.crossings = input.crossings;
    card.slopes_ab = input.slopes_ab;
    card.slope_c = input.slope_c;
    let point = impostor_point(card);
    let shaded = impostor_shaded(card, point, vec3<f32>(0.0));
    let fading = select(0u, DEBUG_FADING, (input.batch & IMPOSTOR_FADING) != 0u);
    let targets = debug_targets(input.object.x | fading, input.object.y,
        impostor_index(input.batch), point);
    return DebugImpostorTargets(targets.ids, targets.count, targets.terrain, shaded.depth);
}

@fragment
fn fs_debug_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    return debug_targets(DEBUG_KIND_COVER, 0u, 0u, cover_surface(input, front));
}

fn debug_span_code(span: f32) -> f32 {
    if span <= 0.0 {
        return 0.0;
    }
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = clamp(round(log2(span / WATER_SPAN_LEAST) / octaves * DEBUG_SPAN_STEPS), 0.0,
        DEBUG_SPAN_STEPS);
    return (steps + 1.0) / (DEBUG_SPAN_STEPS + 1.0);
}

fn debug_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt) -> DebugTargets {
    let shade = shade_water(point, pixel, at, false);
    if shade.coverage < 0.5 && !DEBUG_WATER_VIEWS { // optics: water covers as its shading sees it
        discard;
    }
    let seen = debug_point(eye_world(point), shade.shaded.normal);
    let depth = at.depth;
    let speed = length(at.words.flow);
    let froude = speed / sqrt(DEBUG_GRAVITY * max(depth, DEBUG_DRY));
    let eddies = water_eddy_froude(speed, length(at.slope), depth);
    var targets = debug_targets(DEBUG_KIND_WATER, pack2x16float(vec2<f32>(depth, speed)),
        pack4x8unorm(vec4<f32>(debug_span_code(at.words.span), shade.shaded.roughness,
            0.0, 0.0)), seen);
    targets.terrain = vec2<u32>(pack2x16float(vec2<f32>(froude, eddies)),
        pack4x8unorm(vec4<f32>(at.words.state.xy, 0.0, 0.0))
            | (min(at.words.reach, 0xffu) << 24u));
    return targets;
}

@fragment
fn fs_debug_water(input: WaterOutput) -> DebugTargets {
    let at = water_drawn(input.relative);
    return debug_water(water_on_surface(input.relative, at), input.clip_position.xy, at);
}

@fragment
fn fs_debug_ocean(input: OceanOutput) -> DebugTargets {
    let shade = shade_ocean(input, false);
    if shade.coverage < 0.5 { // optics: the sea covers as its shading sees it
        discard;
    }
    let point = debug_point(eye_world(input.relative), shade.shaded.normal);
    return debug_targets(DEBUG_KIND_SEA, 0u, 0u, point);
}

@fragment
fn fs_debug_fall(input: FallOutput) -> DebugTargets {
    let shade = shade_fall(input);
    if shade.shown < 0.5 { // optics: falls show as their shading sees them
        discard;
    }
    let point = debug_point(eye_world(input.relative), shade.shaded.normal);
    return debug_targets(DEBUG_KIND_FALLS, 0u, 0u, point);
}

@fragment
fn fs_debug_sky(input: SkyOutput) -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(DEBUG_KIND_SKY, 0u, 0u, 0u);
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}
