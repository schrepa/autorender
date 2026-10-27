// Debug views (`engine_core::DebugView`): pipelines of their own drawing,
// in place of each pixel's shading, what drew it into the debug targets,
// which `debug_view.wgsl` shows. A fragment writes its draw's record (its
// kind in the low four bits, its level in the next four, then its flags)
// with the sun's lookup at its point above DEBUG_SOURCE_SHIFT, the object
// drawn, a detail of it (a cluster, an impostor's record) and its normal;
// adds one to the fragments that passed the depth test there; and writes
// terrain's and water's numbers into the third target, zeros elsewhere
// (`meshes/terrain_debug.wgsl`, `debug_water`). Meshes,
// impostors and virtual geometry read their draw's record from a vertex
// stream of one element (`@location(14)`); the others know their kind.

// The kinds whose fragments know them; the others' come from their records.
const DEBUG_KIND_TERRAIN: u32 = 6u;
const DEBUG_KIND_COVER: u32 = 7u;
const DEBUG_KIND_WATER: u32 = 8u;
const DEBUG_KIND_SEA: u32 = 9u;
const DEBUG_KIND_FALLS: u32 = 10u;
const DEBUG_KIND_SKY: u32 = 11u;
const DEBUG_LEVEL_SHIFT: u32 = 4u;
// A card fading in or out over its mesh.
const DEBUG_FADING: u32 = 0x100u;
// The sun's lookup: none (no sun shadows), the clipmap's levels from the
// finest, a terrain's horizon, no level drawn there; DEBUG_SHADOWED where
// it leaves the point in shadow.
const DEBUG_SOURCE_SHIFT: u32 = 16u;
const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_LEVEL: u32 = 1u;
const DEBUG_SUN_HORIZON: u32 = 21u;
const DEBUG_SUN_BEYOND: u32 = 22u;
const DEBUG_SHADOWED: u32 = 0x80u;
// Terrain pixels: on a tile's edge, on a patch's, in their record, with the
// patch within its tile above DEBUG_PATCH_SHIFT; a material slot holding
// none.
const DEBUG_TILE_EDGE: u32 = 0x800u;
const DEBUG_PATCH_EDGE: u32 = 0x1000u;
const DEBUG_PATCH_SHIFT: u32 = 24u;
const DEBUG_NO_MATERIAL: u32 = 0xffu;
// Gravity, and the meters of water under which it lies dry, by which water's
// Froude number is reckoned (`engine_terrain`'s hydraulics and
// shallow-water scheme).
const DEBUG_GRAVITY: f32 = 9.81;
const DEBUG_DRY: f32 = 1.0e-4;
// Steps of a water pixel's span on its logarithmic scale, a byte's codes
// past none (0), from WATER_SPAN_LEAST (1) to WATER_SPAN_MOST (255)
// (`gpu_layouts::constants::DEBUG_SPAN_STEPS`).
const DEBUG_SPAN_STEPS: f32 = 254.0;
// Whether the frame shows one of the water's views of its state
// (DEBUG_VIEW_WATER_* of `debug_view.wgsl`), which draw tiles' and
// simulated water wherever it lies; the other views draw it where it covers
// at least half the pixel.
override DEBUG_WATER_VIEWS: bool = false;

struct DebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
}

// Debug fragments shade nothing; they write what the shaded pipeline would
// show of `point`, drawn by `record`'s draw, with `object` and `detail`.
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

// A point of `world` with `normal` for kinds without a surface point:
// where it lies, how it moves per pixel and how far from the eye.
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

// `source` with DEBUG_SHADOWED where the lookup leaves under half the sun.
fn debug_shadowed(source: u32, visible: f32) -> u32 {
    return source | select(0u, DEBUG_SHADOWED, visible < 0.5);
}

// Which lookup decides the sun at the surface point `world` (a DEBUG_SUN_
// value), in `sun_visibility`'s order, with DEBUG_SHADOWED where it leaves
// the point in shadow: a terrain's horizon, then the clipmap's level that
// `map_lookup` reads first, or none where no level holds a drawn tile
// there.
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

// Meshes' and virtual geometry's vertex outputs with the second stream's
// (white and the first coordinates for meshes without one), and the
// object: the draw's record, the slot and the cluster. Positions are
// invariant, as `VertexOutput`'s, and masked meshes' depth pre-pass reads
// these outputs as `ExtendedOutput`'s.
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

// What the shaded pipelines' fragment leaves of the material's point: its
// fades and masks discard as theirs do.
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

// Impostor cards' outputs and the object: the draw's record and the slot.
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

// A card fragment at the depth its shaded pipeline writes, fading cards
// flagged.
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

// The byte of a span of `span` meters across a water body, as a share of
// its top: none (0) where none is known, else its nearest step on a
// logarithmic scale from WATER_SPAN_LEAST (1) to WATER_SPAN_MOST
// (DEBUG_SPAN_STEPS more), each about 4% wider than the one before.
fn debug_span_code(span: f32) -> f32 {
    if span <= 0.0 {
        return 0.0;
    }
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = clamp(round(log2(span / WATER_SPAN_LEAST) / octaves * DEBUG_SPAN_STEPS), 0.0,
        DEBUG_SPAN_STEPS);
    return (steps + 1.0) / (DEBUG_SPAN_STEPS + 1.0);
}

// Water where its shading covers at least half the pixel, or in the
// water's views wherever it lies, at the pixel `pixel` as `at` holds it,
// drawn at the point `mesh` of its mesh over the point `point` of its
// surface: the tiles', which gives way where water is simulated, or the
// simulated water's (`simulated`). Its object word holds its depth (meters)
// and its speed (meters a second), half floats; its detail word four bytes:
// how far across its water body is (`debug_span_code`), the roughness its
// shading gives it, and none; the third target its Froude number and
// its eddies' turbulent Froude number (`water_eddy_froude`), half floats,
// then its turbulence, white water and the foam it carries (a byte each, 0
// to 1, the foam a share of WATER_FOAM_MOST) with its reach in the top byte
// (0 for none, else one more than `engine_terrain::Reach`).
fn debug_water(mesh: vec3<f32>, point: vec3<f32>, pixel: vec2<f32>, at: WaterAt,
        simulated: bool) -> DebugTargets {
    let shade = shade_water(mesh, point, pixel, at, false, simulated);
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
        pack4x8unorm(vec4<f32>(at.words.state.xy, at.words.foam / WATER_FOAM_MOST, 0.0))
            | (min(at.words.reach, 0xffu) << 24u));
    return targets;
}

@fragment
fn fs_debug_water(input: WaterOutput) -> DebugTargets {
    let at = water_drawn(input.relative);
    return debug_water(input.relative, water_on_surface(input.relative, at),
        input.clip_position.xy, at, false);
}

@fragment
fn fs_debug_sim_water(input: SimWaterOutput) -> DebugTargets {
    return debug_water(input.relative, input.relative, input.clip_position.xy,
        sim_water_values(input), true);
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

// The sky over the pixels nothing covers: no sun lookup, no normal.
@fragment
fn fs_debug_sky(input: SkyOutput) -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(DEBUG_KIND_SKY, 0u, 0u, 0u);
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}
