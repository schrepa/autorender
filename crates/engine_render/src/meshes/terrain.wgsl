// Terrain (decision 051) in the mesh passes, after `terrain_vertex.wgsl`:
// vertex stages into the frame's targets, the procedural material and the
// forward fragment stages.

// Hole bits of the tiles with holes (decision 051), each tile's cells
// starting at samples 0 to 256 row by row, 32 to a word.
@group(1) @binding(20) var<storage, read> terrain_holes: array<u32>;

// Pipelines of patches with holes discard their hole cells; the others
// never test.
override TERRAIN_HOLES: bool = false;

struct TerrainOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // Relative to the view's eye; w: the patch's sample spacing.
    @location(1) relative: vec4<f32>,
    // Where the point lies in its tile's cells, and where the tile's hole
    // bits start.
    @location(2) cell: vec2<f32>,
    @location(3) @interpolate(flat) holes: u32,
}

struct TerrainTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) relative: vec4<f32>,
    @location(2) cell: vec2<f32>,
    @location(3) @interpolate(flat) holes: u32,
    @location(4) previous: vec3<f32>,
}

fn terrain_output(input: PatchInput) -> TerrainOutput {
    let vertex = terrain_vertex(input);
    let relative = vertex.relative + terrain_view.eye_offset.xyz;
    var output: TerrainOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.normal = vertex.normal;
    output.relative = vec4<f32>(relative, input.shape.x);
    output.cell = vertex.cell;
    output.holes = input.holes;
    return output;
}

// Leaves no fragment in hole cells.
fn terrain_holes_test(cell: vec2<f32>, holes: u32) {
    if TERRAIN_HOLES && holes != 0xffffffffu {
        let at = vec2<u32>(clamp(floor(cell), vec2<f32>(0.0), vec2<f32>(256.0)));
        let bit = at.y * 257u + at.x;
        if (terrain_holes[holes + bit / 32u] & (1u << (bit % 32u))) != 0u {
            discard;
        }
    }
}

@vertex
fn vs_terrain(input: PatchInput) -> TerrainOutput {
    return terrain_output(input);
}

// Terrain stands still: its motion is the camera's.
@vertex
fn vs_terrain_temporal(input: PatchInput) -> TerrainTemporalOutput {
    let output = terrain_output(input);
    var temporal: TerrainTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.normal = output.normal;
    temporal.relative = output.relative;
    temporal.cell = output.cell;
    temporal.holes = output.holes;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(output.relative.xyz + camera.position.xyz, 1.0)).xyw;
    return temporal;
}

// Value noise of the material, 0 to 1.
fn terrain_hash(cell: vec2<f32>) -> f32 {
    let p = vec2<u32>(vec2<i32>(cell));
    var h = (p.x * 0x8da6b343u) ^ (p.y * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

fn terrain_value(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let a = terrain_hash(cell);
    let b = terrain_hash(cell + vec2<f32>(1.0, 0.0));
    let c = terrain_hash(cell + vec2<f32>(0.0, 1.0));
    let d = terrain_hash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Noise of feature size `size` meters at `p`, fading to its mean where a
// pixel spans more than its features (`footprint` meters).
fn terrain_detail(p: vec2<f32>, size: f32, footprint: f32) -> f32 {
    let fade = saturate(2.0 - 2.0 * footprint / size);
    return 0.5 + (terrain_value(p / size) - 0.5) * fade;
}

// Colors of levels 0 to 7 in the levels view.
const TERRAIN_LEVEL_COLORS = array<vec3<f32>, 8>(
    vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.6, 0.15), vec3<f32>(0.9, 0.9, 0.2),
    vec3<f32>(0.3, 0.85, 0.3), vec3<f32>(0.2, 0.75, 0.9), vec3<f32>(0.3, 0.4, 0.95),
    vec3<f32>(0.7, 0.35, 0.95), vec3<f32>(0.85, 0.85, 0.85),
);

// The albedo of terrain's development views (decision 051): 1 its levels,
// 2 its normals, 3 its tiles and patches outlined; `albedo` otherwise.
fn terrain_view_albedo(albedo: vec3<f32>, normal: vec3<f32>, spacing: f32, cell: vec2<f32>)
        -> vec3<f32> {
    let view = u32(terrain.origin.w);
    if view == 1u {
        let level = u32(clamp(round(log2(spacing / terrain.anchor.w)), 0.0, 7.0));
        return TERRAIN_LEVEL_COLORS[level];
    }
    if view == 2u {
        return normal * 0.5 + 0.5;
    }
    if view == 3u {
        let width = fwidth(cell) * 1.5;
        let patch_edge = abs(fract(cell / 16.0 + 0.5) - 0.5) * 16.0;
        let tile_edge = abs(fract(cell / 256.0 + 0.5) - 0.5) * 256.0;
        if any(tile_edge < width * 2.0) {
            return vec3<f32>(0.95, 0.15, 0.1);
        }
        if any(patch_edge < width) {
            return vec3<f32>(0.1, 0.1, 0.1);
        }
        return vec3<f32>(0.6);
    }
    return albedo;
}

// The surface a terrain fragment shows: the procedural material of its
// slope and height, broken up by noise across the ground.
fn terrain_point(normal_in: vec3<f32>, relative_spacing: vec4<f32>, cell: vec2<f32>)
        -> SurfacePoint {
    let relative = relative_spacing.xyz;
    let world = relative + camera.position.xyz;
    let normal = stable_normalize(normal_in);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let footprint = max(length(world_dx), length(world_dy));
    let height = world.y - terrain.origin.y;
    let broad = terrain_detail(world.xz, 60.0, footprint);
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    let level = normal.y;
    let rock_level = terrain.rock.w + (broad - 0.5) * 0.12;
    let rock = smoothstep(rock_level + 0.05, rock_level - 0.05, level);
    let snow_line = terrain.snow.w + (broad - 0.5) * 120.0 + (patchy - 0.5) * 30.0;
    let snow = smoothstep(snow_line - 15.0, snow_line + 15.0, height)
        * smoothstep(0.6, 0.78, level + (patchy - 0.5) * 0.1);
    let bare = smoothstep(0.5, 0.75, broad * 0.7 + patchy * 0.3 + (1.0 - level) * 1.2);
    var albedo = mix(terrain.grass.rgb * (0.8 + 0.45 * patchy), terrain.dirt.rgb, bare);
    albedo = mix(albedo, terrain.rock.rgb * (0.75 + 0.5 * patchy), rock);
    albedo = mix(albedo, terrain.snow.rgb, snow);
    albedo *= 0.85 + 0.3 * fine;
    albedo = terrain_view_albedo(albedo, normal, relative_spacing.w, cell);
    let roughness = mix(mix(terrain.grass.w, 0.82, rock), 0.55, snow);
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.roughness = roughness;
    point.alpha = roughness * roughness;
    point.f0 = vec3<f32>(0.04);
    point.diffuse_color = albedo;
    point.geometric = normal;
    point.normal = normal;
    point.distance = length(relative);
    point.view = stable_normalize(-relative);
    point.n_v = max(dot(normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    return widened(point, dpdx(normal), dpdy(normal));
}

fn shade_terrain(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.normal, input.relative, input.cell);
    terrain_holes_test(input.cell, input.holes);
    return finish(point, vec3<f32>(0.0));
}

fn shade_terrain_lit(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.normal, input.relative, input.cell);
    terrain_holes_test(input.cell, input.holes);
    return finish(point, local_light(point, input.clip_position.xy));
}

fn terrain_temporal_output(input: TerrainTemporalOutput) -> TerrainOutput {
    var output: TerrainOutput;
    output.clip_position = input.clip_position;
    output.normal = input.normal;
    output.relative = input.relative;
    output.cell = input.cell;
    output.holes = input.holes;
    return output;
}

@fragment
fn fs_terrain(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_terrain(input).color), 1.0);
}

@fragment
fn fs_terrain_lit(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_terrain_lit(input).color), 1.0);
}

@fragment
fn fs_terrain_post(input: TerrainOutput) -> SceneTargets {
    return scene_targets(shade_terrain(input));
}

@fragment
fn fs_terrain_post_lit(input: TerrainOutput) -> SceneTargets {
    return scene_targets(shade_terrain_lit(input));
}

@fragment
fn fs_terrain_temporal(input: TerrainTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_terrain(terrain_temporal_output(input)), input.clip_position,
        input.previous);
}

@fragment
fn fs_terrain_temporal_lit(input: TerrainTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_terrain_lit(terrain_temporal_output(input)),
        input.clip_position, input.previous);
}

// Reflection probe captures (decision 043): linear light before the grade.
@fragment
fn fs_terrain_probe(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_terrain(input).linear, 1.0);
}

@fragment
fn fs_terrain_probe_lit(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_terrain_lit(input).linear, 1.0);
}
