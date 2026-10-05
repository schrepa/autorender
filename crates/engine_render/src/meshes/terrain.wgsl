// Terrain (decision 051) in the mesh passes, after `terrain_vertex.wgsl`:
// vertex stages into the frame's targets, the ground's surface (decision
// 054) and the forward fragment stages.

// Hole bits of the tiles with holes (decision 051), each tile's cells
// starting at samples 0 to 256 row by row, 32 to a word.
@group(1) @binding(20) var<storage, read> terrain_holes: array<u32>;

// Pipelines of patches with holes discard their hole cells; the others
// never test.
override TERRAIN_HOLES: bool = false;

// Every resident tile's normals, in the same layers as its heights, and
// their sampler.
@group(1) @binding(21) var terrain_normals: texture_2d_array<f32>;
@group(1) @binding(22) var terrain_normal_sampler: sampler;
// Each terrain's levels, finest first, from `terrain.directory.x`: four
// words each (where its tiles' entries start, its tiles along x and z, and
// the meters from the eye its tiles load within, as bits); then the entries,
// row by row, each the layer of a resident tile or all ones.
@group(1) @binding(23) var<storage, read> terrain_directory: array<u32>;
// Each tile's control values (decision 054), in the same layers as its
// heights: per sample, its two strongest surface materials (bits 0 to 7
// and 8 to 15), the second's share (16 to 23) and its strongest biome.
@group(1) @binding(24) var terrain_control: texture_2d_array<u32>;

// A surface material: its linear color and roughness, then its tiling,
// height contrast and projection.
struct TerrainPaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
}
// Every terrain's surface materials, each terrain's from
// `terrain.directory.z`.
@group(1) @binding(25) var<uniform> terrain_palette: array<TerrainPaletteEntry, 512>;

struct TerrainOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Relative to the view's eye; w: the patch's sample spacing.
    @location(1) relative: vec4<f32>,
    // Where the point lies in its tile's cells, and where the tile's hole
    // bits start.
    @location(2) cell: vec2<f32>,
    @location(3) @interpolate(flat) holes: u32,
}

struct TerrainTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
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

// The normal at terrain-local `local` in `level`'s normal maps, or in the
// nearest coarser level's where its tile is not resident.
fn terrain_level_normal(local: vec2<f32>, level: u32) -> vec3<f32> {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        // Layers keep a border sample before the tile's first.
        let uv = (samples - vec2<f32>(tile * 256u) + vec2<f32>(1.5)) / 259.0;
        return textureSampleLevel(terrain_normals, terrain_normal_sampler, uv, layer, 0.0).xyz;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// Which level shades a point `distance` from the eye its tiles load
// around: the finest whose tiles load that far, turning into the next
// coarser over the second half of its reach. Reaches double level by
// level, so detail fades continuously with distance, as trilinear
// filtering turns between mipmaps, and the ground's light and materials
// hold still as levels of detail hand over, whatever level the geometry
// draws at.
struct TerrainShadingLevel {
    level: u32,
    blend: f32,
}

fn terrain_shading_level(distance: f32) -> TerrainShadingLevel {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var shading: TerrainShadingLevel;
    shading.level = 0u;
    var reach = 0.0;
    loop {
        reach = bitcast<f32>(terrain_directory[base + shading.level * 4u + 3u]);
        if distance < reach || shading.level + 1u >= levels {
            break;
        }
        shading.level++;
    }
    shading.blend = select(0.0, saturate((distance - 0.5 * reach) / (0.5 * reach)),
        shading.level + 1u < levels);
    return shading;
}

// The terrain's normal at terrain-local `local` at `shading`'s levels.
fn terrain_surface_normal(local: vec2<f32>, shading: TerrainShadingLevel) -> vec3<f32> {
    var normal = terrain_level_normal(local, shading.level);
    if shading.blend > 0.0 {
        normal = mix(normal, terrain_level_normal(local, shading.level + 1u), shading.blend);
    }
    return stable_normalize(normal);
}

// A distinct color for surface material `material` in the materials view.
fn terrain_material_color(material: u32) -> vec3<f32> {
    let hue = fract(f32(material) * 0.618034 + 0.13);
    let k = (vec3<f32>(0.0, 2.0, 4.0) / 6.0 + hue) * 6.0;
    return 0.15 + 0.7 * saturate(abs(k % 6.0 - 3.0) - 1.0);
}

// The ground of one control texel: its two materials' linear colors (rgb)
// and roughnesses (a), mixed by its blend.
fn terrain_texel_ground(texel: vec2<u32>, layer: u32) -> vec4<f32> {
    let control = textureLoad(terrain_control, texel, layer, 0).x;
    let blend = f32((control >> 16u) & 0xffu) * (1.0 / 255.0);
    let a_id = control & 0xffu;
    let b_id = (control >> 8u) & 0xffu;
    var a = terrain_palette[min(terrain.directory.z + a_id, 511u)].color;
    var b = terrain_palette[min(terrain.directory.z + b_id, 511u)].color;
    if u32(terrain.origin.w) == 4u {
        a = vec4<f32>(terrain_material_color(a_id), a.w);
        b = vec4<f32>(terrain_material_color(b_id), b.w);
    }
    return mix(a, b, blend);
}

// The ground at terrain-local `local` from `level`'s control values, or
// the nearest coarser level's where its tile is not resident: the four
// samples around it, each mixing its two materials, mixed bilinearly.
fn terrain_level_ground(local: vec2<f32>, level: u32) -> vec4<f32> {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        // Samples 0 to 256 within the tile; layers keep a border texel
        // before the first.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let near = mix(terrain_texel_ground(texel, layer),
            terrain_texel_ground(texel + vec2<u32>(1u, 0u), layer), f.x);
        let far = mix(terrain_texel_ground(texel + vec2<u32>(0u, 1u), layer),
            terrain_texel_ground(texel + vec2<u32>(1u, 1u), layer), f.x);
        return mix(near, far, f.y);
    }
    return terrain_palette[min(terrain.directory.z, 511u)].color;
}

// The ground at terrain-local `local` at `shading`'s levels.
fn terrain_surface_ground(local: vec2<f32>, shading: TerrainShadingLevel) -> vec4<f32> {
    var ground = terrain_level_ground(local, shading.level);
    if shading.blend > 0.0 {
        ground = mix(ground, terrain_level_ground(local, shading.level + 1u), shading.blend);
    }
    return ground;
}

// The surface a terrain fragment shows: its surface materials, broken up by
// noise across the ground.
fn terrain_point(relative_spacing: vec4<f32>, cell: vec2<f32>) -> SurfacePoint {
    let relative = relative_spacing.xyz;
    let world = relative + camera.position.xyz;
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_shading_level(length(world - terrain.eye.xyz));
    let normal = terrain_surface_normal(local, shading);
    let ground = terrain_surface_ground(local, shading);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let footprint = max(length(world_dx), length(world_dy));
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    var albedo = ground.rgb;
    if u32(terrain.origin.w) != 4u {
        albedo *= (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
    }
    albedo = terrain_view_albedo(albedo, normal, relative_spacing.w, cell);
    let roughness = ground.a;
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
    point.on_terrain = true;
    return widened(point, dpdx(normal), dpdy(normal));
}

fn shade_terrain(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative, input.cell);
    terrain_holes_test(input.cell, input.holes);
    return finish(point, vec3<f32>(0.0));
}

fn shade_terrain_lit(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative, input.cell);
    terrain_holes_test(input.cell, input.holes);
    return finish(point, local_light(point, input.clip_position.xy));
}

fn terrain_temporal_output(input: TerrainTemporalOutput) -> TerrainOutput {
    var output: TerrainOutput;
    output.clip_position = input.clip_position;
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
