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

// A surface material: its linear color (its base color factor with maps)
// and roughness; its tiling in meters, height contrast, projection and its
// maps' layer plus one (0 without maps); its normal scale, occlusion
// strength and metalness.
struct TerrainPaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
}
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const TERRAIN_PALETTE_ENTRIES: u32 = 341u;
// Every terrain's surface materials, each terrain's from
// `terrain.directory.z`.
@group(1) @binding(25) var<uniform> terrain_palette: array<TerrainPaletteEntry, TERRAIN_PALETTE_ENTRIES>;
// Surface materials' maps, a layer per material with maps: base color,
// tangent-space normal (green up the image, toward -z), and occlusion,
// roughness, metalness and height; and their sampler.
@group(1) @binding(26) var terrain_albedo: texture_2d_array<f32>;
@group(1) @binding(27) var terrain_material_normals: texture_2d_array<f32>;
@group(1) @binding(28) var terrain_orm: texture_2d_array<f32>;
@group(1) @binding(29) var terrain_material_sampler: sampler;

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

// Surface material `material` of this fragment's terrain.
fn terrain_material(material: u32) -> TerrainPaletteEntry {
    return terrain_palette[min(terrain.directory.z + material, TERRAIN_PALETTE_ENTRIES - 1u)];
}

// Up to four surface materials meeting at a point, and their shares.
struct TerrainBlend {
    ids: vec4<u32>,
    weights: vec4<f32>,
}

// Adds `weight` of material `material` to `blend`: to its slot, or over
// the lightest when heavier than it.
fn terrain_blend_add(blend: ptr<function, TerrainBlend>, material: u32, weight: f32) {
    var lightest = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
        if (*blend).ids[slot] == material {
            (*blend).weights[slot] += weight;
            return;
        }
        if (*blend).weights[slot] < (*blend).weights[lightest] {
            lightest = slot;
        }
    }
    if weight > (*blend).weights[lightest] {
        (*blend).ids[lightest] = material;
        (*blend).weights[lightest] = weight;
    }
}

// Adds `weight` shared between one control texel's two materials by its
// blend.
fn terrain_texel_blend(blend: ptr<function, TerrainBlend>, texel: vec2<u32>, layer: u32,
        weight: f32) {
    let control = textureLoad(terrain_control, texel, layer, 0).x;
    let share = f32((control >> 16u) & 0xffu) * (1.0 / 255.0);
    terrain_blend_add(blend, control & 0xffu, weight * (1.0 - share));
    terrain_blend_add(blend, (control >> 8u) & 0xffu, weight * share);
}

// Adds `weight` of the materials at terrain-local `local` in `level`'s
// control values, or the nearest coarser level's where its tile is not
// resident: the four samples around it, bilinearly.
fn terrain_level_blend(blend: ptr<function, TerrainBlend>, local: vec2<f32>, level: u32,
        weight: f32) {
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
        terrain_texel_blend(blend, texel, layer, weight * (1.0 - f.x) * (1.0 - f.y));
        terrain_texel_blend(blend, texel + vec2<u32>(1u, 0u), layer, weight * f.x * (1.0 - f.y));
        terrain_texel_blend(blend, texel + vec2<u32>(0u, 1u), layer, weight * (1.0 - f.x) * f.y);
        terrain_texel_blend(blend, texel + vec2<u32>(1u, 1u), layer, weight * f.x * f.y);
        return;
    }
    terrain_blend_add(blend, 0u, weight);
}

// What a terrain fragment's ground shows: its linear albedo, roughness,
// occlusion and metalness, its normal, and the share of it that has maps.
struct TerrainGround {
    albedo: vec3<f32>,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    metalness: f32,
    mapped: f32,
}

// Where a material's maps lie at a fragment: their meters per repeat,
// their layer, and their planes' shares by axis (x, y, z). Top materials
// lie from above; biplanar ones also from the sides, each side's share
// growing as the ground turns to face it, so cliffs do not stretch them.
struct TerrainMapping {
    scale: f32,
    layer: u32,
    planes: vec3<f32>,
}

fn terrain_mapping(entry: TerrainPaletteEntry, normal: vec3<f32>) -> TerrainMapping {
    var mapping: TerrainMapping;
    mapping.scale = 1.0 / entry.params.x;
    mapping.layer = u32(entry.params.w) - 1u;
    mapping.planes = vec3<f32>(0.0, 1.0, 0.0);
    if entry.params.z > 0.5 {
        let sharp = pow(abs(normal), vec3<f32>(4.0));
        // Planes barely showing are skipped.
        let shares = max(sharp / (sharp.x + sharp.y + sharp.z) - 0.03, vec3<f32>(0.0));
        mapping.planes = shares / (shares.x + shares.y + shares.z);
    }
    return mapping;
}

// Coordinates on axis `axis`'s plane: x sees (z, -y), y sees (x, z) and z
// sees (x, -y), so on the sides the maps' up is the world's.
fn terrain_project(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}

// The cosine and sine of the turn of the second lattice on top.
const TERRAIN_TURN: vec2<f32> = vec2<f32>(0.5, 0.8660254);
// How much wider the second lattice repeats on the sides.
const TERRAIN_STRETCH: f32 = 1.31;

// Plane coordinates `uv` (or their derivatives) in lattice `lattice`: 0
// the maps' own; 1 their second, turned 60 degrees on top and stretched
// across on the sides, so their strata stay level, where two lattices meet
// their repeats never line up.
fn terrain_lattice(uv: vec2<f32>, axis: u32, lattice: u32) -> vec2<f32> {
    if lattice == 0u {
        return uv;
    }
    if axis == 1u {
        return vec2<f32>(TERRAIN_TURN.x * uv.x - TERRAIN_TURN.y * uv.y,
            TERRAIN_TURN.y * uv.x + TERRAIN_TURN.x * uv.y);
    }
    return vec2<f32>(uv.x * TERRAIN_STRETCH, uv.y);
}

// `map`'s texel at `world` (with its screen derivatives) as `mapping`
// lays it in `lattice`, mixed over its planes.
fn terrain_map(map: texture_2d_array<f32>, mapping: TerrainMapping, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, lattice: u32) -> vec4<f32> {
    let shift = vec2<f32>(0.37, 0.71) * f32(lattice);
    var texel = vec4<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let share = mapping.planes[axis];
        if share <= 0.0 {
            continue;
        }
        texel += share * textureSampleGrad(map, terrain_material_sampler,
            terrain_lattice(terrain_project(world, axis) * mapping.scale, axis, lattice) + shift,
            mapping.layer,
            terrain_lattice(terrain_project(world_dx, axis) * mapping.scale, axis, lattice),
            terrain_lattice(terrain_project(world_dy, axis) * mapping.scale, axis, lattice));
    }
    return texel;
}

// The normal of a material's normal maps in `lattice` at `world` on the
// terrain's `normal`, deflections scaled by `scale`: each plane's
// tangent-space normal (x along its first coordinate, y up its image, z
// out), turned back from the lattice, added to the terrain's normal in
// that plane (whiteout blending), mixed over the planes.
fn terrain_mapped_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, lattice: u32, normal: vec3<f32>, scale: f32) -> vec3<f32> {
    let shift = vec2<f32>(0.37, 0.71) * f32(lattice);
    var mixed = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let share = mapping.planes[axis];
        if share <= 0.0 {
            continue;
        }
        let sampled = textureSampleGrad(terrain_material_normals, terrain_material_sampler,
            terrain_lattice(terrain_project(world, axis) * mapping.scale, axis, lattice) + shift,
            mapping.layer,
            terrain_lattice(terrain_project(world_dx, axis) * mapping.scale, axis, lattice),
            terrain_lattice(terrain_project(world_dy, axis) * mapping.scale, axis, lattice))
            .xyz * 2.0 - 1.0;
        var n = vec3<f32>(sampled.xy * scale, sampled.z);
        if lattice == 1u {
            // Slopes in the second lattice, in the plane's: its turn, or
            // its stretch.
            if axis == 1u {
                n = vec3<f32>(TERRAIN_TURN.x * n.x - TERRAIN_TURN.y * n.y,
                    TERRAIN_TURN.y * n.x + TERRAIN_TURN.x * n.y, n.z);
            } else {
                n.x *= TERRAIN_STRETCH;
            }
        }
        var turned: vec3<f32>;
        if axis == 0u {
            // Along +z, up +y, out along x.
            turned = vec3<f32>(abs(n.z) * normal.x, n.y + normal.y, n.x + normal.z);
        } else if axis == 2u {
            // Along +x, up +y, out along z.
            turned = vec3<f32>(n.x + normal.x, n.y + normal.y, abs(n.z) * normal.z);
        } else {
            // Along +x, up -z, out along y.
            turned = vec3<f32>(n.x + normal.x, abs(n.z) * normal.y, normal.z - n.y);
        }
        mixed += share * stable_normalize(turned);
    }
    return mixed;
}

// How much of a material repeating every `tiling` meters shows its second
// lattice at `world`: in patches about three repeats across, varying up
// cliffs as well as across the ground; where a pixel's `footprint` spans a
// patch, its maps are too fine to repeat visibly and the first shows alone.
fn terrain_second_lattice(world: vec3<f32>, tiling: f32, footprint: f32) -> f32 {
    let size = tiling * 3.1;
    let fade = saturate(2.0 - 2.0 * footprint / size);
    let p = (world.xz + world.y * vec2<f32>(0.71, -0.43)) / size;
    return smoothstep(0.4, 0.6, terrain_value(p + vec2<f32>(7.3, -2.9))) * fade;
}

// The share of `second` (0 to 1) after the two lattices' heights `first`
// and `raised` meet as materials do, as sharply as `contrast` asks.
fn terrain_lattice_share(second: f32, first: f32, raised: f32, contrast: f32) -> f32 {
    let a = 1.0 - second + contrast * first;
    let b = second + contrast * raised;
    let top = max(a, b) - 0.08;
    let lifted = max(vec2<f32>(a, b) - top, vec2<f32>(0.0));
    return mix(second, lifted.y / max(lifted.x + lifted.y, 1.0e-6), saturate(contrast));
}

// The ground at `world` (with its screen derivatives and a pixel's
// `footprint` there) on the terrain's `geometric` normal, from the control
// values of `shading`'s levels: up to four materials, each where it has
// maps raised by its height map as sharply as its height contrast asks,
// so stones and tufts stand through what they meet before it covers them.
fn terrain_ground(world: vec3<f32>, world_dx: vec3<f32>, world_dy: vec3<f32>, footprint: f32,
        geometric: vec3<f32>, shading: TerrainShadingLevel) -> TerrainGround {
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    let local = world.xz - terrain.origin.xz;
    terrain_level_blend(&blend, local, shading.level, 1.0 - shading.blend);
    if shading.blend > 0.0 {
        terrain_level_blend(&blend, local, shading.level + 1u, shading.blend);
    }
    // Each material's occlusion, roughness, metalness and height texel and
    // the share of its second lattice; materials without maps lie flat at
    // middle height.
    var texels: array<vec4<f32>, 4>;
    var seconds = vec4<f32>(0.0);
    var raised = vec4<f32>(-1.0e9);
    var contrast = 0.0;
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = blend.weights[slot];
        texels[slot] = vec4<f32>(1.0, 1.0, 1.0, 0.5);
        if weight <= 0.0 {
            continue;
        }
        let entry = terrain_material(blend.ids[slot]);
        if entry.params.w > 0.0 {
            let mapping = terrain_mapping(entry, geometric);
            let second = terrain_second_lattice(world, entry.params.x, footprint);
            var first = vec4<f32>(0.0);
            var other = vec4<f32>(0.0);
            if second < 1.0 {
                first = terrain_map(terrain_orm, mapping, world, world_dx, world_dy, 0u);
            }
            if second > 0.0 {
                other = terrain_map(terrain_orm, mapping, world, world_dx, world_dy, 1u);
            }
            var share = second;
            if second > 0.0 && second < 1.0 {
                share = terrain_lattice_share(second, first.a, other.a, entry.params.y);
            }
            texels[slot] = mix(first, other, share);
            seconds[slot] = share;
        }
        raised[slot] = weight + entry.params.y * texels[slot].a;
        contrast += weight * entry.params.y;
    }
    // Height blending: materials within a narrow band of the highest show,
    // turned toward the plain shares as contrast falls to 0.
    let highest = max(max(raised.x, raised.y), max(raised.z, raised.w));
    let lifted = max(raised - vec4<f32>(highest - 0.08), vec4<f32>(0.0));
    let weights = mix(blend.weights, lifted / max(dot(lifted, vec4<f32>(1.0)), 1.0e-6),
        saturate(contrast));
    var ground: TerrainGround;
    ground.albedo = vec3<f32>(0.0);
    ground.roughness = 0.0;
    ground.occlusion = 0.0;
    ground.metalness = 0.0;
    ground.mapped = 0.0;
    var normal = vec3<f32>(0.0);
    let materials_view = u32(terrain.origin.w) == 4u;
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = weights[slot];
        if weight <= 0.0 {
            continue;
        }
        let entry = terrain_material(blend.ids[slot]);
        var color = entry.color.rgb;
        var turned = geometric;
        let texel = texels[slot];
        if entry.params.w > 0.0 {
            let mapping = terrain_mapping(entry, geometric);
            let second = seconds[slot];
            var albedo = vec3<f32>(0.0);
            turned = vec3<f32>(0.0);
            if second < 1.0 {
                albedo += (1.0 - second)
                    * terrain_map(terrain_albedo, mapping, world, world_dx, world_dy, 0u).rgb;
                turned += (1.0 - second) * terrain_mapped_normal(mapping, world, world_dx,
                    world_dy, 0u, geometric, entry.maps.x);
            }
            if second > 0.0 {
                albedo += second
                    * terrain_map(terrain_albedo, mapping, world, world_dx, world_dy, 1u).rgb;
                turned += second * terrain_mapped_normal(mapping, world, world_dx, world_dy, 1u,
                    geometric, entry.maps.x);
            }
            color *= albedo;
            ground.mapped += weight;
        }
        if materials_view {
            color = terrain_material_color(blend.ids[slot]);
        }
        ground.albedo += weight * color;
        ground.roughness += weight * entry.color.w * texel.g;
        ground.occlusion += weight * (1.0 + entry.maps.y * (texel.r - 1.0));
        ground.metalness += weight * entry.maps.z * texel.b;
        normal += weight * turned;
    }
    ground.normal = stable_normalize(normal);
    return ground;
}

// The surface a terrain fragment shows: its surface materials, varied
// across the ground so their maps do not repeat evenly.
fn terrain_point(relative_spacing: vec4<f32>, cell: vec2<f32>) -> SurfacePoint {
    let relative = relative_spacing.xyz;
    let world = relative + camera.position.xyz;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let footprint = max(length(world_dx), length(world_dy));
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_shading_level(length(world - terrain.eye.xyz));
    let geometric = terrain_surface_normal(local, shading);
    let ground = terrain_ground(world, world_dx, world_dy, footprint, geometric, shading);
    let broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, footprint);
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    let view = u32(terrain.origin.w);
    var albedo = ground.albedo;
    // The normals and materials views show the terrain's own normals.
    var normal = ground.normal;
    if view == 2u || view == 4u {
        normal = geometric;
    }
    if view != 4u {
        // Plain colors vary at every scale; maps carry their own detail
        // and vary broadly.
        let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
        let mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
        albedo *= mix(plain, mapped, ground.mapped);
    }
    albedo = terrain_view_albedo(albedo, normal, relative_spacing.w, cell);
    let roughness = max(ground.roughness, 0.03);
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = ground.occlusion;
    point.roughness = roughness;
    point.alpha = roughness * roughness;
    point.f0 = mix(vec3<f32>(0.04), albedo, ground.metalness);
    point.diffuse_color = albedo * (1.0 - ground.metalness);
    point.geometric = geometric;
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
