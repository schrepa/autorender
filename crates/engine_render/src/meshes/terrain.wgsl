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
// strength, metalness and mean occlusion; and its maps' mean color and
// roughness, its factors applied, which stand in for them where they
// repeat too small on screen to show.
struct TerrainPaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    // The meters its heights span where the ground's relief shows (decision
    // 082), the meters tracks sink it at most (decision 083), and the
    // variance of its normal maps' slopes at their scale, which its
    // roughness takes over where they fade (decision 098).
    relief: vec4<f32>,
    // Its cover's canopy (decision 090): the linear color its blades show
    // seen from afar, and their leaves' area per square meter of ground (0
    // without cover).
    canopy: vec4<f32>,
}
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const TERRAIN_PALETTE_ENTRIES: u32 = 170u;
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
// Each terrain's sky visibility map (decision 055): over the whole terrain,
// the share of the sky its ground sees about its normal (red).
@group(1) @binding(30) var terrain_sky: texture_2d_array<f32>;
// Runtime deformation's map (decision 083; `deformation.wgsl`), in its
// reference units: each block's deepest track or highest rim, widened by a
// block; each texel's track depth and rim height; and each texel's slopes.
@group(1) @binding(37) var<storage, read> terrain_deformation: array<u32>;

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

// How much of the next coarser level's normals (x) and surface materials
// (y) still show over layer `layer`'s tile (decision 114): fading in after
// it arrived, and wholly the coarser level's materials until its own
// arrive. The first words of `terrain_holes` hold them, two layers a word.
fn terrain_layer_fades(layer: u32) -> vec2<f32> {
    let half = (terrain_holes[layer / 2u] >> ((layer % 2u) * 16u)) & 0xffffu;
    return vec2<f32>(f32(half & 0xffu), f32(half >> 8u)) * (1.0 / 255.0);
}

// The normal at terrain-local `local` in `level`'s normal maps, or in the
// nearest coarser level's where its tile is not resident; a tile fading in
// shows the next resident coarser level's under it (decision 114).
fn terrain_level_normal(local: vec2<f32>, level: u32) -> vec3<f32> {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var normal = vec3<f32>(0.0);
    var left = 1.0;
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
        let fade = terrain_layer_fades(layer).x;
        normal += textureSampleLevel(terrain_normals, terrain_normal_sampler, uv, layer, 0.0).xyz
            * (left * (1.0 - fade));
        left *= fade;
        if left <= 0.0 {
            return normal;
        }
    }
    return normal + vec3<f32>(0.0, left, 0.0);
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
// the first lightest when heavier than it. Whole-vector selects, so the
// slots stay in registers.
fn terrain_blend_add(blend: ptr<function, TerrainBlend>, material: u32, weight: f32) {
    let same = (*blend).ids == vec4<u32>(material);
    if any(same) {
        (*blend).weights += select(vec4<f32>(0.0), vec4<f32>(weight), same);
        return;
    }
    let weights = (*blend).weights;
    let lightest = min(min(weights.x, weights.y), min(weights.z, weights.w));
    if weight > lightest {
        let at = weights == vec4<f32>(lightest);
        let first = at & !vec4<bool>(false, at.x, at.x | at.y, at.x | at.y | at.z);
        (*blend).ids = select((*blend).ids, vec4<u32>(material), first);
        (*blend).weights = select(weights, vec4<f32>(weight), first);
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
// resident: the four samples around it, bilinearly. A tile fading in, or
// without its own materials yet, shows the next resident coarser level's
// under it (decision 114).
fn terrain_level_blend(blend: ptr<function, TerrainBlend>, local: vec2<f32>, level: u32,
        weight: f32) {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var left = weight;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        let fade = terrain_layer_fades(layer).y;
        if fade < 1.0 {
            // Samples 0 to 256 within the tile; layers keep a border texel
            // before the first.
            let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
            let corner = min(vec2<u32>(within), vec2<u32>(255u));
            let f = within - vec2<f32>(corner);
            let texel = corner + vec2<u32>(1u);
            let own = left * (1.0 - fade);
            terrain_texel_blend(blend, texel, layer, own * (1.0 - f.x) * (1.0 - f.y));
            terrain_texel_blend(blend, texel + vec2<u32>(1u, 0u), layer, own * f.x * (1.0 - f.y));
            terrain_texel_blend(blend, texel + vec2<u32>(0u, 1u), layer, own * (1.0 - f.x) * f.y);
            terrain_texel_blend(blend, texel + vec2<u32>(1u, 1u), layer, own * f.x * f.y);
        }
        left *= fade;
        if left <= 0.0 {
            return;
        }
    }
    terrain_blend_add(blend, 0u, left);
}

// What a terrain fragment's ground shows: its linear albedo, roughness,
// occlusion and metalness, its normal, and the share of it that has maps;
// for its variety and detail (decision 090), the shares of it growing
// cover, showing as canopy, showing all its maps and showing their color.
struct TerrainGround {
    albedo: vec3<f32>,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    metalness: f32,
    mapped: f32,
    vegetated: f32,
    canopy: f32,
    full: f32,
    colored: f32,
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

// `map`'s texel at `world` (with its screen derivatives) on axis `axis`'s
// plane in lattice `lattice`, as `mapping` lays it.
fn terrain_plane_texel(map: texture_2d_array<f32>, mapping: TerrainMapping, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, axis: u32, lattice: u32) -> vec4<f32> {
    let shift = vec2<f32>(0.37, 0.71) * f32(lattice);
    return textureSampleGrad(map, terrain_material_sampler,
        terrain_lattice(terrain_project(world, axis) * mapping.scale, axis, lattice) + shift,
        mapping.layer,
        terrain_lattice(terrain_project(world_dx, axis) * mapping.scale, axis, lattice),
        terrain_lattice(terrain_project(world_dy, axis) * mapping.scale, axis, lattice));
}

// `map`'s texel at `world` (with its screen derivatives) as `mapping`
// lays it in `lattice`, mixed over its planes.
fn terrain_map(map: texture_2d_array<f32>, mapping: TerrainMapping, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, lattice: u32) -> vec4<f32> {
    var texel = vec4<f32>(0.0);
    if mapping.planes.x > 0.0 {
        texel += mapping.planes.x
            * terrain_plane_texel(map, mapping, world, world_dx, world_dy, 0u, lattice);
    }
    if mapping.planes.y > 0.0 {
        texel += mapping.planes.y
            * terrain_plane_texel(map, mapping, world, world_dx, world_dy, 1u, lattice);
    }
    if mapping.planes.z > 0.0 {
        texel += mapping.planes.z
            * terrain_plane_texel(map, mapping, world, world_dx, world_dy, 2u, lattice);
    }
    return texel;
}

// The normal of a material's normal map on axis `axis`'s plane in
// `lattice` at `world` on the terrain's `normal`, deflections scaled by
// `scale`: the plane's tangent-space normal (x along its first coordinate,
// y up its image, z out), turned back from the lattice, added to the
// terrain's normal in that plane (whiteout blending).
fn terrain_plane_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, axis: u32, lattice: u32, normal: vec3<f32>, scale: f32)
        -> vec3<f32> {
    let sampled = terrain_plane_texel(terrain_material_normals, mapping, world, world_dx,
        world_dy, axis, lattice).xyz * 2.0 - 1.0;
    var n = vec3<f32>(sampled.xy * scale, sampled.z);
    if lattice == 1u {
        // Slopes in the second lattice, in the plane's: its turn, or its
        // stretch.
        if axis == 1u {
            n = vec3<f32>(TERRAIN_TURN.x * n.x - TERRAIN_TURN.y * n.y,
                TERRAIN_TURN.y * n.x + TERRAIN_TURN.x * n.y, n.z);
        } else {
            n.x *= TERRAIN_STRETCH;
        }
    }
    if axis == 0u {
        // Along +z, up +y, out along x.
        return stable_normalize(vec3<f32>(abs(n.z) * normal.x, n.y + normal.y, n.x + normal.z));
    }
    if axis == 2u {
        // Along +x, up +y, out along z.
        return stable_normalize(vec3<f32>(n.x + normal.x, n.y + normal.y, abs(n.z) * normal.z));
    }
    // Along +x, up -z, out along y.
    return stable_normalize(vec3<f32>(n.x + normal.x, abs(n.z) * normal.y, normal.z - n.y));
}

// The normal of a material's normal maps in `lattice` at `world` on the
// terrain's `normal`, deflections scaled by `scale`, mixed over its
// planes.
fn terrain_mapped_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, lattice: u32, normal: vec3<f32>, scale: f32) -> vec3<f32> {
    var mixed = vec3<f32>(0.0);
    if mapping.planes.x > 0.0 {
        mixed += mapping.planes.x
            * terrain_plane_normal(mapping, world, world_dx, world_dy, 0u, lattice, normal, scale);
    }
    if mapping.planes.y > 0.0 {
        mixed += mapping.planes.y
            * terrain_plane_normal(mapping, world, world_dx, world_dy, 1u, lattice, normal, scale);
    }
    if mapping.planes.z > 0.0 {
        mixed += mapping.planes.z
            * terrain_plane_normal(mapping, world, world_dx, world_dy, 2u, lattice, normal, scale);
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

// How much of its maps a material repeating every `tiling` meters shows
// where a pixel spans `footprint` meters: all of them while a repeat spans
// 16 pixels or more, none from 8 down, where the mipmaps' last levels would
// show little but their mean, which stands in.
fn terrain_map_detail(footprint: f32, tiling: f32) -> f32 {
    return 1.0 - smoothstep(0.0625, 0.125, footprint / tiling);
}

// How much of its normal maps' deflection a material shows: all while a
// repeat spans 64 pixels or more, none from 32 down, where their mipmaps
// have smoothed out.
fn terrain_relief(footprint: f32, tiling: f32) -> f32 {
    return 1.0 - smoothstep(0.015625, 0.03125, footprint / tiling);
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

// Where a fragment is and how it is seen, as the ground's passes read it:
// its world position and how that changes per pixel on the terrain's smooth
// surface (decision 098), a pixel's footprint there (meters, its long
// side), the footprint the maps' anisotropic filtering resolves (its short
// side, or a sixteenth of its long side where filtering runs out), the
// terrain's normal, the way toward the eye and the meters to it, and the
// share of its cover's leaves blades no longer show there (decision 098).
struct TerrainSite {
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    footprint: f32,
    texel: f32,
    geometric: vec3<f32>,
    toward: vec3<f32>,
    distance: f32,
    thinned: f32,
}

// The share of the cover's leaves blades no longer show `distance` meters
// from the eye at `world` by their law (`terrain.cover`, decision 098): all
// of them where a terrain's surroundings show its cover as canopy from the
// eye on; none without cover.
fn terrain_thinned(world: vec3<f32>, distance: f32) -> f32 {
    let cover = terrain.cover;
    if cover.z <= 0.0 {
        return 0.0;
    }
    if cover.x <= 0.0 {
        return 1.0;
    }
    let share = cover_share(distance, cover.x, cover.y, cover_end(world.xz, cover.y));
    return 1.0 - cover_shown(distance, share, cover.x);
}

// How much light reaches blades on average, which their canopy takes
// (decision 105): their occlusion rises from 0.35 at the root to 1 at the
// tip (`meshes/cover.wgsl`), 0.675 over their height, and the half of them
// seen through from the side away from the sun shows 0.8 of its color.
const TERRAIN_CANOPY_OCCLUSION: f32 = 0.6;

// How bright blades seen at `sine` (the sine of the view's angle over the
// ground) show their color on average, which their canopy shows too
// (decision 105): seen along the ground their lit tops give way to their
// sides and the shade among them. Measured against blades under the low
// and the high sun alike: about 0.6 along the ground, 0.7 from 25 degrees
// up and 0.75 from 45.
fn terrain_canopy_shown(sine: f32) -> f32 {
    return 0.6 + 0.22 * saturate(sine);
}

// The normal blades standing on ground of normal `ground` light with, on
// average (decision 090; `meshes/cover.wgsl`): leaning most of the way up
// from their sides toward the sun, which face it by two over pi on
// average.
fn terrain_blade_normal(ground: vec3<f32>) -> vec3<f32> {
    let across = vec3<f32>(lighting.sun.x, 0.0, lighting.sun.z);
    let side = across * (0.6366 / max(length(across), 1.0e-4));
    return stable_normalize(mix(side, ground, 0.6));
}

// How much of the ground at `site` a cover whose leaves' area is `leaves`
// per square meter shows as its canopy (decision 090): as much of the
// leaves as its blades no longer show by their law (`terrain.cover`,
// decision 098) hides of the ground along the view, 1 - exp(-G leaves /
// sin), all of it at grazing views.
fn terrain_canopy(leaves: f32, site: TerrainSite) -> f32 {
    if site.thinned <= 0.0 || leaves <= 0.0 {
        return 0.0;
    }
    let sine = max(abs(dot(site.toward, site.geometric)), 0.03);
    return site.thinned * (1.0 - exp(-0.5 * leaves * terrain.cover.z / sine));
}

// One material of a fragment's blend after its first pass: its occlusion,
// roughness, metalness and height texel, the share of its second lattice,
// how much of all its maps shows and how much of its color map, the second
// lattice's share by its pattern alone, its weight raised by its height,
// and its share of the height contrast.
struct TerrainSlot {
    texel: vec4<f32>,
    second: f32,
    detail: f32,
    colored: f32,
    pattern: f32,
    raised: f32,
    contrast: f32,
}

// The first pass of material `material` at weight `weight`: its heights
// where its maps show; materials without maps, or too far to show them,
// lie flat at middle height. All its maps show while a repeat spans enough
// pixels along the footprint's long side; its color map alone farther out,
// as long as its filtering keeps it (decision 090).
fn terrain_slot(material: u32, weight: f32, site: TerrainSite) -> TerrainSlot {
    var slot: TerrainSlot;
    slot.texel = vec4<f32>(1.0, 1.0, 1.0, 0.5);
    slot.raised = -1.0e9;
    if weight <= 0.0 {
        return slot;
    }
    let entry = terrain_material(material);
    if entry.params.w > 0.0 {
        slot.detail = terrain_map_detail(site.footprint, entry.params.x);
        slot.colored = terrain_map_detail(site.texel, entry.params.x);
    }
    if slot.colored > 0.0 {
        slot.pattern = terrain_second_lattice(site.world, entry.params.x, site.texel);
        slot.second = slot.pattern;
    }
    if slot.detail > 0.0 {
        let mapping = terrain_mapping(entry, site.geometric);
        let second = slot.pattern;
        var first = vec4<f32>(0.0);
        var other = vec4<f32>(0.0);
        if second < 1.0 {
            first = terrain_map(terrain_orm, mapping, site.world, site.world_dx, site.world_dy,
                0u);
        }
        if second > 0.0 {
            other = terrain_map(terrain_orm, mapping, site.world, site.world_dx, site.world_dy,
                1u);
        }
        slot.second = second;
        if second > 0.0 && second < 1.0 {
            slot.second = terrain_lattice_share(second, first.a, other.a, entry.params.y);
        }
        slot.texel = mix(first, other, slot.second);
    }
    slot.raised = weight + entry.params.y * mix(0.5, slot.texel.a, slot.detail);
    slot.contrast = weight * entry.params.y;
    return slot;
}

// Adds material `material`'s share `weight` of the ground to `ground` and
// its normal to `normal`, from its first pass `slot`: its maps' colors
// and normals where they show, their means beyond, or its plain color.
fn terrain_slot_ground(ground: ptr<function, TerrainGround>, normal: ptr<function, vec3<f32>>,
        material: u32, weight: f32, slot: TerrainSlot, site: TerrainSite) {
    if weight <= 0.0 {
        return;
    }
    let entry = terrain_material(material);
    var color = entry.far.rgb;
    var roughness = entry.far.w;
    var occlusion = entry.maps.w;
    var metalness = entry.maps.z;
    var turned = site.geometric;
    if slot.colored > slot.detail {
        // Past its maps' full detail, its color map alone, the lattices
        // shared by their pattern.
        let mapping = terrain_mapping(entry, site.geometric);
        var albedo = vec3<f32>(0.0);
        if slot.pattern < 1.0 {
            albedo += (1.0 - slot.pattern) * terrain_map(terrain_albedo, mapping, site.world,
                site.world_dx, site.world_dy, 0u).rgb;
        }
        if slot.pattern > 0.0 {
            albedo += slot.pattern * terrain_map(terrain_albedo, mapping, site.world,
                site.world_dx, site.world_dy, 1u).rgb;
        }
        color = mix(color, entry.color.rgb * albedo, slot.colored);
    }
    // Normal maps where their deflections still show, fading with the
    // other maps (decision 098).
    let shown = terrain_relief(site.texel, entry.params.x) * slot.detail;
    if slot.detail > 0.0 {
        let mapping = terrain_mapping(entry, site.geometric);
        let relief = entry.maps.x;
        let deflected = shown > 0.0 && relief > 0.0;
        var albedo = vec3<f32>(0.0);
        var mapped = vec3<f32>(0.0);
        if slot.second < 1.0 {
            albedo += (1.0 - slot.second) * terrain_map(terrain_albedo, mapping, site.world,
                site.world_dx, site.world_dy, 0u).rgb;
            if deflected {
                mapped += (1.0 - slot.second) * terrain_mapped_normal(mapping, site.world,
                    site.world_dx, site.world_dy, 0u, site.geometric, relief);
            }
        }
        if slot.second > 0.0 {
            albedo += slot.second * terrain_map(terrain_albedo, mapping, site.world,
                site.world_dx, site.world_dy, 1u).rgb;
            if deflected {
                mapped += slot.second * terrain_mapped_normal(mapping, site.world,
                    site.world_dx, site.world_dy, 1u, site.geometric, relief);
            }
        }
        color = mix(color, entry.color.rgb * albedo, slot.detail);
        roughness = mix(roughness, entry.color.w * slot.texel.g, slot.detail);
        occlusion = mix(occlusion, 1.0 + entry.maps.y * (slot.texel.r - 1.0), slot.detail);
        metalness *= mix(1.0, slot.texel.b, slot.detail);
        if deflected {
            turned = stable_normalize(mix(site.geometric, mapped, shown));
        }
    }
    // The slopes normal maps no longer show roughen the surface instead
    // (decision 098), as their variance widens its highlights.
    let lost = entry.relief.z * (1.0 - shown);
    if lost > 0.0 {
        let alpha = roughness * roughness;
        roughness = sqrt(sqrt(min(alpha * alpha + lost, 1.0)));
    }
    if entry.params.w > 0.0 {
        (*ground).mapped += weight;
    }
    (*ground).full += weight * slot.detail;
    (*ground).colored += weight * slot.colored;
    if entry.canopy.w > 0.0 {
        // Ground under cover shows as its canopy where blades thin out
        // (decision 090): their color, rougher and shaded among the
        // blades, and lit as blades are, their sides toward the sun.
        let canopy = terrain_canopy(entry.canopy.w, site);
        let shown = terrain_canopy_shown(abs(dot(site.toward, site.geometric)));
        color = mix(color, entry.canopy.rgb * shown, canopy);
        roughness = mix(roughness, 0.7, canopy);
        // The sky they reflect too: a flat ground seen along it would
        // mirror much more of it than blades facing the eye do.
        occlusion = mix(occlusion, TERRAIN_CANOPY_OCCLUSION * shown, canopy);
        turned = mix(turned, terrain_blade_normal(site.geometric), canopy);
        (*ground).vegetated += weight;
        (*ground).canopy += weight * canopy;
    }
    if u32(terrain.origin.w) == 4u {
        color = terrain_material_color(material);
    }
    (*ground).albedo += weight * color;
    (*ground).roughness += weight * roughness;
    (*ground).occlusion += weight * occlusion;
    (*ground).metalness += weight * metalness;
    *normal += weight * turned;
}

// The ground at `site` from the control values of `shading`'s levels: up
// to four materials, each where it has maps raised by its height map as
// sharply as its height contrast asks, so stones and tufts stand through
// what they meet before it covers them.
fn terrain_ground(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainGround {
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    let local = site.world.xz - terrain.origin.xz;
    terrain_level_blend(&blend, local, shading.level, 1.0 - shading.blend);
    if shading.blend > 0.0 {
        terrain_level_blend(&blend, local, shading.level + 1u, shading.blend);
    }
    let first = terrain_slot(blend.ids.x, blend.weights.x, site);
    let second = terrain_slot(blend.ids.y, blend.weights.y, site);
    let third = terrain_slot(blend.ids.z, blend.weights.z, site);
    let fourth = terrain_slot(blend.ids.w, blend.weights.w, site);
    // Height blending: materials within a narrow band of the highest show,
    // turned toward the plain shares as contrast falls to 0.
    let raised = vec4<f32>(first.raised, second.raised, third.raised, fourth.raised);
    let contrast = first.contrast + second.contrast + third.contrast + fourth.contrast;
    let highest = max(max(raised.x, raised.y), max(raised.z, raised.w));
    let lifted = max(raised - vec4<f32>(highest - 0.08), vec4<f32>(0.0));
    let weights = mix(blend.weights, lifted / max(dot(lifted, vec4<f32>(1.0)), 1.0e-6),
        saturate(contrast));
    var ground: TerrainGround;
    var normal = vec3<f32>(0.0);
    terrain_slot_ground(&ground, &normal, blend.ids.x, weights.x, first, site);
    terrain_slot_ground(&ground, &normal, blend.ids.y, weights.y, second, site);
    terrain_slot_ground(&ground, &normal, blend.ids.z, weights.z, third, site);
    terrain_slot_ground(&ground, &normal, blend.ids.w, weights.w, fourth, site);
    ground.normal = stable_normalize(normal);
    return ground;
}

// The terrain's sky map at terrain-local `local` once built: the share of
// the sky the ground sees (red, decision 055), and its large-scale variety
// (decision 090), hollows (green) and ridges (blue); before, all the sky
// and level ground.
fn terrain_sky_map(local: vec2<f32>) -> vec4<f32> {
    if terrain.directory.w == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 1.0);
    }
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    return textureSampleLevel(terrain_sky, terrain_normal_sampler, local / extent,
        terrain.directory.w - 1u, 0.0);
}

// The light the terrain around a hollow gives back per unit albedo for the
// sky it hides: sunlit and skylit ground, as level ground would be.
fn terrain_bounce() -> vec3<f32> {
    return lighting.sun_color.rgb * (0.35 * saturate(lighting.sun.y))
        + sky_average(vec3<f32>(0.0, 1.0, 0.0)) * (0.5 * lighting.zenith.w);
}

// One material's part in the ground's relief (decision 082): the meters it
// sinks the ground at most, already weighted and faded, and where its
// maps' heights lie: their scale, layer and the derivatives they are read
// with.
struct TerrainReliefLayer {
    depth: f32,
    scale: f32,
    layer: u32,
    dx: vec2<f32>,
    dy: vec2<f32>,
}

fn terrain_relief_layer(material: u32, weight: f32, fade: f32, site: TerrainSite)
        -> TerrainReliefLayer {
    var layer: TerrainReliefLayer;
    layer.depth = 0.0;
    if weight <= 0.0 || material == 0xffffffffu {
        return layer;
    }
    let entry = terrain_material(material);
    // Materials with maps, lying from above, close enough to show them.
    if entry.params.w <= 0.0 || entry.params.z > 0.5 {
        return layer;
    }
    layer.scale = 1.0 / entry.params.x;
    layer.layer = u32(entry.params.w) - 1u;
    layer.dx = site.world_dx.xz * layer.scale;
    layer.dy = site.world_dy.xz * layer.scale;
    layer.depth = entry.relief.x * weight * fade
        * terrain_map_detail(site.footprint, entry.params.x);
    return layer;
}

// How far below the ground's top `layer` sinks it at `point` (x and z).
fn terrain_relief_sink(layer: TerrainReliefLayer, point: vec2<f32>) -> f32 {
    if layer.depth <= 0.0 {
        return 0.0;
    }
    let height = textureSampleGrad(terrain_orm, terrain_material_sampler, point * layer.scale,
        layer.layer, layer.dx, layer.dy).a;
    return layer.depth * (1.0 - height);
}

// World texel `texel`'s slot in the deformation map, whose texels wrap.
fn terrain_deform_slot(texel: vec2<i32>) -> vec2<i32> {
    let size = terrain.deform_grid.z;
    return ((texel % size) + size) % size;
}

// World point `point` (x and z) in the deformation map's texels, from its
// first texel's corner.
fn terrain_deform_local(point: vec2<f32>) -> vec2<f32> {
    return point * terrain.deform.x - vec2<f32>(terrain.deform_grid.xy);
}

// The most tracks or rims near world `point` raise or sink the ground
// (meters, as refilled): 0 beyond the map, or while deformation is off.
fn terrain_deform_reach(point: vec2<f32>) -> f32 {
    let size = terrain.deform_grid.z;
    if size <= 0 {
        return 0.0;
    }
    let local = terrain_deform_local(point);
    if any(local < vec2<f32>(0.0)) || any(local >= vec2<f32>(f32(size))) {
        return 0.0;
    }
    let block = terrain_deform_slot(vec2<i32>(floor(local)) + terrain.deform_grid.xy) / 16;
    let word = u32(block.y * terrain.deform_grid.w + block.x);
    return bitcast<f32>(terrain_deformation[word]) * terrain.deform.y;
}

// The deformation map's texel words from `start` (tracks', or slopes'),
// bilinear at world `point`, as refilled and fading over the map's outer
// eighth.
fn terrain_deform_sample(point: vec2<f32>, start: i32) -> vec2<f32> {
    let size = terrain.deform_grid.z;
    let local = terrain_deform_local(point);
    let corner = local - 0.5;
    let base = floor(corner);
    let f = corner - base;
    var values: array<vec2<f32>, 4>;
    for (var index = 0; index < 4; index++) {
        let texel = vec2<i32>(base) + vec2<i32>(index & 1, index >> 1);
        let slot = terrain_deform_slot(texel + terrain.deform_grid.xy);
        let word = u32(start + slot.y * size + slot.x);
        let inside = all(texel >= vec2<i32>(0)) && all(texel < vec2<i32>(size));
        values[index] = select(vec2<f32>(0.0), unpack2x16float(terrain_deformation[word]),
            inside);
    }
    let edge = min(local, vec2<f32>(f32(size)) - local) / (0.125 * f32(size));
    let fade = smoothstep(0.0, 1.0, saturate(min(edge.x, edge.y)));
    return mix(mix(values[0], values[1], f.x), mix(values[2], values[3], f.x), f.y)
        * (terrain.deform.y * fade);
}

// Tracks' depth and rims' height at world `point` (meters).
fn terrain_deform_track(point: vec2<f32>) -> vec2<f32> {
    let blocks = terrain.deform_grid.w;
    return terrain_deform_sample(point, blocks * blocks);
}

// The slopes tracks and rims leave at world `point` (rise per meter along
// x and z).
fn terrain_deform_slope(point: vec2<f32>) -> vec2<f32> {
    let size = terrain.deform_grid.z;
    let blocks = terrain.deform_grid.w;
    return terrain_deform_sample(point, blocks * blocks + size * size);
}

// The height (meters, up) tracks and rims of `track` leave ground whose
// materials let tracks sink it `deform` meters at most: rims as high as
// half that, only where no track sinks it, and nothing on ground too firm
// to keep a track.
fn terrain_track_height(track: vec2<f32>, deform: f32) -> f32 {
    let carve = min(track.x, deform);
    let rim = min(track.y, 0.5 * deform) * (1.0 - saturate(4.0 * track.x / max(track.y, 1.0e-6)));
    return (rim - carve) * saturate(deform / 0.02);
}

// How much of the map's slopes the ground keeps: none where tracks reach
// deeper than its materials let them, which leaves their floors level.
fn terrain_track_slope_share(track: vec2<f32>, deform: f32) -> f32 {
    return saturate(deform / 0.02) * (1.0 - smoothstep(0.85 * deform, deform, track.x));
}

// Where a fragment's view ray meets the ground: `world` sunk to where it
// shades, and `deform` the meters its materials let tracks sink it at
// most where the deformation map holds tracks nearby (otherwise 0).
struct TerrainRelief {
    world: vec3<f32>,
    deform: f32,
}

// How far below the ground's top the relief layers `a` and `b` and the
// tracks of the ground's `deform` (faded by `fade`) sink it at `point`.
fn terrain_sink(a: TerrainReliefLayer, b: TerrainReliefLayer, point: vec2<f32>, deform: f32,
        fade: f32) -> f32 {
    var sink = terrain_relief_sink(a, point) + terrain_relief_sink(b, point);
    if deform > 0.0 {
        sink -= terrain_track_height(terrain_deform_track(point), deform) * fade;
    }
    return sink;
}

// Where a fragment's view ray meets the ground's relief (decision 082):
// within the relief distance, the ray marches down through the heights of
// its two heaviest top-projected materials, each as deep as its relief
// reaches and flattening toward the distance's end, and through the
// tracks deformers pressed (decision 083), from the top of their rims to
// the bottom of the deepest; the ground shades from where the ray meets
// them, sunk that deep.
fn terrain_parallax(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainRelief {
    var relief: TerrainRelief;
    relief.world = site.world;
    relief.deform = 0.0;
    let reach = terrain.relief.x;
    let to_point = site.world - camera.position.xyz;
    let distance = length(to_point);
    let tracked = terrain_deform_reach(site.world.xz);
    let marching = reach > 0.0 && distance < reach;
    if !marching && tracked < 5.0e-4 {
        return relief;
    }
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    terrain_level_blend(&blend, site.world.xz - terrain.origin.xz, shading.level, 1.0);
    let weights = blend.weights;
    if tracked >= 5.0e-4 {
        var deform = 0.0;
        for (var index = 0u; index < 4u; index++) {
            if weights[index] > 0.0 {
                deform += weights[index] * terrain_material(blend.ids[index]).relief.y;
            }
        }
        relief.deform = deform;
    }
    if !marching {
        return relief;
    }
    // The two heaviest materials.
    var heaviest = 0u;
    for (var index = 1u; index < 4u; index++) {
        if weights[index] > weights[heaviest] {
            heaviest = index;
        }
    }
    var next = select(0u, 1u, heaviest == 0u);
    for (var index = 0u; index < 4u; index++) {
        if index != heaviest && weights[index] > weights[next] {
            next = index;
        }
    }
    // Depth runs into the ground along its normal; grazing rays, which
    // would stretch the relief across the ground, see it flatten.
    let ray = to_point / distance;
    let into = -dot(ray, site.geometric);
    let fade = (1.0 - smoothstep(0.6 * reach, reach, distance)) * smoothstep(0.08, 0.35, into);
    let a = terrain_relief_layer(blend.ids[heaviest], weights[heaviest], fade, site);
    let b = terrain_relief_layer(blend.ids[next], weights[next], fade, site);
    // Tracks as deep and rims as high as the map's nearby tracks reach and
    // the ground's materials let them.
    let deform = select(0.0, relief.deform, fade > 0.0);
    let sunk = min(tracked, deform) * fade;
    let raised = min(tracked, 0.5 * deform) * fade;
    let top = -raised;
    let depth = a.depth + b.depth + sunk;
    if depth - top < 0.002 {
        return relief;
    }
    // Down the ray a step of the relief's depth at a time, the ray's
    // travel across per meter of depth bounded; from above the rims, the
    // ray starting before the ground it meets.
    let travel = ray / max(into, 0.08);
    let across = length(travel.xz);
    let along = travel.xz * min(1.0, 3.0 / max(across, 1.0e-6));
    // Relief whose deepest point would move under a pixel takes no march.
    if (depth - top) * min(across, 3.0) < 0.75 * site.footprint {
        return relief;
    }
    let base = mix(3.0, terrain.relief.y, fade);
    // A step every two centimeters of tracks, up to twice the relief's.
    let steps = max(u32(round(clamp((depth - top) / 0.02, base, 2.0 * terrain.relief.y))), 2u);
    var previous = top;
    var gap = top - terrain_sink(a, b, site.world.xz + along * top, deform, fade);
    var hit = top;
    if gap < 0.0 {
        hit = depth;
        for (var step = 1u; step <= steps; step++) {
            let at = top + (depth - top) * f32(step) / f32(steps);
            let point = site.world.xz + along * at;
            let next_gap = at - terrain_sink(a, b, point, deform, fade);
            if next_gap >= 0.0 {
                // Between the last step and this one, where the ray reaches
                // the ground.
                hit = mix(previous, at, gap / min(gap - next_gap, -1.0e-6));
                break;
            }
            previous = at;
            gap = next_gap;
        }
    }
    relief.world = site.world + vec3<f32>(along.x, travel.y, along.y) * hit;
    return relief;
}

// The surface a terrain fragment shows: its surface materials, varied
// across the ground so their maps do not repeat evenly, lit by the sky it
// sees and by the ground around where that hides the sky.
fn terrain_point(relative_spacing: vec4<f32>, cell: vec2<f32>) -> SurfacePoint {
    let relative = relative_spacing.xyz;
    let world = relative + camera.position.xyz;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let away = length(relative);
    let ray = relative / max(away, 1.0e-6);
    let ray_dx = dpdx(ray);
    let ray_dy = dpdy(ray);
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_shading_level(length(world - terrain.eye.xyz));
    let geometric = terrain_surface_normal(local, shading);
    // How the point moves per pixel on the plane of the terrain's smooth
    // normal (decision 098): where the rays one pixel over meet that plane,
    // which change smoothly across the triangles drawn, so fading detail and
    // mipmaps change with them rather than from one triangle to the next.
    let facing = min(dot(geometric, ray), -0.015625);
    let smooth_dx = away * (ray_dx - ray * (dot(geometric, ray_dx) / facing));
    let smooth_dy = away * (ray_dy - ray * (dot(geometric, ray_dy) / facing));
    let along = vec2<f32>(length(smooth_dx), length(smooth_dy));
    let footprint = max(along.x, along.y);
    var site: TerrainSite;
    site.world = world;
    site.world_dx = smooth_dx;
    site.world_dy = smooth_dy;
    site.footprint = footprint;
    site.texel = max(min(along.x, along.y), footprint / 16.0);
    site.geometric = geometric;
    site.distance = away;
    site.toward = -ray;
    site.thinned = terrain_thinned(world, away);
    // Sunk where the ground's relief and tracks show (decisions 082 and
    // 083).
    let relief = terrain_parallax(site, shading);
    site.world = relief.world;
    let ground = terrain_ground(site, shading);
    let broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, footprint);
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    let view = u32(terrain.origin.w);
    var albedo = ground.albedo;
    // The normals and materials views show the terrain's own normals.
    var normal = ground.normal;
    var occlusion = ground.occlusion;
    if relief.deform > 0.0 {
        // Tracks (decision 083) tilt the ground along their walls and rims,
        // and darken it where they press it down.
        let track = terrain_deform_track(site.world.xz);
        let slope = terrain_deform_slope(site.world.xz)
            * terrain_track_slope_share(track, relief.deform);
        // Walls steeper than the map's texels resolve (narrower tracks than
        // a few texels) tilt as steeply as about fifty degrees, not flat on
        // their sides.
        let steep = length(slope);
        let tilt = slope * min(1.0, 1.2 / max(steep, 1.0e-6));
        normal = stable_normalize(normal - vec3<f32>(tilt.x, 0.0, tilt.y));
        let pressed = saturate(-terrain_track_height(track, relief.deform) / 0.06);
        albedo *= 1.0 - 0.15 * pressed;
        occlusion *= 1.0 - 0.25 * pressed;
    }
    if view == 2u || view == 4u {
        normal = geometric;
    }
    if view < 4u {
        // Plain colors vary at every scale; maps carry their own detail
        // and vary broadly.
        let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
        let mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
        albedo *= mix(plain, mapped, ground.mapped);
    }
    let sky_map = terrain_sky_map(local);
    let sky = sky_map.r;
    if view < 4u {
        // Its large-scale variety (decision 090), as on the blades growing
        // on it.
        albedo *= ground_variety(sky_map.gb, world.xz, ground.vegetated);
    }
    albedo = terrain_view_albedo(albedo, normal, relative_spacing.w, cell);
    if view == 6u {
        // The detail view (decision 090): green where all maps show, yellow
        // where their color alone does, red where their means stand in,
        // blue where the ground shows as its cover's canopy.
        let mean = 1.0 - ground.colored;
        albedo = vec3<f32>(0.1, 0.65, 0.15) * ground.full
            + vec3<f32>(0.85, 0.75, 0.1) * (ground.colored - ground.full)
            + vec3<f32>(0.75, 0.12, 0.1) * mean;
        albedo = mix(albedo, vec3<f32>(0.15, 0.3, 0.9), saturate(ground.canopy));
    }
    let roughness = max(ground.roughness, 0.03);
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = occlusion * sky;
    point.bounce = (1.0 - sky) * albedo * terrain_bounce();
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
    if view == 5u {
        // The sky view: how much sky the ground sees, as unlit gray, cubed
        // so the grade leaves differences visible.
        point.albedo = vec3<f32>(0.0);
        point.diffuse_color = vec3<f32>(0.0);
        point.f0 = vec3<f32>(0.0);
        point.bounce = vec3<f32>(0.0);
        point.emissive = vec3<f32>(0.6 * sky * sky * sky);
    }
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
