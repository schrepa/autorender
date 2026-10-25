// Terrain in the mesh passes, after `terrain_vertex.wgsl` and
// `terrain_surface.wgsl`: vertex stages drawing its subdivision's leaves
// into the frame's targets, deep materials displacing them, the ground's
// surface and the forward fragment stages.

// Each layer's fades (`terrain_layer_fades`); from `terrain.surface.y`, each
// layer's word naming where its tile's hole bits start (all ones without
// holes); and the hole bits of the tiles with holes, each tile's cells
// starting at samples 0 to 256 row by row, 32 to a word.
@group(1) @binding(20) var<storage, read> terrain_holes: array<u32>;

// Pipelines of terrains with holes discard their hole cells; the others
// never test.
override TERRAIN_HOLES: bool = false;

// Every resident tile's normal maps, in the same layers as its heights: the
// smooth surface's gradient at each sample, two halves of a float
// (`terrain_normals.wgsl`), gathered through the control values' sampler.
@group(1) @binding(21) var terrain_normals: texture_2d_array<u32>;
// The sky maps' sampler (`meshes/terrain_sky.wgsl`).
@group(1) @binding(22) var terrain_sky_sampler: sampler;
// Each tile's control values, in the same layers as its
// heights: per sample, its two strongest surface materials (bits 0 to 7
// and 8 to 15), the second's share (16 to 23) and how wet its ground is
// (24 to 31).
@group(1) @binding(24) var terrain_control: texture_2d_array<u32>;
// Gathers a 2x2 block of control values, or of normal maps' texels, at
// once.
@group(1) @binding(32) var terrain_control_sampler: sampler;

// A surface material: its linear color (its base color factor with maps)
// and roughness; its tiling in meters, height contrast, projection and its
// maps' layer plus one (0 without maps); its normal scale, occlusion
// strength, metalness and mean occlusion; and its maps' mean color and
// roughness, its factors applied, which stand in for them where a whole
// repeat spans less than a pixel, as their mipmaps' last level does there
// (`terrain_mapped`).
struct TerrainPaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    // The meters its heights span where the ground's relief shows (none
    // where it displaces the ground), the meters tracks sink it at most, the
    // mean square slope its normal maps' mean holds, all their slopes
    // averaged (`spread_slopes`), which roughens it where their means stand
    // in, and the meters its heights displace the ground
    // (`terrain_displacement`).
    relief: vec4<f32>,
    // Its cover's canopy: its blades' tip color (linear), and their
    // leaves' area per square meter of ground (0 without cover).
    canopy: vec4<f32>,
    // Its placement layers' objects, which the ground takes on where they
    // retire: their mean linear color and the share of the ground they
    // cover where it lies alone (w, 0 without any); the base-2 logarithm
    // of the bounding spheres' diameters (meters) of those ranked first
    // (`placed_size.x`) and last (y), the largest and smallest, as layers
    // rank them by size; and how they thin with distance: the share
    // standing far away (`placed_thinning.x`), and the levels whose tiles
    // hold them near and far (y and z), each plus one, 0 where those stay
    // resident everywhere.
    placed: vec4<f32>,
    placed_size: vec4<f32>,
    placed_thinning: vec4<f32>,
    // Its height map's mean (x), which height blending takes where its maps'
    // means stand in.
    heights: vec4<f32>,
}
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const TERRAIN_PALETTE_ENTRIES: u32 = 102u;
// Every terrain's surface materials, each terrain's from
// `terrain.directory.z`.
@group(1) @binding(25) var<uniform> terrain_palette: array<TerrainPaletteEntry, TERRAIN_PALETTE_ENTRIES>;
// Surface materials' maps: base color, a layer per material with maps;
// data, two layers per material: tangent-space normal (green up the image,
// toward -z), then occlusion, roughness, metalness and height; the runtime
// virtual texture's sheets, three layers each; and the maps' sampler.
@group(1) @binding(26) var terrain_albedo: texture_2d_array<f32>;
@group(1) @binding(27) var terrain_material_data: texture_2d_array<f32>;
@group(1) @binding(28) var terrain_vt_sheets: texture_2d_array<f32>;
@group(1) @binding(29) var terrain_material_sampler: sampler;
// Runtime deformation's map (`deformation.wgsl`), in its
// reference units: each block's deepest track or highest rim, widened by a
// block; each texel's track depth and rim height; and each texel's slopes.
@group(1) @binding(31) var<storage, read> terrain_deformation: array<u32>;
// Each terrain's macro colors with their mips, a layer each from
// `terrain.macro_map.x` less one, over its extent as its sky map lies: the
// tint its bakes give its ground, halved (red, green and blue), and how
// lush plants grow there (alpha).
@group(1) @binding(35) var terrain_macro: texture_2d_array<f32>;

// The terrain's macro colors at terrain-local `local`, filtered over
// `footprint` meters (a pixel's, or a page's texel's), once they arrived;
// before, ground as it is (`macro_tint`).
fn terrain_macro_texel(local: vec2<f32>, footprint: f32) -> vec4<f32> {
    let layer = terrain.macro_map.x;
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    let size = vec2<f32>(textureDimensions(terrain_macro));
    let step = max(extent.x / size.x, extent.y / size.y);
    let lod = max(log2(max(footprint, 1.0e-6) / step), 0.0);
    // Held within the texels of the coarser level read, which the
    // materials' sampler would otherwise wrap past the terrain's edges.
    let edge = 0.5 * exp2(ceil(lod)) / size;
    let uv = clamp(local / extent, edge, vec2<f32>(1.0) - edge);
    return textureSampleLevel(terrain_macro, terrain_material_sampler, uv, layer - 1u, lod);
}

// The tint the terrain's detail and macro colors give `ground` at `world`,
// filtered over `footprint` meters (a pixel's, or a page's texel's): plain
// colors vary at every scale, maps carry their own detail and vary
// broadly, and the macro colors tint it as they do the blades growing on
// it, the same wherever the eye stands. Pages bake it in; pixels computing
// their ground apply it.
fn terrain_tint(world: vec3<f32>, ground: TerrainGround, footprint: f32) -> vec3<f32> {
    let broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, footprint);
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
    let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
    let mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
    let colors = terrain_macro_texel(world.xz - terrain.origin.xz, footprint);
    return mix(plain, mapped, ground.mapped) * macro_tint(colors, ground.vegetated);
}

struct TerrainOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Relative to the view's eye; w: the meters between its leaf's meshlet
    // vertices.
    @location(1) relative: vec4<f32>,
    // Where the point lies on its true surface, in level-0 samples from the
    // terrain's origin, which hole cells are looked up by.
    @location(2) cell: vec2<f32>,
}

struct TerrainTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) relative: vec4<f32>,
    @location(2) cell: vec2<f32>,
    @location(4) previous: vec3<f32>,
}

// A vertex of the terrain's subdivision as every pass drawing the camera's
// ground places it: relative to the terrain's eye, terrain-local where it
// stands on the true surface, and the meters between its leaf's meshlet
// vertices.
struct TerrainSurfaceVertex {
    relative: vec3<f32>,
    local: vec2<f32>,
    spacing: f32,
}

// Lattice vertex `vertex` of leaf `leaf` of this frame's subdivision
// (`surface_vertex`): on the ground's true surface, moved along its smooth
// normal where deep materials displace it (`terrain_displacement`). Every
// quantity it reads is a function of where it stands, so leaves sharing a
// side meet without cracks and each of the meshlet's pieces meeting at the
// point draws it alike.
fn terrain_surface_vertex(leaf: u32, vertex: u32) -> TerrainSurfaceVertex {
    let triangle = surface_leaf(leaf);
    var point: TerrainSurfaceVertex;
    point.local = surface_vertex(triangle, vertex);
    point.relative = surface_relative(point.local, terrain_true_height(point.local));
    point.relative += terrain_displacement(point.relative + terrain.eye.xyz, point.local);
    point.spacing = length(triangle.c - triangle.a) / f32(1u << SURFACE_MESHLET_LEVEL);
    return point;
}

fn terrain_output(leaf: u32, vertex: u32) -> TerrainOutput {
    let point = terrain_surface_vertex(leaf, vertex);
    let relative = point.relative + terrain_view.eye_offset.xyz;
    var output: TerrainOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = vec4<f32>(relative, point.spacing);
    output.cell = point.local / terrain.anchor.w;
    return output;
}

// Leaves no fragment in hole cells: at `cell` (level-0 samples from the
// terrain's origin), the cell of the finest resident level's tile there,
// whose triangles the true surface lies on, where that tile's hole bits
// mark it.
fn terrain_holes_test(cell: vec2<f32>) {
    if !TERRAIN_HOLES {
        return;
    }
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(cell / f32(1u << index), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        let holes = terrain_holes[terrain.surface.y + layer];
        if holes == 0xffffffffu {
            return;
        }
        let at = vec2<u32>(clamp(floor(samples - vec2<f32>(tile * 256u)), vec2<f32>(0.0),
            vec2<f32>(256.0)));
        let bit = at.y * 257u + at.x;
        if (terrain_holes[holes + bit / 32u] & (1u << (bit % 32u))) != 0u {
            discard;
        }
        return;
    }
}

@vertex
fn vs_terrain(@builtin(instance_index) leaf: u32, @builtin(vertex_index) vertex: u32)
        -> TerrainOutput {
    return terrain_output(leaf, vertex);
}

// Terrain stands still: its motion is the camera's.
@vertex
fn vs_terrain_temporal(@builtin(instance_index) leaf: u32,
        @builtin(vertex_index) vertex: u32) -> TerrainTemporalOutput {
    let output = terrain_output(leaf, vertex);
    var temporal: TerrainTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.cell = output.cell;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative.xyz), 1.0)).xyw;
    return temporal;
}

// Value noise of the material, 0 to 1.
fn terrain_value(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    // The four corners' hashes: the next cell's products are the cell's
    // plus each multiplier (wrapping integers), one product per axis.
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let a = terrain_hash_of(x0 ^ y0);
    let b = terrain_hash_of(x1 ^ y0);
    let c = terrain_hash_of(x0 ^ y1);
    let d = terrain_hash_of(x1 ^ y1);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The hash of a lattice cell, 0 to 1, from its coordinates' products, x
// times 0x8da6b343 and y times 0xd8163841 (wrapping), mixed.
fn terrain_hash_of(mixed: u32) -> f32 {
    var h = mixed;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

// Two values of the hash `terrain_hash_of` mixes from `mixed`, each 0 to 1:
// its high and low halves.
fn terrain_hash2(mixed: u32) -> vec2<f32> {
    var h = mixed;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return vec2<f32>(f32(h >> 16u), f32(h & 0xffffu)) * (1.0 / 65535.0);
}

// Two channels of value noise at `p` on `terrain_value`'s lattice, each 0
// to 1.
fn terrain_value2(p: vec2<f32>) -> vec2<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let q = vec2<u32>(vec2<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    return mix(mix(terrain_hash2(x0 ^ y0), terrain_hash2(x1 ^ y0), u.x),
        mix(terrain_hash2(x0 ^ y1), terrain_hash2(x1 ^ y1), u.x), u.y);
}

// Noise of feature size `size` meters at `p`, as a pixel spanning
// `footprint` meters averages it: whole while the pixel spans half a
// feature or less, its mean once it spans a whole one. A noise computed
// where it shows has no mipmaps to filter it, and a pixel spanning its
// features would otherwise show one of them at random.
fn terrain_detail(p: vec2<f32>, size: f32, footprint: f32) -> f32 {
    let kept = saturate(2.0 - 2.0 * footprint / size);
    // Averaged away entirely, it is its mean.
    if kept == 0.0 {
        return 0.5;
    }
    return 0.5 + (terrain_value(p / size) - 0.5) * kept;
}

// How much of the next coarser level's normals (x) and surface materials
// (y) still show over layer `layer`'s tile: fading in after
// it arrived, and wholly the coarser level's materials until its own
// arrive. The first words of `terrain_holes` hold them, two layers a word.
fn terrain_layer_fades(layer: u32) -> vec2<f32> {
    let half = (terrain_holes[layer / 2u] >> ((layer % 2u) * 16u)) & 0xffffu;
    return vec2<f32>(f32(half & 0xffu), f32(half >> 8u)) * (1.0 / 255.0);
}

// The smooth surface's gradient (rise per meter along x and z) at
// terrain-local `local` in `level`'s normal maps, or in the nearest coarser
// level's where its tile is not resident: the gradients at the four samples
// around it blended bilinearly, as `engine_terrain::Tile::sample` blends
// them; a terrain's coarser levels' are level 0's averaged. A tile fading
// in, or shading with its shading before that arrived, shows the next
// resident coarser level's under it; ground no level holds is level.
fn terrain_level_gradient(local: vec2<f32>, level: u32) -> vec2<f32> {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var gradient = vec2<f32>(0.0);
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
        // Samples 0 to 256 within the tile; layers keep a border texel
        // before the first.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        // The 2x2 block's gradients in one gather at its shared corner (x:
        // (0, 1), y: (1, 1), z: (1, 0), w: (0, 0)), which no subtexel
        // rounding moves off the block.
        let texels = textureGather(0, terrain_normals, terrain_control_sampler,
            (vec2<f32>(corner) + vec2<f32>(2.0)) / 259.0, layer);
        let low = mix(unpack2x16float(texels.w), unpack2x16float(texels.z), f.x);
        let high = mix(unpack2x16float(texels.x), unpack2x16float(texels.y), f.x);
        let fade = terrain_layer_fades(layer).x;
        gradient += mix(low, high, f.y) * (left * (1.0 - fade));
        left *= fade;
        if left <= 0.0 {
            return gradient;
        }
    }
    return gradient;
}

// Which level shades ground where a pixel spans `footprint` meters of it:
// the finest whose samples span at least half `terrain.pixel_error.x`
// pixels there, whole while they span that many and turning into the next
// coarser as they shrink to half as many. At a view facing the ground that
// is the second half of each level's load reach, where its geometry's
// samples stand from the pixel error to half of it apart and its tiles
// still load; seen obliquely, where its samples crowd on screen. Spacings
// double level by level, so detail fades continuously with what a pixel
// spans, as trilinear filtering turns between mipmaps, and the ground's
// light and materials hold still as levels of detail hand over, whatever
// level the geometry draws at. Lookups fall back to coarser levels where a
// level's tiles are not resident.
struct TerrainShadingLevel {
    level: u32,
    blend: f32,
}

fn terrain_shading_level(footprint: f32) -> TerrainShadingLevel {
    let levels = terrain.directory.y;
    // The pixel error over the pixels level 0's samples span there: level
    // L's is this over 2^L.
    let over = terrain.pixel_error.x * footprint / terrain.anchor.w;
    var shading: TerrainShadingLevel;
    shading.level = u32(clamp(ceil(log2(max(over, 1.0e-6))) - 1.0, 0.0, f32(levels - 1u)));
    shading.blend = select(0.0, saturate(over / f32(1u << shading.level) - 1.0),
        shading.level + 1u < levels);
    return shading;
}

// The levels shading ground at `world` as a pixel seen from the detail eye
// shows it: by the meters it spans across the view there
// (`detail_across`). Seen obliquely, a level's samples crowd along the view
// but still span as many pixels across it, where they show, so no view of
// the ground hands them over sooner; and the levels' tiles load by the same
// distance, so it never asks for a finer level than is loaded.
fn terrain_pixel_shading(world: vec3<f32>) -> TerrainShadingLevel {
    return terrain_shading_level(detail_across(world));
}

// Texels of a level of the shadow clipmap a sample of the ground it holds
// spans at least (`gpu_layouts::constants::GROUND_TEXELS`).
const KEPT_GROUND_TEXELS: f32 = 2.0;

// The meters the ground of four levels of the shadow clipmap stands above
// the ground's point `world`, terrain-local `local` (`scene.wgsl`'s
// `kept_lift`), from the level a pixel's size picks there, which lookups
// read first: each level holds the ground of the finest pyramid level whose
// samples span at least KEPT_GROUND_TEXELS of its texels
// (`shadow_fit::ground_level`), drawn whole, so the point looks the level
// up from its height on that ground, where the level's ground meets itself
// whatever level draws it here.
fn terrain_kept_lift(world: vec3<f32>, local: vec2<f32>) -> KeptLift {
    var lift: KeptLift;
    lift.first = shadow_level(world).level;
    let count = u32(shadow.level_state.x);
    let levels = terrain.directory.y;
    for (var index = 0u; index < 4u && lift.first + index < count; index++) {
        let texel = shadow.windows[lift.first + index].w;
        var level = 0u;
        while level + 1u < levels
            && terrain.anchor.w * f32(1u << level) < KEPT_GROUND_TEXELS * texel {
            level++;
        }
        let held = terrain_level_height(local, level);
        if held.found {
            lift.lifts[index] = held.height + terrain.origin.y - world.y;
        }
    }
    return lift;
}

// The normal of the terrain's smooth surface at terrain-local `local` at
// `shading`'s levels, their gradients blended.
fn terrain_surface_normal(local: vec2<f32>, shading: TerrainShadingLevel) -> vec3<f32> {
    var gradient = terrain_level_gradient(local, shading.level);
    if shading.blend > 0.0 {
        gradient = mix(gradient, terrain_level_gradient(local, shading.level + 1u),
            shading.blend);
    }
    return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
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

// Adds `weight` shared between control value `control`'s two materials by
// its blend. Adding nothing changes no weight and inserts no material.
fn terrain_control_blend(blend: ptr<function, TerrainBlend>, control: u32, weight: f32) {
    let share = f32((control >> 16u) & 0xffu) * (1.0 / 255.0);
    let first = weight * (1.0 - share);
    if first != 0.0 {
        terrain_blend_add(blend, control & 0xffu, first);
    }
    let second = weight * share;
    if second != 0.0 {
        terrain_blend_add(blend, (control >> 8u) & 0xffu, second);
    }
}

// Whether this terrain's coarser levels shade with level 0's shading
// averaged (`terrain.steep.w`): every terrain's but surroundings'.
fn terrain_averaged() -> bool {
    return terrain.steep.w > 0.5;
}

// Adds `weight` of the materials at terrain-local `local` in `level`'s own
// control values, the four samples around it, bilinearly: as much as its
// tile shows of its own, a tile fading in, or without its own materials yet,
// showing the next coarser level's. Returns the weight it leaves to the
// coarser levels: all of it where `level`'s tile is not resident.
fn terrain_level_own(blend: ptr<function, TerrainBlend>, local: vec2<f32>, level: u32,
        weight: f32) -> f32 {
    let header = terrain.directory.x + level * 4u;
    let samples = max(local / (terrain.anchor.w * f32(1u << level)), vec2<f32>(0.0));
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
    let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
    if layer == 0xffffffffu {
        return weight;
    }
    let fade = terrain_layer_fades(layer).y;
    if fade < 1.0 {
        // Samples 0 to 256 within the tile; layers keep a border texel
        // before the first.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let own = weight * (1.0 - fade);
        // The 2x2 block's four control values in one gather at its
        // shared corner (x: (0, 1), y: (1, 1), z: (1, 0), w: (0, 0)),
        // which no subtexel rounding moves off the block; layers are 259
        // samples a side.
        let controls = textureGather(0, terrain_control, terrain_control_sampler,
            (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer);
        let weights = vec4<f32>(own * (1.0 - f.x) * (1.0 - f.y), own * f.x * (1.0 - f.y),
            own * (1.0 - f.x) * f.y, own * f.x * f.y);
        if weights.x != 0.0 {
            terrain_control_blend(blend, controls.w, weights.x);
        }
        if weights.y != 0.0 {
            terrain_control_blend(blend, controls.z, weights.y);
        }
        if weights.z != 0.0 {
            terrain_control_blend(blend, controls.x, weights.z);
        }
        if weights.w != 0.0 {
            terrain_control_blend(blend, controls.y, weights.w);
        }
    }
    return weight * fade;
}

// Adds `weight` of the materials at terrain-local `local` in `level`'s
// control values, or the nearest coarser level's where its tile is not
// resident: the four samples around it, bilinearly. A tile fading in, or
// without its own materials yet, shows the next resident coarser level's
// under it; ground no level holds is the first material.
fn terrain_level_blend(blend: ptr<function, TerrainBlend>, local: vec2<f32>, level: u32,
        weight: f32) {
    let levels = terrain.directory.y;
    var left = weight;
    for (var index = level; index < levels; index++) {
        left = terrain_level_own(blend, local, index, left);
        if left <= 0.0 {
            return;
        }
    }
    terrain_blend_add(blend, 0u, left);
}

// Paints the terrain's steep material (`terrain.steep`) over `blend` on
// ground of smooth normal `normal`, decided at this point of the ground as
// the samples decide it at theirs: none up to the first tangent of its
// slopes, all of it from the second, smoothly between, the other materials
// sharing what it leaves as they share the rest. Ground whose samples hold
// the steep material alone keeps it. Where coarser levels hold level 0's
// shading averaged, it paints level 0's part of the blend alone
// (`terrain_blend`): their averages hold it as level 0's samples did.
fn terrain_steep_blend(blend: ptr<function, TerrainBlend>, normal: vec3<f32>) {
    let steep = terrain.steep;
    if steep.x <= 0.0 {
        return;
    }
    let material = u32(steep.x) - 1u;
    let share = smoothstep(steep.y, steep.z, length(normal.xz) / max(normal.y, 1.0e-4));
    let weights = (*blend).weights;
    let others = select(weights, vec4<f32>(0.0), (*blend).ids == vec4<u32>(material));
    let rest = dot(others, vec4<f32>(1.0));
    if rest <= 0.0 {
        return;
    }
    let total = dot(weights, vec4<f32>(1.0));
    (*blend).weights = others * ((1.0 - share) * total / rest);
    if share > 0.0 {
        terrain_blend_add(blend, material, share * total);
    }
}

// How far noise carries the borders between materials: each material's
// weight is scaled by e raised to this times the noise's turn along a
// direction of its own, which moves a border between two most of the way
// across its cell of samples.
const TERRAIN_BORDER_TURN: f32 = 4.0;

// The noise moving the borders between the materials of `level`'s control
// values at terrain-local `local` (two channels, 0 to 1): features about
// two and a half of the level's samples across, on a lattice turned off
// theirs so that neither lines up with the other.
fn terrain_border_noise(local: vec2<f32>, level: u32) -> vec2<f32> {
    let turned = vec2<f32>(0.8 * local.x - 0.6 * local.y, 0.6 * local.x + 0.8 * local.y);
    return terrain_value2(turned / (2.5 * terrain.anchor.w * f32(1u << level))
        + vec2<f32>(11.3, -6.7));
}

// Moves the borders between `blend`'s materials at terrain-local `local` by
// noise at `shading`'s levels: each material's weight is scaled by e to
// the noise's turn along a direction of its own, then all of them back to
// their sum, so ground of one material keeps it while borders wander
// across their cells instead of following the samples' grid.
fn terrain_border_blend(blend: ptr<function, TerrainBlend>, local: vec2<f32>,
        shading: TerrainShadingLevel) {
    let weights = (*blend).weights;
    // Ground of one material has no border to move.
    let present = select(vec4<f32>(0.0), vec4<f32>(1.0), weights > vec4<f32>(0.0));
    if dot(present, vec4<f32>(1.0)) < 2.0 {
        return;
    }
    var noise = terrain_border_noise(local, shading.level);
    if shading.blend > 0.0 {
        noise = mix(noise, terrain_border_noise(local, shading.level + 1u), shading.blend);
    }
    let turn = noise - vec2<f32>(0.5);
    var moved = weights;
    for (var slot = 0u; slot < 4u; slot++) {
        if weights[slot] > 0.0 {
            let direction = terrain_hash2((*blend).ids[slot] * 0x9e3779b9u) * 2.0 - 1.0;
            moved[slot] = weights[slot] * exp(TERRAIN_BORDER_TURN * dot(turn, direction));
        }
    }
    (*blend).weights = moved * (dot(weights, vec4<f32>(1.0)) / dot(moved, vec4<f32>(1.0)));
}

// The materials of the ground at terrain-local `local`, of smooth normal
// `normal`, at `shading`'s levels: each level's control values at the four
// samples around it, bilinearly; the steep material decided again at this
// point, over the whole blend on surroundings and, where coarser levels
// hold level 0's averaged, over level 0's own part before the coarser
// levels' are added, theirs holding it as averaged; and the borders
// between them moved by noise.
fn terrain_blend(local: vec2<f32>, normal: vec3<f32>, shading: TerrainShadingLevel)
        -> TerrainBlend {
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    let averaged = terrain_averaged();
    var near = 1.0 - shading.blend;
    var level = shading.level;
    if averaged && level == 0u {
        // Level 0's samples decided the steep material by the same rule.
        near = terrain_level_own(&blend, local, 0u, near);
        terrain_steep_blend(&blend, normal);
        level = 1u;
    }
    if near > 0.0 {
        terrain_level_blend(&blend, local, level, near);
    }
    if shading.blend > 0.0 {
        terrain_level_blend(&blend, local, shading.level + 1u, shading.blend);
    }
    if !averaged {
        terrain_steep_blend(&blend, normal);
    }
    terrain_border_blend(&blend, local, shading);
    return blend;
}

// The meters apart a level's own samples stand at most for its wetness to
// show whole: wet banks a few meters wide drawn between samples further
// apart alias into scallops, so surroundings' coarser levels, which give
// their samples their own wetness, fade theirs out as their samples
// spread. A terrain's coarser levels hold level 0's wetness averaged,
// which never scallops, and show it whole.
const TERRAIN_WET_SPACING: f32 = 2.0;

// How wet the ground is at terrain-local `local` in `level`'s control
// values, or the nearest coarser level's where its tile is not resident,
// from 0, dry, to 1: the four samples' wetness around it (their top bytes),
// bilinearly, as `terrain_level_blend` blends their materials; on
// surroundings fading out on levels whose samples stand further apart than
// TERRAIN_WET_SPACING. A tile fading in, or without its own materials yet,
// shows the next resident coarser level's under it; ground no level holds
// is dry.
fn terrain_level_wetness(local: vec2<f32>, level: u32) -> f32 {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    let averaged = terrain_averaged();
    var wet = 0.0;
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
        let fade = terrain_layer_fades(layer).y;
        if fade < 1.0 {
            let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
            let corner = min(vec2<u32>(within), vec2<u32>(255u));
            let f = within - vec2<f32>(corner);
            let texel = corner + vec2<u32>(1u);
            // The block's control values gathered as `terrain_level_own`
            // gathers them (x: (0, 1), y: (1, 1), z: (1, 0), w: (0, 0)).
            let controls = textureGather(0, terrain_control, terrain_control_sampler,
                (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer);
            let wets = vec4<f32>(controls >> vec4<u32>(24u)) * (1.0 / 255.0);
            let low = mix(wets.w, wets.z, f.x);
            let high = mix(wets.x, wets.y, f.x);
            let spacing = terrain.anchor.w * f32(1u << index);
            let kept = select(saturate(TERRAIN_WET_SPACING / spacing), 1.0, averaged);
            wet += mix(low, high, f.y) * (left * (1.0 - fade)) * kept;
        }
        left *= fade;
        if left <= 0.0 {
            return wet;
        }
    }
    return wet;
}

// How wet the ground is at terrain-local `local` at `shading`'s levels,
// their wetness blended.
fn terrain_wetness(local: vec2<f32>, shading: TerrainShadingLevel) -> f32 {
    var wet = terrain_level_wetness(local, shading.level);
    if shading.blend > 0.0 {
        wet = mix(wet, terrain_level_wetness(local, shading.level + 1u), shading.blend);
    }
    return wet;
}

// Wet ground (Lagarde 2013, "Water drop 3b: physically based wet
// surfaces"): ground rougher than `TERRAIN_POROUS` is porous, wholly so
// `TERRAIN_POROUS_RANGE` rougher still, and water filling its pores darkens
// it to `TERRAIN_SOAKED` of its albedo.
const TERRAIN_POROUS: f32 = 0.5;
const TERRAIN_POROUS_RANGE: f32 = 0.4;
const TERRAIN_SOAKED: f32 = 0.2;

// Ground of `roughness` as `wet` (0 to 1) wets it: the factor its albedo
// darkens by, the more the more porous it is, and its roughness, glossed by
// the water filming it.
fn terrain_wet_ground(roughness: f32, wet: f32) -> vec2<f32> {
    let gloss = 1.0 - roughness;
    let porosity = saturate(((1.0 - gloss) - TERRAIN_POROUS) / TERRAIN_POROUS_RANGE);
    let factor = mix(1.0, TERRAIN_SOAKED, porosity);
    return vec2<f32>(mix(1.0, factor, wet), 1.0 - mix(1.0, gloss, mix(1.0, factor, 0.5 * wet)));
}

// What a terrain fragment's ground shows: its linear albedo, roughness,
// occlusion and metalness, its normal, and the share of it that has maps;
// for its macro colors and detail, the shares of it growing cover and
// showing as canopy; and the shares showing its maps' detail, filtered over
// the pixel, rather than their means, and of their normal maps' slopes that
// filtering averages into its roughness.
struct TerrainGround {
    albedo: vec3<f32>,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    metalness: f32,
    mapped: f32,
    vegetated: f32,
    canopy: f32,
    detailed: f32,
    averaged: f32,
    // The cover's tip colors and leaves, each weighted by its material's
    // share, which the runtime virtual texture keeps.
    canopy_color: vec3<f32>,
    leaves: f32,
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

// The most anisotropy surface materials' maps filter to
// (`gpu_layouts::constants::MATERIAL_ANISOTROPY`, their sampler's clamp):
// past it the filtering under-samples a footprint's long side, and crests
// seen almost edge on ripple.
const TERRAIN_ANISOTROPY: f32 = 16.0;

// Derivative `short` of a map's coordinates widened, beside `long`, to at
// least `least`: along its own way, or across `long` where it has none.
fn terrain_widened(short: vec2<f32>, long: vec2<f32>, least: f32) -> vec2<f32> {
    let span = length(short);
    if span >= least {
        return short;
    }
    if span > 0.0 {
        return short * (least / span);
    }
    return vec2<f32>(-long.y, long.x) * (least / max(length(long), 1.0e-20));
}

// A map's coordinates' derivatives `dx` and `dy` (columns), the shorter
// widened to at least a TERRAIN_ANISOTROPY-th of the longer, so its
// filtering resolves the footprint whole, as a site's `texel` measures it.
fn terrain_filtered(dx: vec2<f32>, dy: vec2<f32>) -> mat2x2<f32> {
    let least = max(length(dx), length(dy)) / TERRAIN_ANISOTROPY;
    return mat2x2<f32>(terrain_widened(dx, dy, least), terrain_widened(dy, dx, least));
}

// Layer `layer` of `map` at `world` (with its screen derivatives) on axis
// `axis`'s plane in lattice `lattice`, as `mapping` lays it, filtered over
// no more anisotropy than the maps' filtering takes (`terrain_filtered`).
fn terrain_plane_texel(map: texture_2d_array<f32>, layer: u32, mapping: TerrainMapping,
        world: vec3<f32>, world_dx: vec3<f32>, world_dy: vec3<f32>, axis: u32, lattice: u32)
        -> vec4<f32> {
    let shift = vec2<f32>(0.37, 0.71) * f32(lattice);
    let gradients = terrain_filtered(
        terrain_lattice(world_plane(world_dx, axis) * mapping.scale, axis, lattice),
        terrain_lattice(world_plane(world_dy, axis) * mapping.scale, axis, lattice));
    return textureSampleGrad(map, terrain_material_sampler,
        terrain_lattice(world_plane(world, axis) * mapping.scale, axis, lattice) + shift,
        layer, gradients[0], gradients[1]);
}

// Layer `layer` of `map` at `world` (with its screen derivatives) as
// `mapping` lays it in `lattice`, mixed over its planes.
fn terrain_map(map: texture_2d_array<f32>, layer: u32, mapping: TerrainMapping,
        world: vec3<f32>, world_dx: vec3<f32>, world_dy: vec3<f32>, lattice: u32) -> vec4<f32> {
    var texel = vec4<f32>(0.0);
    if mapping.planes.x > 0.0 {
        texel += mapping.planes.x
            * terrain_plane_texel(map, layer, mapping, world, world_dx, world_dy, 0u, lattice);
    }
    if mapping.planes.y > 0.0 {
        texel += mapping.planes.y
            * terrain_plane_texel(map, layer, mapping, world, world_dx, world_dy, 1u, lattice);
    }
    if mapping.planes.z > 0.0 {
        texel += mapping.planes.z
            * terrain_plane_texel(map, layer, mapping, world, world_dx, world_dy, 2u, lattice);
    }
    return texel;
}

// The normal of a material's normal map on axis `axis`'s plane in
// `lattice` at `world` on the terrain's `normal`, deflections scaled by
// `scale`: the plane's tangent-space normal filtered over the pixel (x
// along its first coordinate, y up its image, z out), turned back from the
// lattice, added to the terrain's normal in that plane (whiteout
// blending); and (w) the mean square slope the filtering averaged away,
// which the filtered normal's shortening gives (`spread_slopes`), scaled
// as its deflections are. The mipmaps hold their texels' mean normals
// unnormalized, so the shortening holds every slope a pixel spans.
fn terrain_plane_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, axis: u32, lattice: u32, normal: vec3<f32>, scale: f32)
        -> vec4<f32> {
    let filtered = terrain_plane_texel(terrain_material_data, 2u * mapping.layer, mapping, world,
        world_dx, world_dy, axis, lattice).xyz * 2.0 - 1.0;
    let shortened = length(filtered);
    let lost = spread_slopes(shortened) * scale * scale;
    let sampled = filtered / max(shortened, 1.0e-4);
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
        return vec4<f32>(stable_normalize(vec3<f32>(abs(n.z) * normal.x, n.y + normal.y,
            n.x + normal.z)), lost);
    }
    if axis == 2u {
        // Along +x, up +y, out along z.
        return vec4<f32>(stable_normalize(vec3<f32>(n.x + normal.x, n.y + normal.y,
            abs(n.z) * normal.z)), lost);
    }
    // Along +x, up -z, out along y.
    return vec4<f32>(stable_normalize(vec3<f32>(n.x + normal.x, abs(n.z) * normal.y,
        normal.z - n.y)), lost);
}

// The normal of a material's normal maps in `lattice` at `world` on the
// terrain's `normal`, deflections scaled by `scale`, and the mean square
// slope their filtering averaged away (w), mixed over its planes.
fn terrain_mapped_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, lattice: u32, normal: vec3<f32>, scale: f32) -> vec4<f32> {
    var mixed = vec4<f32>(0.0);
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
// cliffs as well as across the ground. Wherever its maps show
// (`terrain_mapped`) a patch spans more than three of the footprints their
// filtering resolves.
fn terrain_second_lattice(world: vec3<f32>, tiling: f32) -> f32 {
    let p = (world.xz + world.y * vec2<f32>(0.71, -0.43)) / (tiling * 3.1);
    return smoothstep(0.4, 0.6, terrain_value(p + vec2<f32>(7.3, -2.9)));
}

// Whether material `entry` shows its maps at `site`: while it has maps and
// a whole repeat spans more than the footprint their filtering resolves
// (`site.texel`). Past that their mipmaps' last level is all the filtering
// reads, their means the palette holds, which stand in without reading
// them; nearer every map reads through its mipmaps and anisotropic
// filtering by the pixel's own derivatives, at every distance alike.
fn terrain_mapped(entry: TerrainPaletteEntry, site: TerrainSite) -> bool {
    return entry.params.w > 0.0 && site.texel < entry.params.x;
}

// How far under the highest raised weight materials still show where their
// heights blend them, sharing the ground by how far they reach into the
// band.
const TERRAIN_HEIGHT_BAND: f32 = 0.08;

// The share of `second` (0 to 1) after the two lattices' heights `first`
// and `raised` meet as materials do, as sharply as `contrast` asks.
fn terrain_lattice_share(second: f32, first: f32, raised: f32, contrast: f32) -> f32 {
    let a = 1.0 - second + contrast * first;
    let b = second + contrast * raised;
    let top = max(a, b) - TERRAIN_HEIGHT_BAND;
    let lifted = max(vec2<f32>(a, b) - top, vec2<f32>(0.0));
    return mix(second, lifted.y / max(lifted.x + lifted.y, 1.0e-6), saturate(contrast));
}

// Where a fragment is and how it is seen, as the ground's passes read it:
// its world position and how that changes per pixel on the terrain's smooth
// surface, which its maps filter by; a pixel's footprint there seen from
// the detail eye (meters, its long side, along the view: `detail_span`),
// which noise computed where it shows averages over, and the footprint the
// maps' anisotropic filtering resolves (its short side, across the view,
// never under a sixteenth of the long one), against which a whole repeat of
// a material's maps shows them or their means (`terrain_mapped`); the
// terrain's normal and the way toward the eye; the share of its cover's
// leaves blades no longer show there, and the meters a pixel spans across
// the view there as level of detail measures objects (`terrain_across`),
// by which placed objects retire. The virtual texture's bakes, whose pages
// pixels of any size across the view read, hold neither: no canopy and no
// retired objects.
struct TerrainSite {
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    footprint: f32,
    texel: f32,
    geometric: vec3<f32>,
    toward: vec3<f32>,
    thinned: f32,
    across: f32,
}

// The share of the cover's leaves blades no longer show at `world` by their
// law (`terrain.cover`, `terrain_variety.wgsl`'s `cover_share`), from the
// detail eye as the blades are: all of them where a terrain's surroundings
// show its cover as canopy from the eye on; none without cover.
fn terrain_thinned(world: vec3<f32>) -> f32 {
    let cover = terrain.cover;
    if cover.z <= 0.0 {
        return 0.0;
    }
    if cover.x <= 0.0 {
        return 1.0;
    }
    let away = detail_distance(world);
    return 1.0 - cover_shown(away, cover_share(away, cover.x, cover.y), cover.x);
}

// The meters a pixel spans across the view at `world` as level of detail
// measures objects standing there, which placed objects retire by
// (`terrain_variety.wgsl`'s `seen_share`, as `scatter_cull.wgsl` culls
// them): `detail_across`, the same whichever way the view turns.
fn terrain_across(world: vec3<f32>) -> f32 {
    return detail_across(world);
}

// The normal blades standing on ground of normal `ground` light with, on
// average (`meshes/cover.wgsl`): leaning most of the way up
// from their sides toward the sun, which face it by two over pi on
// average.
fn terrain_blade_normal(ground: vec3<f32>) -> vec3<f32> {
    let across = vec3<f32>(lighting.sun.x, 0.0, lighting.sun.z);
    let side = across * (0.6366 / max(length(across), 1.0e-4));
    return stable_normalize(mix(side, ground, 0.6));
}

// How many times its leaves' area a view at `site` crosses through a
// cover whose leaves' area is `leaves` per square meter of ground, the
// density's setting applied: G = 0.5 of it across its way, as leaves facing
// every way show, over the sine of its angle to the ground.
fn terrain_canopy_depth(leaves: f32, site: TerrainSite) -> f32 {
    let sine = max(abs(dot(site.toward, site.geometric)), 0.03);
    return 0.5 * leaves * terrain.cover.z / sine;
}

// How much of the ground at `site` a cover whose leaves' area is `leaves` per
// square meter shows as its canopy: as much of the leaves as its blades no
// longer show by their law (`terrain.cover`) hides of the ground along the
// view, 1 - exp(-depth), all of it at grazing views.
fn terrain_canopy(leaves: f32, site: TerrainSite) -> f32 {
    if site.thinned <= 0.0 || leaves <= 0.0 {
        return 0.0;
    }
    return site.thinned * (1.0 - exp(-terrain_canopy_depth(leaves, site)));
}

// The share of their color blades show seen from `site` on average: those
// seen from their side away from the sun show COVER_FAR_SIDE of it, and of
// blades facing every way, as many are as the angle between the view's way
// and the sun's across the ground is of a half turn: none with the sun
// behind the eye, all looking toward it, half seen from straight above.
fn terrain_canopy_through(site: TerrainSite) -> f32 {
    let lengths = length(site.toward.xz) * length(lighting.sun.xz);
    var through = 0.5;
    if lengths > 1.0e-4 {
        through = acos(clamp(dot(site.toward.xz, lighting.sun.xz) / lengths, -1.0, 1.0)) / PI;
    }
    return mix(1.0, COVER_FAR_SIDE, through);
}

// What the canopy of a cover's blades shows at `site`, their tips' color
// `tip` and their leaves' area `leaves` per square meter, standing on
// ground of color `root`, which their roots take: the colors and the light
// of the blades a view meets there on average (`terrain_variety.wgsl`'s
// `cover_seen`), their far sides dimmer (`terrain_canopy_through`).
struct TerrainCanopyLook {
    color: vec3<f32>,
    occlusion: f32,
}

fn terrain_canopy_look(root: vec3<f32>, tip: vec3<f32>, leaves: f32, site: TerrainSite)
        -> TerrainCanopyLook {
    let seen = cover_seen(terrain_canopy_depth(leaves, site));
    var look: TerrainCanopyLook;
    look.color = mix(root, tip, seen.tip) * terrain_canopy_through(site);
    look.occlusion = seen.occlusion;
    return look;
}

// The meters from the detail eye placed objects on the tiles of the level
// coded `code` (`placed_thinning`: the level plus one, 0 where its tiles
// stay resident everywhere) stand within, as culling thins them: the
// level's reach in the directory, or 0.
fn terrain_placed_reach(code: f32) -> f32 {
    if code < 0.5 {
        return 0.0;
    }
    let header = terrain.directory.x + (u32(code + 0.5) - 1u) * 4u;
    return bitcast<f32>(terrain_directory[header + 3u]);
}

// The share of the ground of material `entry` at `site` that shows its
// placement layers' objects' mean color: of the ground they cover, the
// share their objects leave as `scatter_cull.wgsl` culls them, seen from the
// detail eye, whether thinning hides them with distance
// (`terrain_variety.wgsl`'s `placed_thinned`) or they span too few pixels
// across `site.across` for their ranks (`seen_share`). Both keep the objects
// ranked first, the largest, whose diameters' logarithms fall evenly from
// the first's to the last's, each covering the ground as its diameter's
// square; none where the site measures nothing.
fn terrain_placed_share(entry: TerrainPaletteEntry, site: TerrainSite) -> f32 {
    let covered = entry.placed.w;
    if covered <= 0.0 || site.across <= 0.0 {
        return 0.0;
    }
    let first = entry.placed_size.x;
    let octaves = max(first - entry.placed_size.y, 0.0);
    // Objects ranked under this still span enough pixels to show: where a
    // rank meets the share its diameter's pixels keep.
    let seen = saturate((1.0 + first - log2(site.across)) / (2.0 + octaves));
    let thinning = entry.placed_thinning;
    let thin = vec4<f32>(thinning.x, terrain_placed_reach(thinning.y),
        terrain_placed_reach(thinning.z), 1.0);
    let kept = min(placed_thinned(thin, detail_distance(site.world)), seen);
    // The share of their ground the objects ranked under `kept` cover.
    var shown = kept;
    if octaves > 1.0e-3 {
        shown = (1.0 - exp2(-2.0 * octaves * kept)) / (1.0 - exp2(-2.0 * octaves));
    }
    return covered * (1.0 - shown);
}

// One material of a fragment's blend after its first pass: whether it
// shows its maps (`terrain_mapped`), its occlusion, roughness, metalness
// and height texel filtered over the pixel, and the share of its second
// lattice; its weight raised by its height (its texel's where it shows its
// maps, its heights' mean where that stands in) as sharply as its height
// contrast asks, and its weight times that contrast.
struct TerrainSlot {
    mapped: bool,
    texel: vec4<f32>,
    second: f32,
    raised: f32,
    contrast: f32,
}

// The first pass of material `material` at weight `weight`: its heights
// where it shows its maps, their mean where that stands in, and its height
// contrast, so materials without maps blend by their weights alone.
fn terrain_slot(material: u32, weight: f32, site: TerrainSite) -> TerrainSlot {
    var slot: TerrainSlot;
    slot.texel = vec4<f32>(1.0, 1.0, 1.0, 0.5);
    slot.raised = -1.0e9;
    if weight <= 0.0 {
        return slot;
    }
    let entry = terrain_material(material);
    let sharpness = select(0.0, entry.params.y, entry.params.w > 0.0);
    var height = entry.heights.x;
    slot.mapped = terrain_mapped(entry, site);
    if slot.mapped {
        let mapping = terrain_mapping(entry, site.geometric);
        let pattern = terrain_second_lattice(site.world, entry.params.x);
        var first = vec4<f32>(0.0);
        var other = vec4<f32>(0.0);
        if pattern < 1.0 {
            first = terrain_map(terrain_material_data, 2u * mapping.layer + 1u, mapping,
                site.world, site.world_dx, site.world_dy, 0u);
        }
        if pattern > 0.0 {
            other = terrain_map(terrain_material_data, 2u * mapping.layer + 1u, mapping,
                site.world, site.world_dx, site.world_dy, 1u);
        }
        slot.second = pattern;
        if pattern > 0.0 && pattern < 1.0 {
            slot.second = terrain_lattice_share(pattern, first.a, other.a, sharpness);
        }
        slot.texel = mix(first, other, slot.second);
        height = slot.texel.a;
    }
    slot.raised = weight + sharpness * height;
    slot.contrast = weight * sharpness;
    return slot;
}

// Adds material `material`'s share `weight` of the ground to `ground` and
// its normal to `normal`, from its first pass `slot`: where it shows its
// maps, their color, roughness, occlusion, metalness and normals filtered
// over the pixel, the slopes its normal maps' filtering averages away
// roughening it instead (`terrain_plane_normal`), so its highlights widen
// as the relief it no longer resolves would widen them while its light and
// shade read the filtered normal; their means where those stand in, the
// slopes their mean averages away roughening it alike; or its plain color.
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
    var lost = entry.relief.z;
    if slot.mapped {
        let mapping = terrain_mapping(entry, site.geometric);
        let scale = entry.maps.x;
        var albedo = vec3<f32>(0.0);
        var mapped = vec4<f32>(0.0);
        if slot.second < 1.0 {
            albedo += (1.0 - slot.second) * terrain_map(terrain_albedo, mapping.layer, mapping,
                site.world, site.world_dx, site.world_dy, 0u).rgb;
            if scale > 0.0 {
                mapped += (1.0 - slot.second) * terrain_mapped_normal(mapping, site.world,
                    site.world_dx, site.world_dy, 0u, site.geometric, scale);
            }
        }
        if slot.second > 0.0 {
            albedo += slot.second * terrain_map(terrain_albedo, mapping.layer, mapping,
                site.world, site.world_dx, site.world_dy, 1u).rgb;
            if scale > 0.0 {
                mapped += slot.second * terrain_mapped_normal(mapping, site.world,
                    site.world_dx, site.world_dy, 1u, site.geometric, scale);
            }
        }
        color = entry.color.rgb * albedo;
        roughness = entry.color.w * slot.texel.g;
        occlusion = 1.0 + entry.maps.y * (slot.texel.r - 1.0);
        metalness *= slot.texel.b;
        lost = mapped.w;
        if scale > 0.0 {
            turned = stable_normalize(mapped.xyz);
        }
        (*ground).detailed += weight;
        if entry.relief.z > 0.0 {
            (*ground).averaged += weight * saturate(lost / entry.relief.z);
        }
    }
    if lost > 0.0 {
        let alpha = roughness * roughness;
        roughness = sqrt(sqrt(min(alpha * alpha + lost, 1.0)));
    }
    if entry.params.w > 0.0 {
        (*ground).mapped += weight;
    }
    if entry.canopy.w > 0.0 {
        // Ground under cover shows as its canopy where blades retire: the
        // colors a view meets among them on average, rough and shaded
        // among them as they are, and lit as blades are, their sides
        // toward the sun and the sun passing through them (their leaf
        // share, `terrain_point_seen`).
        let canopy = terrain_canopy(entry.canopy.w, site);
        if canopy > 0.0 {
            let look = terrain_canopy_look(color, entry.canopy.rgb, entry.canopy.w, site);
            color = mix(color, look.color, canopy);
            roughness = mix(roughness, COVER_ROUGHNESS, canopy);
            // The sky they reflect too: a flat ground seen along it would
            // mirror much more of it than blades facing the eye do.
            occlusion = mix(occlusion, look.occlusion, canopy);
            turned = mix(turned, terrain_blade_normal(site.geometric), canopy);
        }
        (*ground).vegetated += weight;
        (*ground).canopy += weight * canopy;
        (*ground).canopy_color += weight * entry.canopy.rgb;
        (*ground).leaves += weight * entry.canopy.w;
    }
    // Where its placement layers' objects retire, the ground takes on their
    // mean color over what they covered, standing over its canopy as they
    // stand over the blades.
    let placed = terrain_placed_share(entry, site);
    if placed > 0.0 {
        color = mix(color, entry.placed.rgb, placed);
    }
    (*ground).albedo += weight * color;
    (*ground).roughness += weight * roughness;
    (*ground).occlusion += weight * occlusion;
    (*ground).metalness += weight * metalness;
    *normal += weight * turned;
}

// Up to four surface materials at `site` at `shading`'s levels
// (`terrain_blend`), their slots, and the shares the ground shows them in.
struct TerrainShares {
    ids: vec4<u32>,
    weights: vec4<f32>,
    first: TerrainSlot,
    second: TerrainSlot,
    third: TerrainSlot,
    fourth: TerrainSlot,
}

// The ground at `site` at `shading`'s levels: up to four materials, each
// raised by its height map, filtered over the pixel, as sharply as its
// height contrast asks, so stones and tufts stand through what they meet
// before it covers them.
fn terrain_ground(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainGround {
    return terrain_shared_ground(terrain_shares(site, shading), site);
}

fn terrain_shares(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainShares {
    let blend = terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, shading);
    let first = terrain_slot(blend.ids.x, blend.weights.x, site);
    let second = terrain_slot(blend.ids.y, blend.weights.y, site);
    let third = terrain_slot(blend.ids.z, blend.weights.z, site);
    let fourth = terrain_slot(blend.ids.w, blend.weights.w, site);
    // Height blending: materials within TERRAIN_HEIGHT_BAND of the highest
    // show, turned toward the plain shares as contrast falls to 0.
    let raised = vec4<f32>(first.raised, second.raised, third.raised, fourth.raised);
    let contrast = vec4<f32>(first.contrast, second.contrast, third.contrast, fourth.contrast);
    let highest = max(max(raised.x, raised.y), max(raised.z, raised.w));
    let lifted = max(raised - vec4<f32>(highest - TERRAIN_HEIGHT_BAND), vec4<f32>(0.0));
    let shares = lifted / max(dot(lifted, vec4<f32>(1.0)), 1.0e-6);
    let weights = mix(blend.weights, shares,
        saturate(contrast.x + contrast.y + contrast.z + contrast.w));
    return TerrainShares(blend.ids, weights, first, second, third, fourth);
}

// The ground the materials of `shares` make at `site`.
fn terrain_shared_ground(shares: TerrainShares, site: TerrainSite) -> TerrainGround {
    var ground: TerrainGround;
    var normal = vec3<f32>(0.0);
    terrain_slot_ground(&ground, &normal, shares.ids.x, shares.weights.x, shares.first, site);
    terrain_slot_ground(&ground, &normal, shares.ids.y, shares.weights.y, shares.second, site);
    terrain_slot_ground(&ground, &normal, shares.ids.z, shares.weights.z, shares.third, site);
    terrain_slot_ground(&ground, &normal, shares.ids.w, shares.weights.w, shares.fourth, site);
    ground.normal = stable_normalize(normal);
    return ground;
}

// One material's part in the ground's relief: the meters it
// sinks the ground at most, already weighted (none for materials
// displacing it, whose palette entries hold no relief), and where its
// maps' heights lie: as its maps lie there (from above, or across their
// planes on steep ground), read with the point's change per pixel.
struct TerrainReliefLayer {
    depth: f32,
    mapping: TerrainMapping,
    dx: vec3<f32>,
    dy: vec3<f32>,
}

fn terrain_relief_layer(material: u32, weight: f32, site: TerrainSite) -> TerrainReliefLayer {
    var layer: TerrainReliefLayer;
    layer.depth = 0.0;
    if weight <= 0.0 || material == 0xffffffffu {
        return layer;
    }
    let entry = terrain_material(material);
    // Materials showing their maps; their means stand level.
    if !terrain_mapped(entry, site) {
        return layer;
    }
    layer.mapping = terrain_mapping(entry, site.geometric);
    layer.dx = site.world_dx;
    layer.dy = site.world_dy;
    layer.depth = entry.relief.x * weight;
    return layer;
}

// How far below the ground's top `layer` sinks it at `point`: its heights
// there mixed over its planes.
fn terrain_relief_sink(layer: TerrainReliefLayer, point: vec3<f32>) -> f32 {
    if layer.depth <= 0.0 {
        return 0.0;
    }
    let scale = layer.mapping.scale;
    var height = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = layer.mapping.planes[axis];
        if share > 0.0 {
            height += share * textureSampleGrad(terrain_material_data, terrain_material_sampler,
                world_plane(point, axis) * scale, 2u * layer.mapping.layer + 1u,
                world_plane(layer.dx, axis) * scale, world_plane(layer.dy, axis) * scale).a;
        }
    }
    return layer.depth * (1.0 - height);
}

// The meters a material displacing the ground `depth` deep moves it along
// its smooth normal at `world`, where a pixel seen from the detail eye spans
// `across` meters, its maps lying as `mapping` lays them: its heights less
// their mean (their coarsest mip), so the plain surface far ground draws is
// the displaced one's average. Its heights are read no finer than a pixel,
// nor than the subdivision's finest meshlets can carry
// (`terrain.relief.w`); it shows whole while its depth spans
// TERRAIN_RELIEF_WHOLE pixels or more and none by TERRAIN_RELIEF_GONE, as
// relief does, so far ground is the plain surface its normal maps shade.
fn terrain_material_displacement(depth: f32, mapping: TerrainMapping, world: vec3<f32>,
        across: f32) -> f32 {
    let shown = smoothstep(TERRAIN_RELIEF_GONE, TERRAIN_RELIEF_WHOLE, depth / across);
    if shown <= 0.0 {
        return 0.0;
    }
    let layer = 2u * mapping.layer + 1u;
    let texels = f32(textureDimensions(terrain_material_data).x) * mapping.scale;
    let lod = max(log2(max(across, terrain.relief.w) * texels), 0.0);
    var height = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = mapping.planes[axis];
        if share > 0.0 {
            height += share * textureSampleLevel(terrain_material_data, terrain_material_sampler,
                world_plane(world, axis) * mapping.scale, layer, lod).a;
        }
    }
    let mean = textureSampleLevel(terrain_material_data, terrain_material_sampler,
        vec2<f32>(0.5), layer, f32(textureNumLevels(terrain_material_data) - 1u)).a;
    return depth * (height - mean) * shown;
}

// How far deep materials displace the ground's true surface at `world`,
// terrain-local `local`: along its smooth normal at the levels a pixel seen
// from the detail eye shades it at, each material of its blend by its share
// (`terrain_material_displacement`). None where the deepest of the
// terrain's materials could not move it a pixel (`terrain.relief.z`), which
// then reads no materials. A function of where the point stands alone, so
// every pass drawing the ground, and the subdivision measuring it, moves it
// alike.
fn terrain_displacement(world: vec3<f32>, local: vec2<f32>) -> vec3<f32> {
    let across = max(detail_across(world), 1.0e-6);
    if terrain.relief.z < TERRAIN_RELIEF_GONE * across {
        return vec3<f32>(0.0);
    }
    let shading = terrain_pixel_shading(world);
    let normal = terrain_surface_normal(local, shading);
    let blend = terrain_blend(local, normal, shading);
    var moved = 0.0;
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = blend.weights[slot];
        if weight <= 0.0 {
            continue;
        }
        let entry = terrain_material(blend.ids[slot]);
        if entry.relief.w > 0.0 && entry.params.w > 0.0 {
            moved += weight * terrain_material_displacement(entry.relief.w,
                terrain_mapping(entry, normal), world, across);
        }
    }
    return normal * moved;
}

// World texel `texel`'s slot in the deformation map, whose texels wrap.
fn terrain_deform_slot(texel: vec2<i32>) -> vec2<i32> {
    let size = terrain.deform_grid.z;
    // A power of two wraps by its low bits; any other by one remainder.
    if size > 0 && (size & (size - 1)) == 0 {
        return texel & vec2<i32>(size - 1);
    }
    if size > 0 {
        let rest = texel % size;
        return select(rest, rest + size, rest < vec2<i32>(0));
    }
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

// Where a fragment's view ray meets the ground: `world` moved across the
// ground to where it shades, and `deform` the meters its materials let
// tracks sink it at most where the deformation map holds tracks nearby
// (otherwise 0).
struct TerrainRelief {
    world: vec3<f32>,
    deform: f32,
}

// How far below the ground's top the relief layers `a` and `b` and the
// tracks of the ground's `deform` (faded by `fade`) sink it at `point`.
fn terrain_sink(a: TerrainReliefLayer, b: TerrainReliefLayer, point: vec3<f32>, deform: f32,
        fade: f32) -> f32 {
    var sink = terrain_relief_sink(a, point) + terrain_relief_sink(b, point);
    if deform > 0.0 {
        sink -= terrain_track_height(terrain_deform_track(point.xz), deform) * fade;
    }
    return sink;
}

// The share of its maps' repeat a relief ray crosses at most between two
// steps: a sixty-fourth, about the heights' own features.
const TERRAIN_RELIEF_STRIDE: f32 = 1.0 / 64.0;

// The meters a relief layer's maps repeat over; no limit for a layer
// that sinks nothing.
fn terrain_relief_repeat(layer: TerrainReliefLayer) -> f32 {
    if layer.depth <= 0.0 {
        return 1.0e6;
    }
    return 1.0 / layer.mapping.scale;
}

// The pixels on screen the relief's deepest point moves where the relief
// shows whole, and where it has flattened away: as other detail, it hands
// over once it moves the ground less than a pixel. Displacement hands over
// by its depth on screen alike.
const TERRAIN_RELIEF_WHOLE: f32 = 2.0;
const TERRAIN_RELIEF_GONE: f32 = 1.0;
// The secant steps that narrow where the ray meets the relief between its
// last two steps, so no step's layer shows where the ray skims it.
const TERRAIN_RELIEF_REFINE: u32 = 2u;

// How far across a ray along `way` (toward the ground) travels per meter
// of depth into ground of normal `normal`, at most three meters, rays
// meeting it at a cosine under DETAIL_EDGE_ON counting as edge on.
fn terrain_relief_across(way: vec3<f32>, normal: vec3<f32>) -> f32 {
    let ray = way / max(length(way), 1.0e-6);
    let travel = ray / max(-dot(ray, normal), DETAIL_EDGE_ON);
    return min(length(travel.xz), 3.0);
}

// Where a fragment's view ray meets the ground's relief: the ray marches
// down through the heights of its two heaviest materials, each as deep as
// its relief reaches (a few centimeters at most: deeper materials displace
// the ground instead), and through the tracks deformers pressed, from the
// top of their rims to the bottom of the deepest; the heights lie on the
// ground's surface, from above or across their planes on steep ground as
// their maps lie (`terrain_level_height`), so the ray only moves where on
// that surface the ground shades from, as the virtual texture's pages hold
// it. The relief shows by its size on screen, seen from the detail eye,
// not by its distance or the angle it is seen at: whole while its deepest
// point moves TERRAIN_RELIEF_WHOLE pixels or more, flattening to none by
// TERRAIN_RELIEF_GONE, with about a step per pixel it moves, so far relief
// costs little and no ring or band follows the eye; seen at a slant, it
// stands as deep as head on, as real ground's bumps do.
fn terrain_parallax(site: TerrainSite, shading: TerrainShadingLevel, dither: f32)
        -> TerrainRelief {
    var relief: TerrainRelief;
    relief.world = site.world;
    relief.deform = 0.0;
    let to_point = site.world - camera.position.xyz;
    let ray = to_point / max(length(to_point), 1.0e-6);
    // Depth runs into the ground along its normal; rays meeting it edge on
    // (DETAIL_EDGE_ON) see it flatten, where the drawn surface's own
    // silhouette cuts through it.
    let into = -dot(ray, site.geometric);
    let grazing = smoothstep(0.0, DETAIL_EDGE_ON, into);
    // The pixels a meter of depth moves the point on screen at most, seen
    // from the detail eye: it moves along the view, where a pixel spans its
    // longer side.
    let pixels = terrain_relief_across(site.world - camera.detail.xyz, site.geometric)
        / detail_span(site.world, site.geometric).y;
    let tracked = terrain_deform_reach(site.world.xz);
    // Without tracks, ground whose deepest relief could not move a pixel
    // looks up no materials.
    let shows = grazing > 0.0 && terrain.relief.x * pixels > TERRAIN_RELIEF_GONE;
    if !shows && tracked < 5.0e-4 {
        return relief;
    }
    let blend = terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, shading);
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
    if grazing <= 0.0 {
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
    var a = terrain_relief_layer(blend.ids[heaviest], weights[heaviest], site);
    var b = terrain_relief_layer(blend.ids[next], weights[next], site);
    // Tracks as deep and rims as high as the map's nearby tracks reach and
    // the ground's materials let them.
    let deform = relief.deform;
    let sunk_whole = min(tracked, deform);
    let raised_whole = min(tracked, 0.5 * deform);
    // Its depth shows by how far its deepest point would move on screen.
    let moved = (a.depth + b.depth + sunk_whole + raised_whole) * pixels;
    let fade = smoothstep(TERRAIN_RELIEF_GONE, TERRAIN_RELIEF_WHOLE, moved) * grazing;
    if fade <= 0.0 {
        return relief;
    }
    a.depth *= fade;
    b.depth *= fade;
    let sunk = sunk_whole * fade;
    let raised = raised_whole * fade;
    let top = -raised;
    let depth = a.depth + b.depth + sunk;
    // Down the ray a step of the relief's depth at a time, the ray's
    // travel across per meter of depth bounded; from above the rims, the
    // ray starting before the ground it meets.
    let travel = ray / max(into, DETAIL_EDGE_ON);
    let across = length(travel.xz);
    let along = travel.xz * min(1.0, 3.0 / max(across, 1.0e-6));
    // The surface point the ray stands over, and its move per meter of
    // depth: along the surface's tangent from the point under the fragment,
    // whose height maps lying across planes read.
    let rise = -site.geometric.xz / max(site.geometric.y, 0.05);
    let march = vec3<f32>(along.x, dot(rise, along), along.y);
    var start = site.world;
    if (a.depth > 0.0 && a.mapping.planes.y < 1.0) || (b.depth > 0.0 && b.mapping.planes.y < 1.0) {
        let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
        if held.found {
            start.y = held.height + terrain.origin.y;
        }
    }
    // Enough steps that each moves the point at most a pixel on screen and
    // across at most TERRAIN_RELIEF_STRIDE of its maps' repeat, so the ray
    // never steps over what it should meet, from two up to the most steps;
    // a step every two centimeters of tracks, up to twice that. A count
    // between two whole ones takes either by `dither`, which temporally
    // antialiased frames vary by pixel and frame, so no seam shows where
    // the count changes.
    let crossed = (depth - top) * length(along);
    let stride = TERRAIN_RELIEF_STRIDE * min(terrain_relief_repeat(a), terrain_relief_repeat(b));
    let wanted = max((depth - top) * pixels, crossed / max(stride, 1.0e-4));
    let count = max(clamp(wanted, 2.0, terrain.relief.y),
        min((sunk + raised) / 0.02, 2.0 * terrain.relief.y));
    let steps = max(u32(floor(count + dither)), 2u);
    var previous = top;
    var gap = top - terrain_sink(a, b, start + march * top, deform, fade);
    var hit = top;
    if gap < 0.0 {
        hit = depth;
        for (var step = 1u; step <= steps; step++) {
            let at = top + (depth - top) * f32(step) / f32(steps);
            let point = start + march * at;
            let next_gap = at - terrain_sink(a, b, point, deform, fade);
            if next_gap >= 0.0 {
                // Between the last step and this one, where the ray reaches
                // the ground, narrowed by secant steps (TERRAIN_RELIEF_REFINE).
                var low = vec2<f32>(previous, gap);
                var high = vec2<f32>(at, next_gap);
                for (var refine = 0u; refine < TERRAIN_RELIEF_REFINE; refine++) {
                    let middle = mix(low.x, high.x, low.y / min(low.y - high.y, -1.0e-6));
                    let middle_gap = middle
                        - terrain_sink(a, b, start + march * middle, deform, fade);
                    if middle_gap >= 0.0 {
                        high = vec2<f32>(middle, middle_gap);
                    } else {
                        low = vec2<f32>(middle, middle_gap);
                    }
                }
                hit = mix(low.x, high.x, low.y / min(low.y - high.y, -1.0e-6));
                break;
            }
            previous = at;
            gap = next_gap;
        }
    }
    relief.world = site.world + march * hit;
    return relief;
}

// The surface a terrain fragment `relative` to the eye shows: its surface
// materials, tinted by its macro colors and varied across the ground so
// their maps do not repeat evenly, lit by the sky it sees, from where it
// sees it, and by the ground around where that hides the sky.
fn terrain_point(relative: vec3<f32>) -> SurfacePoint {
    let point = terrain_point_seen(relative, terrain_forward_look(relative));
    return widened(point, dpdx(point.normal), dpdy(point.normal));
}

// How a terrain fragment's point and view ray move per pixel: the point on
// its triangle (`raw`: known on each axis; otherwise its smooth surface's
// stands in) and the ray.
struct TerrainLook {
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    ray_dx: vec3<f32>,
    ray_dy: vec3<f32>,
    raw: vec2<bool>,
    // Whether the runtime virtual texture may stand in for
    // the ground, and the threshold it picks between its two mips and the
    // computed ground by, negative to blend them (`terrain_vt_texels`).
    textured: bool,
    dither: f32,
}

// How a fragment of its own triangle `relative` to the eye looks: from
// the triangle, without the virtual texture.
fn terrain_forward_look(relative: vec3<f32>) -> TerrainLook {
    let world = eye_world(relative);
    let ray = relative / max(length(relative), 1.0e-6);
    var look: TerrainLook;
    look.world_dx = dpdx(world);
    look.world_dy = dpdy(world);
    look.ray_dx = dpdx(ray);
    look.ray_dy = dpdy(ray);
    look.raw = vec2<bool>(true);
    look.textured = false;
    look.dither = 0.5;
    return look;
}

// Where the view ray of a fragment `relative` to the eye, seen as `look`
// says, meets the ground: its site, on the true surface where deep
// materials displaced the ground drawn there and moved across the ground
// where its relief and tracks show, the shading levels its pixel's
// footprint gives and the meters tracks may sink it (`TerrainRelief`).
struct TerrainSeen {
    site: TerrainSite,
    shading: TerrainShadingLevel,
    deform: f32,
}

fn terrain_seen(relative: vec3<f32>, look: TerrainLook) -> TerrainSeen {
    // The true surface's point the drawn one stands for: a step back along
    // the displacement there, so materials lie where their heights moved the
    // ground.
    let drawn = eye_world(relative);
    let world = drawn - terrain_displacement(drawn, drawn.xz - terrain.origin.xz);
    let away = length(relative);
    let ray = relative / max(away, 1.0e-6);
    let ray_dx = look.ray_dx;
    let ray_dy = look.ray_dy;
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_pixel_shading(world);
    let geometric = terrain_surface_normal(local, shading);
    // How the point moves per pixel on the plane of the terrain's smooth
    // normal, which its maps filter by: where the rays one pixel over meet
    // that plane, which change smoothly across the triangles drawn, so
    // mipmaps change with them rather than from one triangle to the next.
    let facing = min(dot(geometric, ray), -0.015625);
    // Its materials as finely as the display shows them: per display pixel
    // (the render size's share of a pixel), as the mip bias samples.
    let detail = material_detail();
    let smooth_dx = away * detail * (ray_dx - ray * (dot(geometric, ray_dx) / facing));
    let smooth_dy = away * detail * (ray_dy - ray * (dot(geometric, ray_dy) / facing));
    // What a pixel spans there seen from the detail eye: along the view, its
    // long side, and across it, which its maps' filtering resolves.
    let span = detail_span(world, geometric);
    var site: TerrainSite;
    site.world = world;
    site.world_dx = smooth_dx;
    site.world_dy = smooth_dy;
    site.footprint = span.y;
    site.texel = span.x;
    site.geometric = geometric;
    site.toward = -ray;
    site.thinned = terrain_thinned(world);
    site.across = terrain_across(world);
    // Moved where the ground's relief and tracks show.
    let relief = terrain_parallax(site, shading, look.dither);
    site.world = relief.world;
    return TerrainSeen(site, shading, relief.deform);
}

// `terrain_point` before its highlights widen with its normal's turn.
fn terrain_point_seen(relative: vec3<f32>, look: TerrainLook) -> SurfacePoint {
    let seen = terrain_seen(relative, look);
    let site = seen.site;
    let world = eye_world(relative);
    let local = world.xz - terrain.origin.xz;
    let geometric = site.geometric;
    // The shadow clipmap's levels' ground over this point, which it looks
    // them up from.
    kept_lift = terrain_kept_lift(world, local);
    // The runtime virtual texture's ground where its pages show, their
    // detail and macro colors baked in; computed here elsewhere and under
    // pages still fading in, its detail and macro colors applied.
    var page: TerrainGround;
    var shown = 0.0;
    if look.textured {
        let texels = terrain_vt_texels(site, look.dither);
        shown = texels.shown;
        if shown > 0.0 {
            page = terrain_vt_ground(texels, site);
        }
    }
    var ground = page;
    if shown < 1.0 {
        // As the pages bake it: the materials, their blend and the tint on
        // the ground's surface where the view meets its relief.
        var surface = site;
        let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
        if held.found {
            surface.world.y = held.height + terrain.origin.y;
        }
        ground = terrain_ground(surface, seen.shading);
        ground.albedo *= terrain_tint(surface.world, ground, site.footprint);
        if shown > 0.0 {
            ground = terrain_vt_shown(ground, page, shown);
        }
    }
    var albedo = ground.albedo;
    var normal = ground.normal;
    var occlusion = ground.occlusion;
    if seen.deform > 0.0 {
        // Tracks tilt the ground along their walls and rims,
        // and darken it where they press it down.
        let track = terrain_deform_track(site.world.xz);
        let slope = terrain_deform_slope(site.world.xz)
            * terrain_track_slope_share(track, seen.deform);
        // Walls steeper than the map's texels resolve (narrower tracks than
        // a few texels) tilt as steeply as about fifty degrees, not flat on
        // their sides.
        let steep = length(slope);
        let tilt = slope * min(1.0, 1.2 / max(steep, 1.0e-6));
        normal = stable_normalize(normal - vec3<f32>(tilt.x, 0.0, tilt.y));
        let pressed = saturate(-terrain_track_height(track, seen.deform) / 0.06);
        albedo *= 1.0 - 0.15 * pressed;
        occlusion *= 1.0 - 0.25 * pressed;
    }
    // Wet where the surface's rules wet it, darker and glossier, over the
    // virtual texture's texels too, which hold it dry.
    var roughness = ground.roughness;
    let wet = terrain_wetness(local, seen.shading);
    if wet > 0.0 {
        let soaked = terrain_wet_ground(roughness, wet);
        albedo *= soaked.x;
        roughness = soaked.y;
    }
    roughness = max(roughness, 0.03);
    let sky_map = terrain_sky_map(local);
    let sky = sky_map.r;
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = occlusion * sky;
    // The light the ground around sends where it hides the sky, as the
    // terrain's probes gather it; none where no probes relight.
    point.bounce = bounce_light(world, normal, true).light * occlusion;
    point.sky_turn = sky_map.gba * 2.0 - 1.0;
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
    // The point moves per render pixel, as the site's maps do per display
    // pixel.
    let detail = material_detail();
    point.world_dx = select(site.world_dx / detail, look.world_dx, look.raw.x);
    point.world_dy = select(site.world_dy / detail, look.world_dy, look.raw.y);
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = true;
    // Its cover's canopy is leaf, as the blades it stands for are.
    point.leaf = ground.canopy;
    return point;
}

// Deferred shading: terrain's visibility pass writes the
// terrain and depth of each pixel it covers, then one pass per terrain
// shades the pixels holding it, once each: forward, the 2x2 quads its small
// triangles cut shaded half again as many fragments.
// Each pixel's terrain id and its depth's bits.
@group(2) @binding(18) var terrain_ids_texture: texture_2d<u32>;

// The id of this fragment's terrain in the visibility pass.
@fragment
fn fs_terrain_ids(input: TerrainOutput) -> @location(0) vec2<u32> {
    terrain_holes_test(input.cell);
    return vec2<u32>(terrain.directory.x + 1u, bitcast<u32>(input.clip_position.z));
}

// The eye-relative point at framebuffer position `pixel` and depth `depth`.
fn terrain_relative_at(pixel: vec2<f32>, depth: f32, size: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(pixel.x * 2.0 / size.x - 1.0, 1.0 - pixel.y * 2.0 / size.y);
    let point = terrain_view.inverse_relative_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w;
}

// The view ray through framebuffer position `pixel`: toward its point on the
// near plane (depth 1, reversed).
fn terrain_ray_at(pixel: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let relative = terrain_relative_at(pixel, 1.0, size);
    return relative / max(length(relative), 1.0e-30);
}

struct TerrainResolved {
    point: SurfacePoint,
    previous: vec3<f32>,
}

// The terrain point a resolve pass shades at `position`: discarded where
// another surface or terrain shows. Its point and normal move per pixel as
// its neighbors on the same surface show; the view ray as the quad's.
// `dither` picks between the runtime virtual texture's mips and the
// computed ground, or blends them where negative, as frames without
// temporal antialiasing do (`terrain_vt_texels`).
fn terrain_resolved(position: vec4<f32>, dither: f32) -> TerrainResolved {
    let pixel = vec2<i32>(position.xy);
    // A pixel of the frame's render size, which it draws at.
    let size = camera.region.xy;
    let texel = textureLoad(terrain_ids_texture, pixel, 0);
    let mine = texel.x == terrain.directory.x + 1u;
    if !mine {
        discard;
    }
    let depth = bitcast<f32>(texel.y);
    let relative = terrain_relative_at(position.xy, depth, size);
    let away = length(relative);
    // Neighbors on the same surface: the same terrain, not much nearer or
    // farther than this pixel's distance changes along it.
    let here = select(0.0, 1.0, mine);
    let reach = 0.05 * away + 0.5;
    let others = vec2<f32>(dpdxFine(here), dpdyFine(here));
    let farther = vec2<f32>(dpdxFine(away), dpdyFine(away));
    let same = (others == vec2<f32>(0.0)) & (abs(farther) < vec2<f32>(reach));
    // The quad's rays change by its first row's and column's differences;
    // discarded lanes still help with them.
    let ray = terrain_ray_at(position.xy, size);
    var look: TerrainLook;
    look.world_dx = dpdxFine(relative);
    look.world_dy = dpdyFine(relative);
    look.ray_dx = dpdxCoarse(ray);
    look.ray_dy = dpdyCoarse(ray);
    look.raw = same;
    look.textured = true;
    look.dither = dither;
    let seen = terrain_point_seen(relative, look);
    var resolved: TerrainResolved;
    resolved.point = widened(seen, select(vec3<f32>(0.0), dpdxFine(seen.normal), same.x),
        select(vec3<f32>(0.0), dpdyFine(seen.normal), same.y));
    resolved.previous = (camera.previous_view_projection
        * vec4<f32>(eye_world(relative), 1.0)).xyw;
    return resolved;
}

@fragment
fn fs_terrain_resolve_post(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(finish(terrain_resolved(position, -1.0).point, vec3<f32>(0.0)),
        position.xy);
}

@fragment
fn fs_terrain_resolve_post_lit(@builtin(position) position: vec4<f32>) -> SceneTargets {
    let point = terrain_resolved(position, -1.0).point;
    return scene_targets(finish(point, local_light(point, position.xy)), position.xy);
}

@fragment
fn fs_terrain_resolve_temporal(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = terrain_resolved(position, fade_noise(position.xy));
    return temporal_targets(finish(resolved.point, vec3<f32>(0.0)), position,
        resolved.previous);
}

@fragment
fn fs_terrain_resolve_temporal_lit(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = terrain_resolved(position, fade_noise(position.xy));
    return temporal_targets(finish(resolved.point, local_light(resolved.point, position.xy)),
        position, resolved.previous);
}

fn shade_terrain(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative.xyz);
    terrain_holes_test(input.cell);
    return finish(point, vec3<f32>(0.0));
}

fn shade_terrain_lit(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative.xyz);
    terrain_holes_test(input.cell);
    return finish(point, local_light(point, input.clip_position.xy));
}

fn terrain_temporal_output(input: TerrainTemporalOutput) -> TerrainOutput {
    var output: TerrainOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.cell = input.cell;
    return output;
}

@fragment
fn fs_terrain(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_terrain(input).color, input.clip_position.xy), 1.0);
}

@fragment
fn fs_terrain_lit(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_terrain_lit(input).color, input.clip_position.xy), 1.0);
}

@fragment
fn fs_terrain_post(input: TerrainOutput) -> SceneTargets {
    return scene_targets(shade_terrain(input), input.clip_position.xy);
}

@fragment
fn fs_terrain_post_lit(input: TerrainOutput) -> SceneTargets {
    return scene_targets(shade_terrain_lit(input), input.clip_position.xy);
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

// Reflection probe captures: linear light before the grade.
@fragment
fn fs_terrain_probe(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_terrain(input).linear, 1.0);
}

@fragment
fn fs_terrain_probe_lit(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_terrain_lit(input).linear, 1.0);
}
