// Terrain in the mesh passes, after `terrain_vertex.wgsl`:
// vertex stages into the frame's targets, the ground's surface
// and the forward fragment stages.

// Hole bits of the tiles with holes, each tile's cells
// starting at samples 0 to 256 row by row, 32 to a word.
@group(1) @binding(20) var<storage, read> terrain_holes: array<u32>;

// Pipelines of patches with holes discard their hole cells; the others
// never test.
override TERRAIN_HOLES: bool = false;

// Every resident tile's normal maps, in the same layers as its heights: the
// smooth surface's gradient at each sample, two halves of a float
// (`terrain_normals.wgsl`), gathered through the control values' sampler.
@group(1) @binding(21) var terrain_normals: texture_2d_array<u32>;
// The sky maps' sampler (`meshes/terrain_sky.wgsl`).
@group(1) @binding(22) var terrain_sky_sampler: sampler;
// Each terrain's levels, finest first, from `terrain.directory.x`: four
// words each (where its tiles' entries start, its tiles along x and z, and
// the meters from the eye its tiles load within, as bits); then the entries,
// row by row, each the layer of a resident tile or all ones.
@group(1) @binding(23) var<storage, read> terrain_directory: array<u32>;
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
// roughness, its factors applied, which stand in for them where they
// repeat too small on screen to show.
struct TerrainPaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    // The meters its heights span where the ground's relief shows,
    // the meters tracks sink it at most, and the
    // variance of its normal maps' slopes at their scale, which its
    // roughness takes over where they fade.
    relief: vec4<f32>,
    // Its cover's canopy: the linear color its blades show
    // seen from afar, and their leaves' area per square meter of ground (0
    // without cover).
    canopy: vec4<f32>,
}
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const TERRAIN_PALETTE_ENTRIES: u32 = 170u;
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

// Noise of feature size `size` meters at `p`, fading to its mean where a
// pixel spans more than its features (`footprint` meters).
fn terrain_detail(p: vec2<f32>, size: f32, footprint: f32) -> f32 {
    let fade = saturate(2.0 - 2.0 * footprint / size);
    // Faded out entirely, it is its mean.
    if fade == 0.0 {
        return 0.5;
    }
    return 0.5 + (terrain_value(p / size) - 0.5) * fade;
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
// them. A tile fading in shows the next resident coarser level's under it;
// ground no level holds is level.
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

// Adds `weight` of the materials at terrain-local `local` in `level`'s
// control values, or the nearest coarser level's where its tile is not
// resident: the four samples around it, bilinearly. A tile fading in, or
// without its own materials yet, shows the next resident coarser level's
// under it.
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
        left *= fade;
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
// the steep material alone keeps it.
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
// point; and the borders between them moved by noise.
fn terrain_blend(local: vec2<f32>, normal: vec3<f32>, shading: TerrainShadingLevel)
        -> TerrainBlend {
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    terrain_level_blend(&blend, local, shading.level, 1.0 - shading.blend);
    if shading.blend > 0.0 {
        terrain_level_blend(&blend, local, shading.level + 1u, shading.blend);
    }
    terrain_steep_blend(&blend, normal);
    terrain_border_blend(&blend, local, shading);
    return blend;
}

// The meters apart a level's samples stand at most for its wetness to show
// whole: wet banks a few meters wide drawn between samples further apart
// alias into scallops, so coarser levels' fade out as their samples
// spread.
const TERRAIN_WET_SPACING: f32 = 2.0;

// How wet the ground is at terrain-local `local` in `level`'s control
// values, or the nearest coarser level's where its tile is not resident,
// from 0, dry, to 1: the four samples' wetness around it (their top bytes),
// bilinearly, as `terrain_level_blend` blends their materials, fading out
// on levels whose samples stand further apart than TERRAIN_WET_SPACING. A
// tile fading in, or without its own materials yet, shows the next
// resident coarser level's under it; ground no level holds is dry.
fn terrain_level_wetness(local: vec2<f32>, level: u32) -> f32 {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
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
            // The block's control values gathered as `terrain_level_blend`
            // gathers them (x: (0, 1), y: (1, 1), z: (1, 0), w: (0, 0)).
            let controls = textureGather(0, terrain_control, terrain_control_sampler,
                (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer);
            let wets = vec4<f32>(controls >> vec4<u32>(24u)) * (1.0 / 255.0);
            let low = mix(wets.w, wets.z, f.x);
            let high = mix(wets.x, wets.y, f.x);
            let spacing = terrain.anchor.w * f32(1u << index);
            wet += mix(low, high, f.y) * (left * (1.0 - fade))
                * saturate(TERRAIN_WET_SPACING / spacing);
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
// for its macro colors and detail, the shares of it growing
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
    // The cover's canopy colors and leaves, each weighted by its material's
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

// Layer `layer` of `map` at `world` (with its screen derivatives) on axis
// `axis`'s plane in lattice `lattice`, as `mapping` lays it.
fn terrain_plane_texel(map: texture_2d_array<f32>, layer: u32, mapping: TerrainMapping,
        world: vec3<f32>, world_dx: vec3<f32>, world_dy: vec3<f32>, axis: u32, lattice: u32)
        -> vec4<f32> {
    let shift = vec2<f32>(0.37, 0.71) * f32(lattice);
    return textureSampleGrad(map, terrain_material_sampler,
        terrain_lattice(world_plane(world, axis) * mapping.scale, axis, lattice) + shift,
        layer,
        terrain_lattice(world_plane(world_dx, axis) * mapping.scale, axis, lattice),
        terrain_lattice(world_plane(world_dy, axis) * mapping.scale, axis, lattice));
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
// `scale`: the plane's tangent-space normal (x along its first coordinate,
// y up its image, z out), turned back from the lattice, added to the
// terrain's normal in that plane (whiteout blending).
fn terrain_plane_normal(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, axis: u32, lattice: u32, normal: vec3<f32>, scale: f32)
        -> vec3<f32> {
    let sampled = terrain_plane_texel(terrain_material_data, 2u * mapping.layer, mapping, world,
        world_dx, world_dy, axis, lattice).xyz * 2.0 - 1.0;
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
    if fade == 0.0 {
        return 0.0;
    }
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
// surface, a pixel's footprint there (meters, its long
// side), the footprint the maps' anisotropic filtering resolves (its short
// side, or a sixteenth of its long side where filtering runs out), the
// terrain's normal, the way toward the eye and the meters to it, and the
// share of its cover's leaves blades no longer show there.
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
// from the eye at `world` by their law (`terrain.cover`): all
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
    var end = cover.y;
    // Within 0.55 of the reach the fade has not begun and past it has
    // ended, wherever the noise draws the end in (0.8 to 1 of the reach).
    if cover.y <= cover.x || (distance > 0.55 * cover.y && distance < cover.y) {
        end = cover_end(world.xz, cover.y);
    }
    let share = cover_share(distance, cover.x, cover.y, end);
    return 1.0 - cover_shown(distance, share, cover.x);
}

// How much light reaches blades on average, which their canopy takes:
// their occlusion rises from 0.35 at the root to 1 at the
// tip (`meshes/cover.wgsl`), 0.675 over their height, and the half of them
// seen through from the side away from the sun shows 0.8 of its color.
const TERRAIN_CANOPY_OCCLUSION: f32 = 0.6;

// How bright blades seen at `sine` (the sine of the view's angle over the
// ground) show their color on average, which their canopy shows too:
// seen along the ground their lit tops give way to their
// sides and the shade among them. Measured against blades under the low
// and the high sun alike: about 0.6 along the ground, 0.7 from 25 degrees
// up and 0.75 from 45.
fn terrain_canopy_shown(sine: f32) -> f32 {
    return 0.6 + 0.22 * saturate(sine);
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

// How much of the ground at `site` a cover whose leaves' area is `leaves` per
// square meter shows as its canopy: as much of the leaves as its blades no
// longer show by their law (`terrain.cover`) hides of the ground along the
// view, 1 - exp(-G leaves / sin), all of it at grazing views.
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
// and its share of the height contrast its maps show.
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
// where its maps show, and its height contrast as far as they show, so
// materials without maps, or too far to show them, blend by their weights
// alone. All its maps show while a repeat spans enough pixels along the
// footprint's long side; its color map alone farther out, as long as its
// filtering keeps it.
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
    let contrast = entry.params.y * slot.detail;
    if slot.detail > 0.0 {
        let mapping = terrain_mapping(entry, site.geometric);
        let second = slot.pattern;
        var first = vec4<f32>(0.0);
        var other = vec4<f32>(0.0);
        if second < 1.0 {
            first = terrain_map(terrain_material_data, 2u * mapping.layer + 1u, mapping,
                site.world, site.world_dx, site.world_dy, 0u);
        }
        if second > 0.0 {
            other = terrain_map(terrain_material_data, 2u * mapping.layer + 1u, mapping,
                site.world, site.world_dx, site.world_dy, 1u);
        }
        slot.second = second;
        if second > 0.0 && second < 1.0 {
            slot.second = terrain_lattice_share(second, first.a, other.a, contrast);
        }
        slot.texel = mix(first, other, slot.second);
    }
    slot.raised = weight + contrast * slot.texel.a;
    slot.contrast = weight * contrast;
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
    // Each lattice's color texels, kept for the full maps' pass below.
    var first_albedo = vec3<f32>(0.0);
    var second_albedo = vec3<f32>(0.0);
    var have_first = false;
    var have_second = false;
    if slot.colored > slot.detail {
        // Past its maps' full detail, its color map alone, the lattices
        // shared by their pattern.
        let mapping = terrain_mapping(entry, site.geometric);
        var albedo = vec3<f32>(0.0);
        if slot.pattern < 1.0 {
            first_albedo = terrain_map(terrain_albedo, mapping.layer, mapping, site.world,
                site.world_dx, site.world_dy, 0u).rgb;
            have_first = true;
            albedo += (1.0 - slot.pattern) * first_albedo;
        }
        if slot.pattern > 0.0 {
            second_albedo = terrain_map(terrain_albedo, mapping.layer, mapping, site.world,
                site.world_dx, site.world_dy, 1u).rgb;
            have_second = true;
            albedo += slot.pattern * second_albedo;
        }
        color = mix(color, entry.color.rgb * albedo, slot.colored);
    }
    // Normal maps where their deflections still show, fading with the
    // other maps.
    let shown = terrain_relief(site.texel, entry.params.x) * slot.detail;
    if slot.detail > 0.0 {
        let mapping = terrain_mapping(entry, site.geometric);
        let relief = entry.maps.x;
        let deflected = shown > 0.0 && relief > 0.0;
        var albedo = vec3<f32>(0.0);
        var mapped = vec3<f32>(0.0);
        if slot.second < 1.0 {
            var texel = first_albedo;
            if !have_first {
                texel = terrain_map(terrain_albedo, mapping.layer, mapping, site.world,
                    site.world_dx, site.world_dy, 0u).rgb;
            }
            albedo += (1.0 - slot.second) * texel;
            if deflected {
                mapped += (1.0 - slot.second) * terrain_mapped_normal(mapping, site.world,
                    site.world_dx, site.world_dy, 0u, site.geometric, relief);
            }
        }
        if slot.second > 0.0 {
            var texel = second_albedo;
            if !have_second {
                texel = terrain_map(terrain_albedo, mapping.layer, mapping, site.world,
                    site.world_dx, site.world_dy, 1u).rgb;
            }
            albedo += slot.second * texel;
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
    // The slopes normal maps no longer show roughen the surface instead,
    // as their variance widens its highlights.
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
        // Ground under cover shows as its canopy where blades thin out:
        // their color, rougher and shaded among the
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
        (*ground).canopy_color += weight * entry.canopy.rgb;
        (*ground).leaves += weight * entry.canopy.w;
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
// where its maps show raised by its height map as sharply as its height
// contrast asks, so stones and tufts stand through what they meet before
// it covers them.
fn terrain_ground(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainGround {
    return terrain_shared_ground(terrain_shares(site, shading), site);
}

fn terrain_shares(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainShares {
    let blend = terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, shading);
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
// sinks the ground at most, already weighted and faded, and where its
// maps' heights lie: as its maps lie there (from above, or across their
// planes on steep ground), read with the point's change per pixel.
struct TerrainReliefLayer {
    depth: f32,
    mapping: TerrainMapping,
    dx: vec3<f32>,
    dy: vec3<f32>,
}

fn terrain_relief_layer(material: u32, weight: f32, fade: f32, site: TerrainSite)
        -> TerrainReliefLayer {
    var layer: TerrainReliefLayer;
    layer.depth = 0.0;
    if weight <= 0.0 || material == 0xffffffffu {
        return layer;
    }
    let entry = terrain_material(material);
    // Materials with maps, close enough to show them.
    if entry.params.w <= 0.0 {
        return layer;
    }
    layer.mapping = terrain_mapping(entry, site.geometric);
    layer.dx = site.world_dx;
    layer.dy = site.world_dy;
    layer.depth = entry.relief.x * weight * fade
        * terrain_map_detail(site.footprint, entry.params.x);
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

// Where a fragment's view ray meets the ground: `world` sunk to where it
// shades, and `deform` the meters its materials let tracks sink it at
// most where the deformation map holds tracks nearby (otherwise 0).
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

// Where a fragment's view ray meets the ground's relief:
// within the relief distance, the ray marches down through the heights of
// its two heaviest materials, from above or across their planes on steep
// ground as their maps lie, each as deep as its relief
// reaches and flattening toward the distance's end, and through the
// tracks deformers pressed, from the top of their rims to
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
    // The point's move per meter of depth.
    let march = vec3<f32>(along.x, travel.y, along.y);
    // Relief whose deepest point would move under a pixel takes no march.
    if (depth - top) * min(across, 3.0) < 0.75 * site.footprint {
        return relief;
    }
    let base = mix(3.0, terrain.relief.y, fade);
    // A step every two centimeters of tracks, up to twice the relief's.
    let steps = max(u32(round(clamp((depth - top) / 0.02, base, 2.0 * terrain.relief.y))), 2u);
    var previous = top;
    var gap = top - terrain_sink(a, b, site.world + march * top, deform, fade);
    var hit = top;
    if gap < 0.0 {
        hit = depth;
        for (var step = 1u; step <= steps; step++) {
            let at = top + (depth - top) * f32(step) / f32(steps);
            let point = site.world + march * at;
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
    // the ground, and how it picks between two mips (`terrain_vt_texels`).
    textured: bool,
    dither: f32,
}

// How a fragment of its own triangle `relative` to the eye looks: from
// the triangle, without the virtual texture.
fn terrain_forward_look(relative: vec3<f32>) -> TerrainLook {
    let world = relative + camera.position.xyz;
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
// says, meets the ground: its site, sunk where the ground's relief and
// tracks show, the shading levels of its distance and the meters tracks
// may sink it (`TerrainRelief`).
struct TerrainSeen {
    site: TerrainSite,
    shading: TerrainShadingLevel,
    deform: f32,
}

fn terrain_seen(relative: vec3<f32>, look: TerrainLook) -> TerrainSeen {
    let world = relative + camera.position.xyz;
    let away = length(relative);
    let ray = relative / max(away, 1.0e-6);
    let ray_dx = look.ray_dx;
    let ray_dy = look.ray_dy;
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_shading_level(length(world - terrain.eye.xyz));
    let geometric = terrain_surface_normal(local, shading);
    // How the point moves per pixel on the plane of the terrain's smooth
    // normal: where the rays one pixel over meet that plane,
    // which change smoothly across the triangles drawn, so fading detail and
    // mipmaps change with them rather than from one triangle to the next.
    let facing = min(dot(geometric, ray), -0.015625);
    // Its materials as finely as the display shows them: per display pixel
    // (the render size's share of a pixel), as the mip bias samples.
    let detail = material_detail();
    let smooth_dx = away * detail * (ray_dx - ray * (dot(geometric, ray_dx) / facing));
    let smooth_dy = away * detail * (ray_dy - ray * (dot(geometric, ray_dy) / facing));
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
    // Sunk where the ground's relief and tracks show.
    let relief = terrain_parallax(site, shading);
    site.world = relief.world;
    return TerrainSeen(site, shading, relief.deform);
}

// `terrain_point` before its highlights widen with its normal's turn.
fn terrain_point_seen(relative: vec3<f32>, look: TerrainLook) -> SurfacePoint {
    let seen = terrain_seen(relative, look);
    let site = seen.site;
    let world = relative + camera.position.xyz;
    let local = world.xz - terrain.origin.xz;
    let geometric = site.geometric;
    let footprint = site.footprint;
    // The runtime virtual texture's ground where its pages hold it, its
    // detail and macro colors baked in; computed here otherwise.
    var ground: TerrainGround;
    var baked = false;
    if look.textured {
        let texels = terrain_vt_texels(site, look.dither);
        if texels.found {
            ground = terrain_vt_ground(texels, site);
            baked = true;
        }
    }
    var broad = 0.5;
    var patchy = 0.5;
    var fine = 0.5;
    if !baked {
        ground = terrain_ground(site, seen.shading);
        broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, footprint);
        patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, footprint);
        fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, footprint);
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
    if !baked {
        // Plain colors vary at every scale; maps carry their own detail
        // and vary broadly.
        let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
        let mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
        albedo *= mix(plain, mapped, ground.mapped);
    }
    if !baked {
        // Its macro colors, as on the blades growing on it.
        let colors = terrain_macro_texel(local, footprint);
        albedo *= macro_tint(colors, ground.vegetated, macro_strength(site.distance));
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
    // The light the ground around sends where it hides the sky: what the
    // terrain's probes gather where they relight, and in the far field
    // past their reach an estimate.
    let around = bounce_light(world, normal, true);
    point.bounce = mix((1.0 - sky) * albedo * terrain_bounce(), around.light * occlusion,
        around.weight);
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
    terrain_holes_test(input.cell, input.holes);
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
// `dither` picks the runtime virtual texture's mips (`terrain_vt_texels`).
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
        * vec4<f32>(relative + camera.position.xyz, 1.0)).xyw;
    return resolved;
}

@fragment
fn fs_terrain_resolve_post(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(finish(terrain_resolved(position, 0.5).point, vec3<f32>(0.0)), position.xy);
}

@fragment
fn fs_terrain_resolve_post_lit(@builtin(position) position: vec4<f32>) -> SceneTargets {
    let point = terrain_resolved(position, 0.5).point;
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
    terrain_holes_test(input.cell, input.holes);
    return finish(point, vec3<f32>(0.0));
}

fn shade_terrain_lit(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative.xyz);
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
