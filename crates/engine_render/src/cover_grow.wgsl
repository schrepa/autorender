// Ground cover: each frame, the blades around the eye, grown
// from a terrain's control values and heights. Every cell of the terrain's
// finest level holds the same candidate blades at every distance, each
// placed, sized and ranked by a hash of the cell and its index, so blades
// stay where they are as the eye moves: its index counts the
// cell's expected blades at which it grows, plus a share of one. A
// candidate grows where that falls under its material's density at its
// point times the cell's area (both cover materials of a control value
// share it by their blend). Density falls with distance as the radius's
// fifth over the distance, so every ring of the radius's disc holds about
// as many blades, which widen to cover as much ground; past the radius,
// sparser as the cube of the distance, blades still widening so they stay
// as wide on screen, until the tiers' ragged reach (`terrain_variety.wgsl`'s
// `cover_share`); without tiers cover thins out over the radius's outer
// half. A workgroup takes one cell of a tier's grid: the radius's cells are
// the finest level's, and each tier beyond has cells twice
// as wide as the last, trying only the first indices of the finest cells
// within them that its density can reach, so wherever a blade grows it is
// the same blade. Candidates near their threshold grow short, so blades
// rise as the eye nears instead of popping in. Blades stand on the ground
// as the terrain draws it, morphing between its levels.
// Survivors in the view append to the frame's blades; past capacity they
// are only counted. Blades take the ground's large-scale variety from its
// sky map, as the ground does (`terrain_variety.wgsl`).

struct Grow {
    // The terrain's origin (world) and its level-0 sample spacing.
    origin: vec4<f32>,
    // The eye (world) and the radius cover grows within.
    eye: vec4<f32>,
    // The first cell of the grid (terrain-local, in cells), the cells along
    // its side, and the candidates each cell of the finest level holds.
    grid: vec4<i32>,
    // Where the terrain's levels start in the directory and how many there
    // are, where its materials start in the palette, and the blades'
    // capacity.
    counts: vec4<u32>,
    // The density scale, how many of each finest cell's candidates this
    // tier tries, and the terrain's sky map's layer plus one
    // (0 until built).
    scale: vec4<f32>,
    // This tier: the meters from the eye it starts at, the
    // radius, where tiers end (0 without tiers), and how many level-0
    // samples wide its cells are.
    tier: vec4<f32>,
    // The view's frustum planes (world).
    planes: array<vec4<f32>, 6>,
    // Runtime deformation's map (`terrain_vertex.wgsl`'s
    // `deform_grid` and `deform`).
    deform_grid: vec4<i32>,
    deform: vec4<f32>,
    // Meters from the eye up to which each of the terrain's first eight
    // levels draws, finest first: each morphs into the next coarser level
    // over the last quarter of its range; 0 for levels that
    // never morph.
    ranges: array<vec4<f32>, 2>,
}

// A surface material's cover: blades per square meter, the shortest and
// tallest blades and their root width, then its tip color and sway.
struct CoverSpec {
    shape: vec4<f32>,
    color: vec4<f32>,
}

// A surface material as the terrain palette holds it (`terrain.wgsl`):
// its maps' mean color, factors applied, in `far`.
struct PaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    relief: vec4<f32>,
    canopy: vec4<f32>,
}

struct CoverBlade {
    root: vec3<f32>,
    height: f32,
    facing: f32,
    width: f32,
    root_color: u32,
    tip_color: u32,
}

// Blades appended this frame, those past capacity included.
struct Counts {
    appended: atomic<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var control: texture_2d_array<u32>;
@group(0) @binding(2) var<storage, read> directory: array<u32>;
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const PALETTE_ENTRIES: u32 = 170u;
@group(0) @binding(3) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(4) var<storage, read> covers: array<CoverSpec>;
@group(0) @binding(5) var<uniform> grow: Grow;
@group(0) @binding(6) var<storage, read_write> blades: array<CoverBlade>;
@group(0) @binding(7) var<storage, read_write> counts: Counts;
@group(0) @binding(9) var<storage, read> deformation: array<u32>;
@group(0) @binding(10) var sky_maps: texture_2d_array<f32>;

const NOT_RESIDENT: u32 = 0xffffffffu;
const TAU: f32 = 6.2831853;

fn hash(value: u32) -> u32 {
    var h = value * 0x9e3779b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

// What the terrain holds at terrain-local `local`: its height, and the
// nearest sample's control value at the finest resident level.
struct Ground {
    height: f32,
    control: u32,
    found: bool,
}

// A tile of a level: its layer (`NOT_RESIDENT` when not loaded) and its
// first sample (the level's samples from the terrain's origin).
struct LevelTile {
    layer: u32,
    first: vec2<u32>,
}

// The tile of `level` holding its sample `samples` (terrain-local).
fn level_tile(level: u32, samples: vec2<f32>) -> LevelTile {
    let header = grow.counts.x + level * 4u;
    let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
    let tile = min(vec2<u32>(max(samples, vec2<f32>(0.0)) / 256.0), tiles - vec2<u32>(1u));
    var found: LevelTile;
    found.layer = directory[directory[header] + tile.y * tiles.x + tile.x];
    found.first = tile * 256u;
    return found;
}

// The height of `tile`'s sample `sample` (its level's, terrain-local);
// layers keep a border sample before the tile's first.
fn tile_height(tile: LevelTile, sample: vec2<u32>) -> f32 {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return textureLoad(heights, texel, tile.layer, 0).r;
}

// Where `level` stands morphing into the next coarser level `away` meters
// from the eye: over the last quarter of its range.
fn level_morph(level: u32, away: f32) -> f32 {
    if level >= 8u {
        return 0.0;
    }
    let four = select(grow.ranges[0], grow.ranges[1], level >= 4u);
    let range = dot(four, vec4<f32>(vec4<u32>(level % 4u) == vec4<u32>(0u, 1u, 2u, 3u)));
    if range <= 0.0 {
        return 0.0;
    }
    return saturate((away - 0.75 * range) / (0.25 * range));
}

// The height within a cell of corners `a` (0, 0), `b` (1, 0), `c` (0, 1)
// and `d` (1, 1) at `f`, on the two triangles patches split cells into,
// from the (0, 1) corner to the (1, 0) one.
fn triangle_height(a: f32, b: f32, c: f32, d: f32, f: vec2<f32>) -> f32 {
    if f.x + f.y <= 1.0 {
        return a + (b - a) * f.x + (c - a) * f.y;
    }
    return d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
}

// The ground at terrain-local `local` as the terrain draws it `away` meters
// from the eye (`terrain_vertex.wgsl`): its finest resident
// level, or the coarser one it has morphed into, morphing into the next
// over the last quarter of its range, its odd samples sliding onto the
// even ones before them as their heights blend into the coarser level's,
// read on the triangles its patches draw. Control values come from the
// nearest sample of the finest resident level. Nothing grows beyond the
// terrain's far edges.
fn drawn_ground(local: vec2<f32>, away: f32) -> Ground {
    var ground: Ground;
    let base = grow.counts.x;
    let levels = grow.counts.y;
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0 * grow.origin.w;
    if any(local > extent) {
        return ground;
    }
    var level = 0u;
    var tile: LevelTile;
    tile.layer = NOT_RESIDENT;
    for (; level < levels; level++) {
        tile = level_tile(level, local / (grow.origin.w * f32(1u << level)));
        if tile.layer != NOT_RESIDENT {
            break;
        }
    }
    if tile.layer == NOT_RESIDENT {
        return ground;
    }
    let finest = max(local / (grow.origin.w * f32(1u << level)), vec2<f32>(0.0));
    ground.control = textureLoad(control,
        min(vec2<u32>(round(finest)) - tile.first, vec2<u32>(256u)) + vec2<u32>(1u),
        tile.layer, 0).x;
    ground.found = true;
    // A level fully morphed draws as the next coarser one.
    var morph = level_morph(level, away);
    var coarse: LevelTile;
    coarse.layer = NOT_RESIDENT;
    while morph > 0.0 && level + 1u < levels {
        coarse = level_tile(level + 1u, local / (grow.origin.w * f32(2u << level)));
        if coarse.layer == NOT_RESIDENT {
            morph = 0.0;
            break;
        }
        if morph < 1.0 {
            break;
        }
        level += 1u;
        tile = coarse;
        coarse.layer = NOT_RESIDENT;
        morph = level_morph(level, away);
    }
    if level + 1u >= levels {
        morph = 0.0;
    }
    // The morphed grid: within each pair of cells from an even sample, the
    // odd sample between them stands `1 - morph` along.
    let samples = max(local / (grow.origin.w * f32(1u << level)), vec2<f32>(0.0));
    let block = floor(samples * 0.5) * 2.0;
    let within = samples - block;
    let edge = 1.0 - morph;
    let second = within >= vec2<f32>(edge);
    let start = select(vec2<f32>(0.0), vec2<f32>(edge), second);
    let width = select(vec2<f32>(edge), vec2<f32>(1.0 + morph), second);
    let f = saturate((within - start) / max(width, vec2<f32>(1.0e-6)));
    let cell = vec2<u32>(block) + select(vec2<u32>(0u), vec2<u32>(1u), second);
    var a = tile_height(tile, cell);
    var b = tile_height(tile, cell + vec2<u32>(1u, 0u));
    var c = tile_height(tile, cell + vec2<u32>(0u, 1u));
    var d = tile_height(tile, cell + vec2<u32>(1u, 1u));
    if morph > 0.0 {
        // Each corner blends into the coarser level's sample at the even
        // sample it slides onto: the block's first, or after the block for
        // the second cell's far corners.
        let pair = vec2<u32>(block) / 2u;
        let next = pair + select(vec2<u32>(0u), vec2<u32>(1u), second);
        a = mix(a, tile_height(coarse, pair), morph);
        b = mix(b, tile_height(coarse, vec2<u32>(next.x, pair.y)), morph);
        c = mix(c, tile_height(coarse, vec2<u32>(pair.x, next.y)), morph);
        d = mix(d, tile_height(coarse, next), morph);
    }
    ground.height = triangle_height(a, b, c, d, f);
    return ground;
}

// The ground at terrain-local `local` from the finest resident level, its
// samples blended: where a cell's nearest blades stand, to bound them.
fn ground_at(local: vec2<f32>) -> Ground {
    var ground: Ground;
    let base = grow.counts.x;
    let levels = grow.counts.y;
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0 * grow.origin.w;
    if any(local > extent) {
        return ground;
    }
    for (var index = 0u; index < levels; index++) {
        let samples = max(local / (grow.origin.w * f32(1u << index)), vec2<f32>(0.0));
        let tile = level_tile(index, samples);
        if tile.layer == NOT_RESIDENT {
            continue;
        }
        let corner = min(vec2<u32>(samples), tile.first + vec2<u32>(255u));
        let f = saturate(samples - vec2<f32>(corner));
        let a = tile_height(tile, corner);
        let b = tile_height(tile, corner + vec2<u32>(1u, 0u));
        let c = tile_height(tile, corner + vec2<u32>(0u, 1u));
        let d = tile_height(tile, corner + vec2<u32>(1u, 1u));
        ground.height = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        ground.found = true;
        return ground;
    }
    return ground;
}

// How deep tracks sink the ground at world `point`
// (meters, its texel's, as refilled): 0 beyond the deformation map.
fn pressed_at(point: vec2<f32>) -> f32 {
    let size = grow.deform_grid.z;
    if size <= 0 {
        return 0.0;
    }
    let local = point * grow.deform.x - vec2<f32>(grow.deform_grid.xy);
    if any(local < vec2<f32>(0.0)) || any(local >= vec2<f32>(f32(size))) {
        return 0.0;
    }
    let texel = vec2<i32>(floor(local)) + grow.deform_grid.xy;
    let slot = ((texel % size) + size) % size;
    let blocks = grow.deform_grid.w;
    let word = u32(blocks * blocks + slot.y * size + slot.x);
    return unpack2x16float(deformation[word]).x * grow.deform.y;
}

// Whether a sphere at `center` of `radius` lies at least partly inside the
// view.
fn in_view(center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < 6u; plane++) {
        if dot(grow.planes[plane].xyz, center) + grow.planes[plane].w < -radius {
            return false;
        }
    }
    return true;
}

// Packs a linear color as 8-bit square roots, the blades' encoding.
fn pack_color(color: vec3<f32>, w: f32) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(saturate(color)), saturate(w)));
}

// The ground's large-scale variety at terrain-local `local`:
// its sky map's hollows and ridges, interpolated between texels;
// level ground before the map is built.
fn variety_at(local: vec2<f32>) -> vec2<f32> {
    let layer = u32(grow.scale.z);
    if layer == 0u {
        return vec2<f32>(0.5);
    }
    let base = grow.counts.x;
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0
        * grow.origin.w;
    let size = vec2<i32>(textureDimensions(sky_maps));
    let at = local / extent * vec2<f32>(size) - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = size - vec2<i32>(1);
    let a = textureLoad(sky_maps, clamp(first, vec2<i32>(0), last), layer - 1u, 0).gb;
    let b = textureLoad(sky_maps, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last), layer - 1u,
        0).gb;
    let c = textureLoad(sky_maps, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last), layer - 1u,
        0).gb;
    let d = textureLoad(sky_maps, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last), layer - 1u,
        0).gb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The cell's ground at its center, read once for its lanes.
var<workgroup> cell_height: f32;
var<workgroup> cell_found: u32;

@compute @workgroup_size(64)
fn grow_cover(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    // The tier's cells, as many level-0 samples wide as it asks.
    let scale = grow.tier.w;
    let side = u32(scale);
    let spacing = grow.origin.w * scale;
    let cell = grow.grid.xy + vec2<i32>(group.xy);
    if any(cell < vec2<i32>(0)) {
        return;
    }
    let corner = vec2<f32>(cell) * spacing;
    let center = grow.origin.xz + corner + vec2<f32>(0.5 * spacing);
    let radius = grow.eye.w;
    let inner = grow.tier.x;
    let half_diagonal = 0.7072 * spacing;
    let across = distance(center, grow.eye.xz);
    if across > radius + half_diagonal {
        return;
    }
    if lane == 0u {
        let ground = ground_at(corner + vec2<f32>(0.5 * spacing));
        cell_height = ground.height;
        cell_found = u32(ground.found);
    }
    if workgroupUniformLoad(&cell_found) == 0u {
        return;
    }
    // The cell's nearest point's share bounds its candidates' before any
    // reads its ground.
    let cell_center = vec3<f32>(center.x, grow.origin.y + workgroupUniformLoad(&cell_height),
        center.y);
    let center_away = distance(cell_center, grow.eye.xyz);
    let nearest = max(center_away - half_diagonal, 0.0);
    // Blades join tiers by their distance through the air, so
    // seen from above a tier's blades stand well within its inner distance
    // across the ground: only cells whose farthest ground (sloping up to
    // twice as steep as level across them) stays nearer than the tier skip.
    if nearest > radius || center_away + 3.0 * half_diagonal < inner {
        return;
    }
    // The finest cells' candidates whose indices the cell's nearest point's
    // share could reach, of those the tier tries: each index stands for a
    // blade more of a finest cell's expected blades.
    let radius_densest = grow.tier.y;
    let reach = grow.tier.z;
    let held = f32(grow.grid.w);
    let tried = min(u32(grow.scale.y),
        u32(ceil(held * cover_share(nearest, radius_densest, reach, reach))));
    let finest = side * side;
    let area = grow.origin.w * grow.origin.w;
    for (var index = lane; index < finest * tried; index += 64u) {
        // A finest cell within the tier's, and its candidate.
        let within = index % finest;
        let candidate = index / finest;
        let at = cell * i32(side) + vec2<i32>(i32(within % side), i32(within / side));
        let cell_seed = hash((u32(at.x) * 0x8da6b343u) ^ (u32(at.y) * 0xd8163841u));
        let seed = hash(cell_seed + candidate * 0x632be5abu);
        let rank = f32(candidate) + unit(hash(seed ^ 0x165667b1u));
        let offset = vec2<f32>(unit(seed), unit(hash(seed ^ 0x5bd1e995u))) * grow.origin.w;
        let local = vec2<f32>(at) * grow.origin.w + offset;
        let world_xz = grow.origin.xz + local;
        // No nearer than across the ground, it can grow no denser.
        if rank >= held * cover_share(distance(world_xz, grow.eye.xz), radius_densest, reach,
            reach) {
            continue;
        }
        // The ground morphs by the distance to the cell's ground there.
        let level_away = distance(vec3<f32>(world_xz.x, cell_center.y, world_xz.y), grow.eye.xyz);
        let ground = drawn_ground(local, level_away);
        if !ground.found {
            continue;
        }
        let root = vec3<f32>(world_xz.x, grow.origin.y + ground.height - 0.02, world_xz.y);
        let away = distance(root, grow.eye.xyz);
        if away > radius || away < inner {
            continue;
        }
        // The control value's two materials share the candidates by
        // their blend.
        let pick = unit(hash(seed ^ 0x27d4eb2fu));
        let share = f32((ground.control >> 16u) & 0xffu) * (1.0 / 255.0);
        let material = select(ground.control & 0xffu, (ground.control >> 8u) & 0xffu,
            pick < share);
        let entry = min(grow.counts.z + material, PALETTE_ENTRIES - 1u);
        let spec = covers[entry];
        let falloff = cover_share(away, radius_densest, reach, cover_end(world_xz, reach));
        let keep = spec.shape.x * grow.scale.x * area * falloff;
        if rank >= keep {
            continue;
        }
        let rise = saturate((keep - rank) / max(0.25 * keep, 1.0e-4));
        // Blades lie down in tracks, as deep as the ground's
        // material lets them sink it.
        let pressed = min(pressed_at(world_xz), palette[entry].relief.y);
        let standing = 1.0 - 0.85 * smoothstep(0.01, 0.06, pressed);
        let height = mix(spec.shape.y, spec.shape.z, unit(hash(seed ^ 0xc2b2ae35u))) * rise
            * standing;
        if height < 0.02 {
            continue;
        }
        if !in_view(root + vec3<f32>(0.0, 0.5 * height, 0.0), height) {
            continue;
        }
        let variation = (0.85 + 0.3 * unit(hash(seed ^ 0x9e3779b9u)))
            * ground_variety(variety_at(local), world_xz, 1.0);
        var blade: CoverBlade;
        blade.root = root;
        blade.height = height;
        blade.facing = unit(hash(seed ^ 0x85ebca6bu)) * TAU;
        // Wider where fewer stand, as wide on screen at every distance.
        blade.width = spec.shape.w * max(1.0, away / (0.2 * radius_densest));
        blade.root_color = pack_color(palette[entry].far.rgb * variation, 0.0);
        blade.tip_color = pack_color(spec.color.rgb * variation, spec.color.w);
        let slot = atomicAdd(&counts.appended, 1u);
        if slot < grow.counts.w {
            blades[slot] = blade;
        }
    }
}

// The indirect draw of the frame's blades: a strip of two vertices per
// joint each, as many blades as fit.
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(8) var<storage, read_write> args: DrawArgs;

// Joints along a blade (`meshes/cover.wgsl`'s segments and one).
const JOINTS: u32 = 5u;

@compute @workgroup_size(1)
fn finish_cover() {
    args.vertex_count = 2u * JOINTS;
    args.instance_count = min(atomicLoad(&counts.appended), grow.counts.w);
    args.first_vertex = 0u;
    args.first_instance = 0u;
}
