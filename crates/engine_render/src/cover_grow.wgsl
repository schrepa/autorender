// Ground cover: each frame, the blades around the detail eye, grown
// from a terrain's control values and heights. Every cell of the terrain's
// finest level holds the same candidate blades at every distance, each
// placed, sized and ranked by a hash of the cell and its index, so blades
// stay where they are as the eye moves: its index counts the
// cell's expected blades at which it grows, plus a share of one. A
// candidate grows where that falls under its material's density at its
// point times the cell's area (both cover materials of a control value
// share it by their blend), lusher ground growing it denser and its blades
// taller, drier ground sparser and shorter (`terrain_variety.wgsl`'s
// `cover_growth`). Density falls with distance as the radius's fifth over
// the distance, so every ring of the radius's disc holds about as many
// blades, which widen to cover as much ground as far as their shape lets
// them stay grass leaves, then thin without widening; past the radius,
// sparser as the square of the distance at the width they reached, so they
// narrow on screen as they recede; and blades narrower on screen than the
// settings' pixels retire in their ranks' order, none standing past the
// tiers' reach (`terrain_variety.wgsl`'s `cover_share`); without tiers cover
// thins out over the radius's outer half. The ground shows the leaves
// thinned and retired blades leave as its canopy, their average
// (`meshes/terrain.wgsl`). A workgroup takes one cell of a tier's grid: the
// radius's cells are the finest level's, and each tier beyond has cells twice
// as wide as the last, trying only the first indices of the finest cells
// within them that its density can reach, so wherever a blade grows it is
// the same blade. Candidates near their threshold grow short, so blades
// rise as the eye nears instead of popping in. Blades stand on the ground's
// true surface, which the terrain draws within about a pixel
// (`terrain_ground.wgsl`).
// Survivors in the view append to the frame's blades; past capacity they
// are only counted. Blades take the tint of the ground's macro colors, as
// the ground does (`terrain_variety.wgsl`).

struct Grow {
    // The terrain's origin (world) and its level-0 sample spacing.
    origin: vec4<f32>,
    // The detail eye (world) and the radius cover grows within.
    eye: vec4<f32>,
    // The first cell of the grid (terrain-local, in cells), the cells along
    // its side, and the candidates each cell of the finest level holds.
    grid: vec4<i32>,
    // Where the terrain's levels start in the directory and how many there
    // are, where its materials start in the palette, and the blades'
    // capacity.
    counts: vec4<u32>,
    // The density scale, how many of each finest cell's candidates this
    // tier tries, the terrain's macro colors' layer plus one
    // (0 until they arrive), and the tallest blade any material grows on the
    // lushest ground.
    scale: vec4<f32>,
    // This tier: the meters from the eye it starts at, the
    // radius, the meters away blades past the radius span the settings'
    // pixels (0: cover within the radius alone), and how many level-0
    // samples wide its cells are.
    tier: vec4<f32>,
    // The view's frustum planes (world).
    planes: array<vec4<f32>, 6>,
    // Runtime deformation's map (`terrain_vertex.wgsl`'s
    // `deform_grid` and `deform`).
    deform_grid: vec4<i32>,
    deform: vec4<f32>,
    // Foliage: 1 while blades keep their ground's normal (its bases blend
    // into the ground).
    foliage: vec4<f32>,
    // The view: pixels a meter spans a meter from the eye (x).
    view: vec4<f32>,
    // The most times their roots' width blades widen (x: `CoverLaw`'s
    // `widest`).
    widening: vec4<f32>,
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
    placed: vec4<f32>,
    placed_size: vec4<f32>,
    placed_thinning: vec4<f32>,
    heights: vec4<f32>,
}

// A blade, as `meshes/cover.wgsl` reads it.
struct CoverBlade {
    root: vec3<f32>,
    height: f32,
    facing: u32,
    width: f32,
    root_color: u32,
    tip_color: u32,
}

// Blades appended this frame, those past capacity included, and of those
// kept the ones drawn whole, from the front of `blades`, and the ones
// drawn straight, from its back.
struct Counts {
    appended: atomic<u32>,
    whole: atomic<u32>,
    straight: atomic<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var control: texture_2d_array<u32>;
@group(0) @binding(2) var<storage, read> directory: array<u32>;
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const PALETTE_ENTRIES: u32 = 102u;
@group(0) @binding(3) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(4) var<storage, read> covers: array<CoverSpec>;
@group(0) @binding(5) var<uniform> grow: Grow;
@group(0) @binding(6) var<storage, read_write> blades: array<CoverBlade>;
@group(0) @binding(7) var<storage, read_write> counts: Counts;
@group(0) @binding(9) var<storage, read> deformation: array<u32>;
@group(0) @binding(10) var macro_maps: texture_2d_array<f32>;
// Every resident tile's normal maps (`terrain_normals.wgsl`): the smooth
// surface's gradient at each sample, two halves of a float.
@group(0) @binding(11) var normals: texture_2d_array<u32>;

const TAU: f32 = 6.2831853;

fn hash(value: u32) -> u32 {
    var h = value * 0x9e3779b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

// The terrain cover grows on, as its ground reads.
fn grown() -> GroundTerrain {
    return GroundTerrain(grow.counts.x, grow.counts.y, grow.origin.w);
}

// What the terrain holds at terrain-local `local`: its height, and the
// nearest sample's control value at the finest resident level, which is
// `level`.
struct Ground {
    height: f32,
    control: u32,
    found: bool,
    level: u32,
}

// The ground at terrain-local `local` on its true surface
// (`terrain_ground.wgsl`'s `true_height`). Control values come from the
// nearest sample of the finest resident level. Nothing grows beyond the
// terrain's far edges.
fn true_ground(local: vec2<f32>) -> Ground {
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
        tile = level_tile(grown(), level, local / (grow.origin.w * f32(1u << level)));
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
    ground.height = true_height(grown(), local, level, tile);
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
        let tile = level_tile(grown(), index, samples);
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
        ground.level = index;
        return ground;
    }
    return ground;
}

// Heights as unsigned integers in the same order, for atomic extremes.
fn ordered_height(height: f32) -> u32 {
    let bits = bitcast<u32>(height);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn ordered_to_height(ordered: u32) -> f32 {
    return bitcast<f32>(select(~ordered, ordered & 0x7fffffffu, (ordered & 0x80000000u) != 0u));
}

// Whether a sphere of `radius` centered anywhere in the box from `low` to
// `high` may lie at least partly inside the view: as `in_view` passes some
// such sphere only where this does.
fn box_in_view(low: vec3<f32>, high: vec3<f32>, radius: f32) -> bool {
    let center = 0.5 * (low + high);
    let extent = 0.5 * (high - low);
    for (var plane = 0u; plane < 6u; plane++) {
        let normal = grow.planes[plane].xyz;
        if dot(normal, center) + dot(abs(normal), extent) + grow.planes[plane].w < -radius {
            return false;
        }
    }
    return true;
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

// Meters a cell's bound adds to the tallest blade, past the rounding its
// arithmetic and the candidates' may differ by.
const CELL_MARGIN: f32 = 0.05;

// The samples of `level` (counted from the terrain's origin at that
// level) the ground of a cell from terrain-local `corner`, `spacing` wide,
// can stand on as `true_ground` reads it on that level: the first, none
// before `first`, and how many along each side.
struct CellSamples {
    first: vec2<u32>,
    count: vec2<u32>,
}

fn cell_samples(corner: vec2<f32>, spacing: f32, level: u32, first: vec2<u32>) -> CellSamples {
    let per = grow.origin.w * f32(1u << level);
    // Corners of the cells that hold its points, a thousandth of a sample
    // wider for rounding.
    let low = max(vec2<u32>(max(floor(corner / per - 0.001), vec2<f32>(0.0))), first);
    let high = vec2<u32>(floor((corner + vec2<f32>(spacing)) / per + 0.001)) + 1u;
    var rect: CellSamples;
    rect.first = low;
    rect.count = high + 1u - min(low, high + 1u);
    return rect;
}

// The ground's normal at terrain-local `local` from its finest resident
// level: its smooth surface's, the gradients at its cell's samples blended
// across it, as the terrain shades it.
fn ground_normal(local: vec2<f32>) -> vec3<f32> {
    for (var index = 0u; index < grow.counts.y; index++) {
        let per = grow.origin.w * f32(1u << index);
        let samples = max(local / per, vec2<f32>(0.0));
        let tile = level_tile(grown(), index, samples);
        if tile.layer == NOT_RESIDENT {
            continue;
        }
        let corner = min(vec2<u32>(samples), tile.first + vec2<u32>(255u));
        let f = saturate(samples - vec2<f32>(corner));
        // Layers keep a border sample before the tile's first.
        let texel = corner - tile.first + vec2<u32>(1u);
        let a = unpack2x16float(textureLoad(normals, texel, tile.layer, 0).x);
        let b = unpack2x16float(textureLoad(normals, texel + vec2<u32>(1u, 0u), tile.layer, 0).x);
        let c = unpack2x16float(textureLoad(normals, texel + vec2<u32>(0u, 1u), tile.layer, 0).x);
        let d = unpack2x16float(textureLoad(normals, texel + vec2<u32>(1u, 1u), tile.layer, 0).x);
        let gradient = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// A blade's facing word while blades keep their ground's normal
// (`meshes/cover.wgsl`'s `blade_facing`): `facing` (radians) in the low 16
// bits, `normal`'s x and z as 8-bit fractions above.
fn grounded_facing(facing: f32, normal: vec3<f32>) -> u32 {
    let turn = u32(round(fract(facing / TAU) * 65535.0));
    let tilt = vec2<u32>(round((clamp(normal.xz, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5
        + 0.5) * 255.0));
    return turn | (tilt.x << 16u) | (tilt.y << 24u);
}

// Packs a linear color as 8-bit square roots, the blades' encoding.
fn pack_color(color: vec3<f32>, w: f32) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(saturate(color)), saturate(w)));
}

// The ground's macro colors at terrain-local `local`, interpolated
// between their texels; ground as it is before they arrive.
fn colors_at(local: vec2<f32>) -> vec4<f32> {
    let layer = u32(grow.scale.z);
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let base = grow.counts.x;
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0
        * grow.origin.w;
    let size = vec2<i32>(textureDimensions(macro_maps));
    let at = local / extent * vec2<f32>(size) - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = size - vec2<i32>(1);
    let a = textureLoad(macro_maps, clamp(first, vec2<i32>(0), last), layer - 1u, 0);
    let b = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last),
        layer - 1u, 0);
    let c = textureLoad(macro_maps, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last),
        layer - 1u, 0);
    let d = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last),
        layer - 1u, 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The cell's ground at its center, read once for its lanes, and its finest
// resident level there.
var<workgroup> cell_height: f32;
var<workgroup> cell_found: u32;
var<workgroup> cell_level: u32;
// The levels whose samples bound the cell's ground (the first in the low
// half, the last in the high; none past the first when it goes unbounded),
// the lowest (inverted) and highest ground its candidates may stand on, in
// `ordered_height`'s order, and whether their blades may show in the view.
var<workgroup> cell_levels: u32;
var<workgroup> cell_low: atomic<u32>;
var<workgroup> cell_high: atomic<u32>;
var<workgroup> cell_seen: u32;

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
        cell_level = ground.level;
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
    // blade more of a finest cell's expected blades on the lushest ground.
    let law = CoverLaw(grow.tier.y, grow.tier.z, grow.widening.x);
    let held = f32(grow.grid.w);
    let tried = min(u32(grow.scale.y), u32(ceil(held * cover_share(nearest, law))));
    let finest = side * side;
    // Whether any candidate's blade can show, before any reads its ground:
    // the samples its ground stands on across the cell (`true_ground`'s, of
    // the finest level resident at its middle and the next coarser, where
    // the cell reaches past that level's tile) bound its root, and the
    // tallest blade its sphere. Cells whose bound would read more samples
    // than four for each candidate (each reads four heights or more), or for
    // each lane, go unbounded. One lane plans it and all read the samples.
    let middle = corner + vec2<f32>(0.5 * spacing);
    if lane == 0u {
        let first = cell_level;
        let last = min(cell_level + 1u, grow.counts.y - 1u);
        var samples = 0u;
        for (var level = first; level <= last; level++) {
            let rect = cell_samples(corner, spacing, level, vec2<u32>(0u));
            samples += rect.count.x * rect.count.y;
        }
        let bounded = samples <= 4u * max(finest * tried, 64u);
        cell_levels = first | (select(first, last + 1u, bounded) << 16u);
    }
    let levels = workgroupUniformLoad(&cell_levels);
    let first_level = levels & 0xffffu;
    let end_level = levels >> 16u;
    if end_level > first_level {
        var lowest = 0xffffffffu;
        var highest = 0u;
        for (var level = first_level; level < end_level; level++) {
            let tile = level_tile(grown(), level,
                middle / (grow.origin.w * f32(1u << level)));
            if tile.layer == NOT_RESIDENT {
                continue;
            }
            let rect = cell_samples(corner, spacing, level, tile.first);
            let count = rect.count.x * rect.count.y;
            // Rounds of the lanes, so all leave together.
            for (var start = 0u; start < count; start += 64u) {
                let index = start + lane;
                if index < count {
                    let sample = rect.first
                        + vec2<u32>(index % rect.count.x, index / rect.count.x);
                    let height = ordered_height(tile_height(tile, sample));
                    lowest = min(lowest, height);
                    highest = max(highest, height);
                }
            }
        }
        if highest != 0u {
            atomicMax(&cell_low, ~lowest);
            atomicMax(&cell_high, highest);
        }
        workgroupBarrier();
        if lane == 0u {
            // Roots stand 2 cm under the ground; blade spheres center
            // half their height above them.
            let tallest = grow.scale.w;
            let low = ordered_to_height(~atomicLoad(&cell_low));
            let high = ordered_to_height(atomicLoad(&cell_high));
            let origin = grow.origin.xyz + vec3<f32>(corner.x, -0.02, corner.y);
            cell_seen = u32(box_in_view(origin + vec3<f32>(0.0, low, 0.0),
                origin + vec3<f32>(spacing, high + 0.5 * tallest, spacing),
                tallest + CELL_MARGIN));
        }
        if workgroupUniformLoad(&cell_seen) == 0u {
            return;
        }
    }
    let area = grow.origin.w * grow.origin.w;
    let lushest = cover_growth(1.0).density;
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
        if rank >= held * cover_share(distance(world_xz, grow.eye.xz), law) {
            continue;
        }
        let ground = true_ground(local);
        if !ground.found {
            continue;
        }
        let root = vec3<f32>(world_xz.x, grow.origin.y + ground.height - 0.02, world_xz.y);
        let away = distance(root, grow.eye.xyz);
        if away > radius || away < inner {
            continue;
        }
        // The control value's two materials share the candidates by
        // their blend, the second's up to half.
        let pick = unit(hash(seed ^ 0x27d4eb2fu));
        let share = f32((ground.control >> 16u) & 0xffu) * (1.0 / 255.0);
        let second = pick < share;
        let material = select(ground.control & 0xffu, (ground.control >> 8u) & 0xffu, second);
        // A material's blades shorten where it gives way: full height where
        // it holds half the ground or more, none where it holds none, so
        // cover tapers into paths, rock and other cover rather than
        // standing full height to their edges.
        let edge = smoothstep(0.0, 0.5, select(1.0 - share, share, second));
        let entry = min(grow.counts.z + material, PALETTE_ENTRIES - 1u);
        let spec = covers[entry];
        let falloff = cover_share(away, law);
        // Its material's own share of the candidates, halfway lush: those
        // past what the lushest ground grows grow nowhere, so only the rest
        // read the ground's colors.
        let own = spec.shape.x * grow.scale.x * area * falloff;
        if rank >= own * lushest {
            continue;
        }
        // Lusher ground grows the material's blades denser and taller, drier
        // ground sparser and shorter.
        let colors = colors_at(local);
        let growth = cover_growth(colors.a);
        let keep = own * growth.density;
        if rank >= keep {
            continue;
        }
        let rise = saturate((keep - rank) / max(0.25 * keep, 1.0e-4));
        // Blades lie down in tracks, as deep as the ground's
        // material lets them sink it.
        let pressed = min(pressed_at(world_xz), palette[entry].relief.y);
        let standing = 1.0 - 0.85 * smoothstep(0.01, 0.06, pressed);
        let height = mix(spec.shape.y, spec.shape.z, unit(hash(seed ^ 0xc2b2ae35u)))
            * growth.height * rise * standing * edge;
        if height < 0.02 {
            continue;
        }
        if !in_view(root + vec3<f32>(0.0, 0.5 * height, 0.0), height) {
            continue;
        }
        let variation = (0.85 + 0.3 * unit(hash(seed ^ 0x9e3779b9u))) * macro_tint(colors, 1.0);
        var blade: CoverBlade;
        blade.root = root;
        blade.height = height;
        let facing = unit(hash(seed ^ 0x85ebca6bu)) * TAU;
        if grow.foliage.x > 0.0 {
            blade.facing = grounded_facing(facing, ground_normal(local));
        } else {
            blade.facing = bitcast<u32>(facing);
        }
        // Wider where fewer stand within the radius, covering as much
        // ground, as far as their shape lets them stay leaves; past that as
        // wide as there, narrowing on screen.
        blade.width = spec.shape.w * cover_widening(away, law);
        blade.root_color = pack_color(palette[entry].far.rgb * variation, 0.0);
        blade.tip_color = pack_color(spec.color.rgb * variation, spec.color.w);
        // Kept blades fill the list from both ends: those whose height
        // spans fewer than STRAIGHT_PIXELS pixels, where no pixel shows
        // their curve, from the back, drawn straight with fewer joints.
        if atomicAdd(&counts.appended, 1u) < grow.counts.w {
            if height * grow.view.x < STRAIGHT_PIXELS * away {
                blades[grow.counts.w - 1u - atomicAdd(&counts.straight, 1u)] = blade;
            } else {
                blades[atomicAdd(&counts.whole, 1u)] = blade;
            }
        }
    }
}

// The indirect draws of the frame's blades, strips of two vertices per
// joint each: the whole ones, then the straight ones, whose vertices count
// from STRAIGHT_FIRST_VERTEX so the vertex stage tells them apart.
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(8) var<storage, read_write> args: array<DrawArgs, 2>;

// Joints along a blade (`meshes/cover.wgsl`'s segments and one), and
// along a straight one.
const JOINTS: u32 = 5u;
const STRAIGHT_JOINTS: u32 = 3u;
const STRAIGHT_FIRST_VERTEX: u32 = 1024u;
// Pixels a blade's height spans below which it draws straight.
const STRAIGHT_PIXELS: f32 = 12.0;

@compute @workgroup_size(1)
fn finish_cover() {
    args[0] = DrawArgs(2u * JOINTS, atomicLoad(&counts.whole), 0u, 0u);
    args[1] = DrawArgs(2u * STRAIGHT_JOINTS, atomicLoad(&counts.straight),
        STRAIGHT_FIRST_VERTEX, 0u);
}
