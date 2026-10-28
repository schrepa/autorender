struct Grow {
    origin: vec4<f32>,
    eye: vec4<f32>,
    grid: vec4<i32>,
    counts: vec4<u32>,
    scale: vec4<f32>,
    tier: vec4<f32>,
    planes: array<vec4<f32>, 6>,
    deform_grid: vec4<i32>,
    deform: vec4<f32>,
    foliage: vec4<f32>,
    view: vec4<f32>,
}

struct CoverSpec {
    shape: vec4<f32>,
    color: vec4<f32>,
}

struct PaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    relief: vec4<f32>,
    canopy: vec4<f32>,
    placed: vec4<f32>,
    placed_size: vec4<f32>,
    heights: vec4<f32>,
}

struct CoverBlade {
    root: vec3<f32>,
    size: u32,
    facing: u32,
    leaves: u32,
    root_color: u32,
    tip_color: u32,
}

struct Counts {
    appended: atomic<u32>,
    whole: atomic<u32>,
    straight: atomic<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var control: texture_2d_array<u32>;
@group(0) @binding(2) var<storage, read> directory: array<u32>;
const PALETTE_ENTRIES: u32 = 113u;
@group(0) @binding(3) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(4) var<storage, read> covers: array<CoverSpec>;
@group(0) @binding(5) var<uniform> grow: Grow;
@group(0) @binding(6) var<storage, read_write> blades: array<CoverBlade>;
@group(0) @binding(7) var<storage, read_write> counts: Counts;
@group(0) @binding(9) var<storage, read> deformation: array<u32>;
@group(0) @binding(10) var macro_maps: texture_2d_array<f32>;
@group(0) @binding(11) var normals: texture_2d_array<u32>;

fn hash(value: u32) -> u32 {
    var h = value * 0x9e3779b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn grown() -> GroundTerrain {
    return GroundTerrain(grow.counts.x, grow.counts.y, grow.origin.w);
}

struct Ground {
    height: f32,
    control: u32,
    found: bool,
    level: u32,
}

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

fn ordered_height(height: f32) -> u32 {
    let bits = bitcast<u32>(height);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn ordered_to_height(ordered: u32) -> f32 {
    return bitcast<f32>(select(~ordered, ordered & 0x7fffffffu, (ordered & 0x80000000u) != 0u));
}

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

fn in_view(center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < 6u; plane++) {
        if dot(grow.planes[plane].xyz, center) + grow.planes[plane].w < -radius {
            return false;
        }
    }
    return true;
}

const CELL_MARGIN: f32 = 0.05;

struct CellSamples {
    first: vec2<u32>,
    count: vec2<u32>,
}

fn cell_samples(corner: vec2<f32>, spacing: f32, level: u32, first: vec2<u32>) -> CellSamples {
    let per = grow.origin.w * f32(1u << level);
    let low = max(vec2<u32>(max(floor(corner / per - 0.001), vec2<f32>(0.0))), first);
    let high = vec2<u32>(floor((corner + vec2<f32>(spacing)) / per + 0.001)) + 1u;
    var rect: CellSamples;
    rect.first = low;
    rect.count = high + 1u - min(low, high + 1u);
    return rect;
}

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

fn grounded_tilt(normal: vec3<f32>) -> u32 {
    let tilt = vec2<u32>(round((clamp(normal.xz, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5
        + 0.5) * 255.0));
    return (tilt.x << 16u) | (tilt.y << 24u);
}

fn pack_color(color: vec3<f32>, w: f32) -> u32 {
    return pack4x8unorm(vec4<f32>(sqrt(saturate(color)), saturate(w)));
}

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

var<workgroup> cell_height: f32;
var<workgroup> cell_found: u32;
var<workgroup> cell_level: u32;
var<workgroup> cell_levels: u32;
var<workgroup> cell_low: atomic<u32>;
var<workgroup> cell_high: atomic<u32>;
var<workgroup> cell_seen: u32;

@compute @workgroup_size(64)
fn grow_cover(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let scale = grow.tier.z;
    let side = u32(scale);
    let spacing = grow.origin.w * scale;
    let cell = grow.grid.xy + vec2<i32>(group.xy);
    if any(cell < vec2<i32>(0)) {
        return;
    }
    let corner = vec2<f32>(cell) * spacing;
    let center = grow.origin.xz + corner + vec2<f32>(0.5 * spacing);
    let outer = grow.eye.w;
    let inner = grow.tier.x;
    let half_diagonal = 0.7072 * spacing;
    let across = distance(center, grow.eye.xz);
    if across > outer + half_diagonal {
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
    let cell_center = vec3<f32>(center.x, grow.origin.y + workgroupUniformLoad(&cell_height),
        center.y);
    let center_away = distance(cell_center, grow.eye.xyz);
    let nearest = max(center_away - half_diagonal, 0.0);
    if nearest > outer || center_away + 3.0 * half_diagonal < inner {
        return;
    }
    let law = CoverLaw(grow.tier.y);
    let held = f32(grow.grid.w);
    let tried = min(u32(grow.scale.y), u32(ceil(held * cover_share(nearest, law))));
    let finest = side * side;
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
            let tallest = grow.scale.w * (1.0 + COVER_CLUMP_HEIGHT);
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
        let within = index % finest;
        let candidate = index / finest;
        let at = cell * i32(side) + vec2<i32>(i32(within % side), i32(within / side));
        let cell_seed = hash((u32(at.x) * 0x8da6b343u) ^ (u32(at.y) * 0xd8163841u));
        let seed = hash(cell_seed + candidate * 0x632be5abu);
        let rank = f32(candidate) + unit(hash(seed ^ 0x165667b1u));
        let offset = vec2<f32>(unit(seed), unit(hash(seed ^ 0x5bd1e995u))) * grow.origin.w;
        let local = vec2<f32>(at) * grow.origin.w + offset;
        let world_xz = grow.origin.xz + local;
        if rank >= held * cover_share(distance(world_xz, grow.eye.xz), law) {
            continue;
        }
        let ground = true_ground(local);
        if !ground.found {
            continue;
        }
        let root = vec3<f32>(world_xz.x, grow.origin.y + ground.height - 0.02, world_xz.y);
        let away = distance(root, grow.eye.xyz);
        if away > outer || away < inner {
            continue;
        }
        let pick = unit(hash(seed ^ 0x27d4eb2fu));
        let share = f32((ground.control >> 16u) & 0xffu) * (1.0 / 255.0);
        let second = pick < share;
        let material = select(ground.control & 0xffu, (ground.control >> 8u) & 0xffu, second);
        let edge = smoothstep(0.0, 0.5, select(1.0 - share, share, second));
        let entry = min(grow.counts.z + material, PALETTE_ENTRIES - 1u);
        let spec = covers[entry];
        let falloff = cover_share(away, law);
        let own = spec.shape.x * grow.scale.x * area * falloff;
        if rank >= own * lushest {
            continue;
        }
        let colors = colors_at(local);
        let growth = cover_growth(colors.a);
        let keep = own * growth.density;
        if rank >= keep {
            continue;
        }
        let rise = saturate((keep - rank) / max(0.25 * keep, 1.0e-4));
        let pressed = min(pressed_at(world_xz), palette[entry].relief.y);
        let standing = 1.0 - 0.85 * smoothstep(0.01, 0.06, pressed);
        let clumped = cover_clumped(cover_clump(world_xz));
        let height = mix(spec.shape.y, spec.shape.z, unit(hash(seed ^ 0xc2b2ae35u)))
            * growth.height * clumped.height * rise * standing * edge;
        if height < 0.02 {
            continue;
        }
        if !in_view(root + vec3<f32>(0.0, 0.5 * height, 0.0), height) {
            continue;
        }
        let variation = (0.85 + 0.3 * unit(hash(seed ^ 0x9e3779b9u))) * macro_tint(colors, 1.0);
        var blade: CoverBlade;
        blade.root = root;
        let tuft = cover_tuft(away, height, spec.shape.w, law);
        blade.size = pack2x16float(vec2<f32>(height, tuft.x));
        blade.facing = u32(round(unit(hash(seed ^ 0x85ebca6bu)) * 65535.0));
        if grow.foliage.x > 0.0 {
            blade.facing |= grounded_tilt(ground_normal(local));
        }
        let lean = cover_clump_lean(world_xz);
        blade.leaves = u32(round(tuft.y * f32(COVER_LEAF_STEPS)))
            | (pack4x8unorm(vec4<f32>(0.0, lean * 0.5 + 0.5, 0.0)) & 0xffff00u);
        blade.root_color = pack_color(palette[entry].far.rgb * variation, 0.0);
        blade.tip_color = pack_color(spec.color.rgb * variation * clumped.tint, spec.color.w);
        if atomicAdd(&counts.appended, 1u) < grow.counts.w {
            if height * grow.view.x < STRAIGHT_PIXELS * away {
                blades[grow.counts.w - 1u - atomicAdd(&counts.straight, 1u)] = blade;
            } else {
                blades[atomicAdd(&counts.whole, 1u)] = blade;
            }
        }
    }
}

struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(8) var<storage, read_write> args: array<DrawArgs, 2>;

const JOINTS: u32 = 5u;
const STRAIGHT_JOINTS: u32 = 3u;
const STRAIGHT_FIRST_VERTEX: u32 = 1024u;
const STRAIGHT_PIXELS: f32 = 12.0;

@compute @workgroup_size(1)
fn finish_cover() {
    args[0] = DrawArgs(COVER_LEAVES * 2u * (JOINTS + 1u), atomicLoad(&counts.whole), 0u, 0u);
    args[1] = DrawArgs(COVER_LEAVES * 2u * (STRAIGHT_JOINTS + 1u), atomicLoad(&counts.straight),
        STRAIGHT_FIRST_VERTEX, 0u);
}
