// Ground cover (decision 056): each frame, the blades within a radius of the
// eye, grown from a terrain's control values and heights. A workgroup takes
// one cell between level-0 samples; its lanes try the cell's candidate
// blades, each placed, sized and ranked by a hash of the cell and its
// index, so blades stay where they are as the eye moves. A candidate
// grows where its rank falls under its material's density at its point
// (both cover materials of a control value share it by their blend).
// Density falls with distance as the radius's fifth over the distance, so
// every ring of the disc holds about as many blades, which widen to cover
// as much ground; it thins out over the radius's outer half. Candidates
// near their threshold grow short, so blades rise as the eye nears instead
// of popping in. Survivors in the view append to the frame's blades; past
// capacity they are only counted.

struct Grow {
    // The terrain's origin (world) and its level-0 sample spacing.
    origin: vec4<f32>,
    // The eye (world) and the radius cover grows within.
    eye: vec4<f32>,
    // The first cell of the grid (terrain-local, in cells), the cells along
    // its side, and the candidates each cell tries.
    grid: vec4<i32>,
    // Where the terrain's levels start in the directory and how many there
    // are, where its materials start in the palette, and the blades'
    // capacity.
    counts: vec4<u32>,
    // The density scale and the candidates' share of the densest cover
    // (blades per candidate per square meter).
    scale: vec4<f32>,
    // The view's frustum planes (world).
    planes: array<vec4<f32>, 6>,
    // Runtime deformation's map (decision 083, `terrain_vertex.wgsl`'s
    // `deform_grid` and `deform`).
    deform_grid: vec4<i32>,
    deform: vec4<f32>,
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
@group(0) @binding(3) var<uniform> palette: array<PaletteEntry, 204>;
@group(0) @binding(4) var<storage, read> covers: array<CoverSpec>;
@group(0) @binding(5) var<uniform> grow: Grow;
@group(0) @binding(6) var<storage, read_write> blades: array<CoverBlade>;
@group(0) @binding(7) var<storage, read_write> counts: Counts;
@group(0) @binding(9) var<storage, read> deformation: array<u32>;

const NOT_RESIDENT: u32 = 0xffffffffu;
const TAU: f32 = 6.2831853;

fn hash(value: u32) -> u32 {
    var h = value * 0x9e3779b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn unit(value: u32) -> f32 {
    return f32(value >> 8u) * (1.0 / 16777216.0);
}

// What the terrain holds at terrain-local `local`, from the finest
// resident level: its height, and the nearest sample's control value.
struct Ground {
    height: f32,
    control: u32,
    found: bool,
}

fn ground_at(local: vec2<f32>) -> Ground {
    var ground: Ground;
    let base = grow.counts.x;
    let levels = grow.counts.y;
    // Nothing grows beyond the terrain's far edges.
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0 * grow.origin.w;
    if any(local > extent) {
        return ground;
    }
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (grow.origin.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = directory[directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        // Layers keep a border sample before the tile's first.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let a = textureLoad(heights, texel, layer, 0).r;
        let b = textureLoad(heights, texel + vec2<u32>(1u, 0u), layer, 0).r;
        let c = textureLoad(heights, texel + vec2<u32>(0u, 1u), layer, 0).r;
        let d = textureLoad(heights, texel + vec2<u32>(1u, 1u), layer, 0).r;
        ground.height = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        ground.control = textureLoad(control, texel + vec2<u32>(round(f)), layer, 0).x;
        ground.found = true;
        return ground;
    }
    return ground;
}

// How deep tracks (decision 083) sink the ground at world `point`
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

// The share of the densest cover growing `away` meters from the eye: all
// of it within a fifth of the radius, then the fifth over the distance,
// thinning to none over the radius's outer half.
fn cover_share(away: f32, radius: f32) -> f32 {
    let near = 0.2 * radius;
    return min(1.0, near / max(away, 1.0e-3)) * (1.0 - smoothstep(0.5 * radius, radius, away));
}

// The cell's ground at its center, read once for its lanes.
var<workgroup> cell_height: f32;
var<workgroup> cell_found: u32;

@compute @workgroup_size(64)
fn grow_cover(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let spacing = grow.origin.w;
    let cell = grow.grid.xy + vec2<i32>(group.xy);
    if any(cell < vec2<i32>(0)) {
        return;
    }
    let corner = vec2<f32>(cell) * spacing;
    let center = grow.origin.xz + corner + vec2<f32>(0.5 * spacing);
    let radius = grow.eye.w;
    let half_diagonal = 0.7072 * spacing;
    if distance(center, grow.eye.xz) > radius + half_diagonal {
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
    let nearest = max(distance(cell_center, grow.eye.xyz) - half_diagonal, 0.0);
    if nearest > radius {
        return;
    }
    let bound = cover_share(nearest, radius);
    let cell_seed = hash((u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u));
    for (var index = lane; index < u32(grow.grid.w); index += 64u) {
        let seed = hash(cell_seed + index * 0x632be5abu);
        let rank = unit(hash(seed ^ 0x165667b1u));
        if rank >= bound {
            continue;
        }
        let offset = vec2<f32>(unit(seed), unit(hash(seed ^ 0x5bd1e995u))) * spacing;
        let local = corner + offset;
        let world_xz = grow.origin.xz + local;
        let ground = ground_at(local);
        if !ground.found {
            continue;
        }
        let root = vec3<f32>(world_xz.x, grow.origin.y + ground.height - 0.02, world_xz.y);
        let away = distance(root, grow.eye.xyz);
        if away > radius {
            continue;
        }
        // The control value's two materials share the candidates by
        // their blend.
        let pick = unit(hash(seed ^ 0x27d4eb2fu));
        let share = f32((ground.control >> 16u) & 0xffu) * (1.0 / 255.0);
        let material = select(ground.control & 0xffu, (ground.control >> 8u) & 0xffu,
            pick < share);
        let entry = min(grow.counts.z + material, 255u);
        let spec = covers[entry];
        let falloff = cover_share(away, radius);
        let keep = spec.shape.x * grow.scale.x * grow.scale.y * falloff;
        if rank >= keep {
            continue;
        }
        let rise = saturate((keep - rank) / max(0.25 * keep, 1.0e-4));
        // Blades lie down in tracks (decision 083), as deep as the ground's
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
        let variation = 0.85 + 0.3 * unit(hash(seed ^ 0x9e3779b9u));
        var blade: CoverBlade;
        blade.root = root;
        blade.height = height;
        blade.facing = unit(hash(seed ^ 0x85ebca6bu)) * TAU;
        // Wider where fewer stand, covering as much ground.
        blade.width = spec.shape.w * min(1.0 / max(min(1.0, 0.2 * radius / away), 1.0e-3), 5.0);
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
