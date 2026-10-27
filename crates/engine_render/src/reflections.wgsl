// Screen-space reflections, gone on through the world where the screen
// cannot see, prepended with scene.wgsl (the main passes' first group as
// terrain binds it, `bindings::CAMERA_TERRAIN`), effects.wgsl,
// terrain_variety.wgsl, terrain_ground.wgsl and screen_points.wgsl. Each
// reflective pixel's own ray walks the depth buffer's pyramid of nearest
// depths and meets what the screen shows where it passes within that
// surface's own span and thickness, keeping the screen's color there.
// Everything else goes on in the world from the last point the walk knew
// was clear: rays leaving the screen, rays passing behind a surface by more
// than it spans, which the eye cannot see, and rays that run out of steps.
// The world trace walks the world's own data: every terrain's resident
// tiles, each at its level, skipped over by their peaks, and the canopies
// of the placed objects their tiles hold, gridded by tile
// (`world_trace.rs`). A world hit is lit as the scene's own shading lights
// it, at the level its footprint matches, then hazed from the eye and
// graded as that shading shows it, as the screen's hits beside it are; a
// ray the world never meets leaves the sky the surface's own shading
// mirrors (`mirrored_sky`).

@group(1) @binding(0) var<uniform> post: Post;
@group(1) @binding(1) var depth_texture: texture_depth_2d;
// Each pixel's glow weight and reflectivity.
@group(1) @binding(3) var glow: texture_2d<f32>;
@group(1) @binding(7) var linear: sampler;
// The nearest depth of each 2x2 block of the depth buffer, then of each
// block of the level below, which rays skip across.
@group(1) @binding(10) var nearest_depth: texture_2d<f32>;
// The surface the opaque pass shaded (`surface_texel`: its shading normal,
// its roughness along the view and across it, the share of its light its
// mirrored environment makes up, and what lies behind it), and the scene
// with its levels, the first the scene itself at the render size and each
// next one half the one before, blurred for rough surfaces.
@group(1) @binding(13) var surface_texture: texture_2d<u32>;
@group(1) @binding(14) var scene_levels: texture_2d<f32>;
// The world the trace walks, as the terrain pass streams it: every resident
// tile's heights, a layer each, and the directory of every terrain's levels
// and their tiles' layers (`terrain_ground.wgsl` reads both by these
// names); each layer's peaks, the highest of its heights over each cell of
// 2^(m + 1) samples a side at mip m (`world_trace::peaks`); each layer's
// control values; every terrain's surface materials, as many as fit 16 KiB
// (`terrain::PALETTE_ENTRIES`); and each terrain's macro colors with their
// mips and its sky map, a layer each.
@group(1) @binding(20) var heights: texture_2d_array<f32>;
@group(1) @binding(21) var<storage, read> directory: array<u32>;
@group(1) @binding(22) var peaks: texture_2d_array<f32>;
@group(1) @binding(23) var control: texture_2d_array<u32>;
const PALETTE_ENTRIES: u32 = 113u;
@group(1) @binding(24) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(1) @binding(25) var macro_maps: texture_2d_array<f32>;
@group(1) @binding(26) var terrain_sky: texture_2d_array<f32>;
// The terrains the trace walks, as many as `post.reflections.w` counts;
// their placements (`world_trace.rs`): where draw groups' covers start,
// each atlas layer's word naming where its tile's grid of placed objects
// starts (NO_PLACED where it holds none), every draw group's cover, then
// the grids; and every placed object's record and pose, as the scatter pass
// set them on the ground this frame.
@group(1) @binding(27) var<storage, read> world_terrains: array<TraceTerrain>;
@group(1) @binding(28) var<storage, read> world_placed: array<u32>;
@group(1) @binding(29) var<storage, read> placed_records: array<Record>;
@group(1) @binding(30) var<storage, read> placed_poses: array<Pose>;

// A surface material as the terrain palette holds it (`terrain.wgsl`):
// its maps' mean color and roughness, its factors applied, in `far`, and
// its cover's canopy: its blades' tip color (linear) and their leaves' area
// per square meter of ground halfway lush (`canopy.w`, 0 without cover).
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

// A terrain's origin (world) and level-0 sample spacing; and where its
// levels start in the directory, how many there are, where its materials
// start in the palette and its macro colors' layer plus one (0 until they
// arrive), as `scatter_snap.wgsl` reads it.
struct TerrainInfo {
    origin: vec4<f32>,
    levels: vec4<u32>,
}

// A terrain as the world trace walks it (`world_trace.rs`): its place; the
// ground its surroundings' cutout leaves to the terrain they surround,
// terrain-local least x and z then greatest (none where they are equal);
// its sky map's layer plus one (0 until it arrives) and the levels whose
// tiles hold placed objects, a bit each; and the widest apron of those
// tiles' grids (meters), ground cover's density, and the highest its
// resident ground or a canopy stands (world).
struct TraceTerrain {
    info: TerrainInfo,
    cutout: vec4<f32>,
    maps: vec4<u32>,
    reach: vec4<f32>,
}

// `scatter_cull.wgsl`'s record: the bounding sphere, draw group and scale,
// then the terrain word (`scatter.rs`) and how far its base sinks.
struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    state: u32,
    pad: array<u32, 3>,
}

// A placed object's pose: the rows of its affine world matrix.
struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

const DEAD: u32 = 0xffffffffu;
// Objects standing far away while tiles holding them instead are resident
// (`scatter.rs`).
const HIDDEN: u32 = 0x80000000u;
// A layer's word in the placements whose tile holds no placed objects.
const NO_PLACED: u32 = 0xffffffffu;
// Quads along a tile's side; mips of each layer's peaks, cells of 2 samples
// a side at the finest up to one of the whole tile, and their cells along a
// side at the finest.
const TILE_QUADS: u32 = 256u;
const PEAK_MIPS: u32 = 8u;
const PEAK_CELLS: u32 = 128u;

// The most steps a reflection ray's walk of the depth pyramid takes: the
// ceiling it walked within before rays went on behind sheets and the
// ground, past which the world trace goes on from where it last knew the
// way clear.
const REFLECTION_STEPS: u32 = 96u;
// The most steps each walk of the world trace takes, over one terrain's
// ground or over the grids of one level of its placed objects: twice the
// cells of a tile's finest peaks along its side, a walk across a tile cell
// by cell climbing and descending as it goes.
const WORLD_STEPS: u32 = 2u * PEAK_CELLS;
// How far past a cell's edge, as a share of the cell, a walk of the world
// trace starts its next step, so it lies inside the next cell.
const WORLD_PAST: f32 = 1.0e-3;
// How thick the sheets reflection rays pass behind are (`BODY_THIN`), in
// world units: cutouts, foliage and blades stand for leaves, needles and
// grass a few centimeters thick.
const REFLECTION_THIN: f32 = 0.05;
// What lies behind a pixel's surface, which the surface target's last word
// holds above its mirrored share's MIRRORED_BITS (`meshes/shading.wgsl`'s
// `surface_texel`): an object's body, as deep as it is wide across where a
// ray passes behind it; a sheet; the ground, solid below its surface, which
// the world trace holds; or water, solid below its surface as far as
// reflections tell, which it does not.
const MIRRORED_BITS: u32 = 14u;
const BODY_SOLID: u32 = 0u;
const BODY_THIN: u32 = 1u;
const BODY_GROUND: u32 = 2u;
const BODY_WATER: u32 = 3u;
// A roughness lobe's half-width in radians per GGX alpha within the plane of
// incidence, the Phong lobe matching it narrowing to REFLECTION_LOBE_EDGE of
// its peak there (as in cone-traced reflections); across that plane, that
// times the cosine of the view with the normal, as slopes across the view
// turn the mirrored ray out of the plane by that much less than slopes
// along it turn it within (Bruneton, Neyret and Holzschuch 2010).
const REFLECTION_CONE: f32 = 1.19;
const REFLECTION_LOBE_EDGE: f32 = 0.244;
// The most taps a lobe is read with along its long axis on screen: as a
// texture unit's anisotropic filter takes at most as many (8x) along a
// footprint's longer side, past which they read a coarser level.
const REFLECTION_TAPS: u32 = 8u;
// The least cosine with the view a reflecting surface's shading normal
// keeps where the surface itself faces the eye at least as squarely: bumps
// turned past the view bend back to it, so their rays leave the surface.
const REFLECTION_FACING: f32 = 0.05;
// The share of a ray's reach at its end over which its reflection fades
// out, so its end shows no edge.
const REFLECTION_END: f32 = 0.1;
const FLOAT_MAX: f32 = 0x1.fffffep+127f;

// The nearest depth (the largest, reversed) of cell `cell` of mip `mip`:
// mip 0 is the depth buffer, mip m > 0 the pyramid's level m - 1, whose
// texels cover 2^m pixels a side (the pyramid repeats the drawn pixels'
// edge), within the cells it reduced: the render size over 2^m, rounded up.
fn nearest_depth_at(cell: vec2<i32>, mip: i32) -> f32 {
    if mip == 0 {
        let last = vec2<i32>(post.size.xy) - 1;
        return textureLoad(depth_texture, clamp(cell, vec2<i32>(0), last), 0);
    }
    let span = 1 << u32(mip);
    let last = max((vec2<i32>(post.size.xy) + span - 1) / span, vec2<i32>(1)) - 1;
    return textureLoad(nearest_depth, clamp(cell, vec2<i32>(0), last), mip - 1).r;
}

// Pixel position (rows downward) and depth of world point `world`, and its
// clip w.
fn screen_point(world: vec3<f32>) -> vec4<f32> {
    let clip = post.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>((ndc.x * 0.5 + 0.5) * post.size.x, (0.5 - ndc.y * 0.5) * post.size.y,
        ndc.z, clip.w);
}

// The unit normal at octahedral coordinates `encoded` in [-1, 1].
fn unfolded_normal(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    return normalize(vec3<f32>(encoded - shift, z));
}

// The world point at pixel position `pixel` (continuous) and depth `depth`.
fn world_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * post.size.zw;
    let world = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

// The surface the opaque pass shaded at a pixel (`surface_texel`'s words):
// its shading normal, its roughness along the view and across it, and the
// share of its light its mirrored environment makes up.
struct Mirroring {
    normal: vec3<f32>,
    roughness: vec2<f32>,
    mirrored: f32,
}

fn mirroring_at(pixel: vec2<i32>) -> Mirroring {
    let words = textureLoad(surface_texture, pixel, 0);
    let most = (1u << MIRRORED_BITS) - 1u;
    var mirroring: Mirroring;
    mirroring.normal = unfolded_normal(vec2<f32>(words.xy) / 65535.0 * 2.0 - 1.0);
    mirroring.roughness = unpack4x8unorm(words.z).xy;
    mirroring.mirrored = f32(words.w & most) / f32(most);
    return mirroring;
}

// What lies behind the surface the opaque pass shaded at `pixel`
// (`BODY_SOLID`, `BODY_THIN`, `BODY_GROUND` or `BODY_WATER`).
fn body_at(pixel: vec2<i32>) -> u32 {
    return textureLoad(surface_texture, pixel, 0).w >> MIRRORED_BITS;
}

// The scene at `at` (render pixels, continuous) as a lobe whose blur spans
// 2^`level` pixels reads it: the scene's level `level`, whose texels span
// as many pixels, mirrors reading the first, the scene itself at the render
// size, and wider lobes blending the two levels around theirs as trilinear
// filtering blends mipmaps.
fn blurred_scene(at: vec2<f32>, level: f32) -> vec3<f32> {
    return textureSampleLevel(scene_levels, linear, drawn_uv(at, post), level).rgb;
}

// The scene's light over a reflection's lobe where it lands about `hit`
// (render pixels, continuous), its half-lengths there `lengths` (pixels):
// along the ray's way on screen `way`, the plane of incidence seen edge on
// from the eye it passes through, and across it. Read at taps along the
// long axis, as many as it spans the short one's width (a pixel at least)
// over, up to REFLECTION_TAPS, each at the level whose blur spans the lobe
// across or the taps' spacing, whichever is wider, and weighted by the
// lobe's Gaussian profile, REFLECTION_LOBE_EDGE of its peak at the
// half-length. A tap counts only where what it shows stands above the
// plane the reflecting surface at `p` lies in (`plane`, its normal: none
// counts every tap) by more than a pixel spans there: the lobe's rays leave
// that plane, so they never meet what lies on it, the reflecting surface
// itself, or under it. Where no tap counts, the hit's own stands in.
fn lobe_light(hit: vec2<f32>, way: vec2<f32>, lengths: vec2<f32>, p: vec3<f32>,
        plane: vec3<f32>) -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let squared = dot(way, way);
    let along = select(vec2<f32>(0.0, 1.0), way * inverseSqrt(max(squared, 1.0e-30)),
        squared > 1.0e-12);
    let count = u32(clamp(ceil(lengths.x / max(lengths.y, 0.5)), 1.0, f32(REFLECTION_TAPS)));
    let spacing = 2.0 * lengths.x / f32(count);
    let level = log2(max(max(2.0 * lengths.y, spacing), 1.0));
    if count == 1u {
        return blurred_scene(hit, level);
    }
    let no_plane = all(plane == vec3<f32>(0.0));
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < count; tap++) {
        let offset = f32(2u * tap + 1u) / f32(count) - 1.0;
        let at = hit + along * (offset * lengths.x);
        let seen = world_at(clamp(vec2<i32>(floor(at)), vec2<i32>(0), last));
        let above = dot(seen.xyz - p, plane) > distance(seen.xyz, post.eye.xyz) / post.eye.w;
        if seen.w != 0.0 && !above && !no_plane {
            continue;
        }
        let weight = pow(REFLECTION_LOBE_EDGE, offset * offset);
        sum += blurred_scene(at, level) * weight;
        total += weight;
    }
    if total <= 0.0 {
        return blurred_scene(hit, level);
    }
    return sum / total;
}

// The depths the surface pixel `pixel` shows spans across it, in world
// units along the sight line through `at` from its point there `shown`, at
// its depth `depth`: where its neighbors along `axis` (a pixel) lie one
// nearer and one farther, a slope runs through it, spanning its smaller
// step to them, which a silhouette takes on its own side. Where both lie
// beyond it on one side (a crest: a thin feature, a silhouette's tip or a
// corner), beside sky or past the screen's edges, it spans none past its
// own depth.
fn surface_span(pixel: vec2<i32>, axis: vec2<i32>, depth: f32, at: vec2<f32>, shown: vec3<f32>)
        -> f32 {
    let before = nearest_depth_at(pixel - axis, 0);
    let after = nearest_depth_at(pixel + axis, 0);
    if before <= 0.0 || after <= 0.0 || (before - depth) * (after - depth) >= 0.0 {
        return 0.0;
    }
    let smaller = select(after, before, abs(before - depth) < abs(after - depth));
    return distance(world_point(at, smaller), shown);
}

// Whether the screen's hit at `at` (render pixels, continuous) is the
// surface at `pixel` reflecting itself, its point there `p` lying in the
// plane `plane` (its normal; none tells no hit apart from it): where the
// pixel the hit lands in shows `pixel`'s own body (`body_at`) and its point
// lies on that plane, measured off it as `lobe_light` measures its taps:
// the plane crosses the sight line through the point within the depths the
// surface there spans across its pixel (`surface_span`, the wider of its
// two axes'). Water reflecting water lies on its own plane. Every other
// hit, however near it lands, is what the ray meets, so a shore meets its
// mirror image at the waterline.
fn reflects_itself(pixel: vec2<i32>, p: vec3<f32>, plane: vec3<f32>, at: vec2<f32>) -> bool {
    let last = vec2<i32>(post.size.xy) - 1;
    let cell = clamp(vec2<i32>(floor(at)), vec2<i32>(0), last);
    if all(plane == vec3<f32>(0.0)) || body_at(cell) != body_at(pixel) {
        return false;
    }
    let depth = nearest_depth_at(cell, 0);
    let center = vec2<f32>(cell) + 0.5;
    let seen = world_point(center, depth);
    let spans = max(surface_span(cell, vec2<i32>(1, 0), depth, center, seen),
        surface_span(cell, vec2<i32>(0, 1), depth, center, seen));
    let sight = normalize(seen - post.eye.xyz);
    return abs(dot(seen - p, plane)) <= spans * abs(dot(sight, plane));
}

// Where a ray's walk of the depth pyramid ended: whether it met the scene
// on screen, and where (pixels and depth); otherwise the share of its
// segment on screen up to which it knew the way clear, from where the world
// trace goes on.
struct ScreenWalk {
    met: bool,
    point: vec3<f32>,
    clear: f32,
}

// The walk of a ray's segment on screen from `origin` along `ray` (pixels
// and depth, the whole segment) over the pyramid of nearest depths: a cell
// the ray crosses in front of everything is skipped and the next one is
// coarser; a cell it may meet is refined, down to single pixels (FidelityFX
// SSSR's hierarchical march), so every pixel along the ray near a surface
// is visited and thin objects are found alike from pixel to pixel. The
// surface a pixel shows is the ray's hit where the ray meets it there, or
// where the ray lies behind it by no more than its own length across the
// pixel, the depths the surface spans across it (`surface_span`) and what
// lies behind it (`body_at`): water all the way down, a sheet
// REFLECTION_THIN deep, an object's body as deep as it is wide across where
// the ray passes behind it, the ground no deeper than its span, the world
// trace holding what lies below. A ray behind an object by more than its
// body so far goes on at single pixels behind it. The walk knows the way
// clear up to where the ray last lay in front of everything; it hands over
// to the world trace there where the ray leaves the screen, passes behind a
// sheet or the ground by more than it spans, comes back in front of
// everything having passed behind objects by more than their bodies, or
// takes REFLECTION_STEPS steps.
fn screen_walk(origin: vec3<f32>, ray: vec3<f32>) -> ScreenWalk {
    var walk: ScreenWalk;
    let moving = abs(ray.xy) > vec2<f32>(1.0e-9);
    let forward = ray.xy >= vec2<f32>(0.0);
    let rate = select(vec2<f32>(1.0), ray.xy, moving);
    // Cell edges ahead of the ray, and a little past them so the next cell
    // starts inside.
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), forward);
    let past = select(vec2<f32>(-0.005), vec2<f32>(0.005), forward);
    let top = i32(textureNumLevels(nearest_depth));
    // The pixels the ray crosses along its longer axis, by which its step
    // across one of them is measured.
    let along = max(max(abs(ray.x), abs(ray.y)), 1.0e-6);
    // Leave the starting pixel first: a surface does not reflect itself.
    let exits = select(vec2<f32>(FLOAT_MAX), (floor(origin.xy) + ahead + past - origin.xy) / rate,
        moving);
    var t = min(exits.x, exits.y);
    var point = origin + ray * t;
    var mip = 0;
    walk.clear = min(t, 1.0);
    // Where on screen the ray began passing behind objects' bodies since it
    // last lay in front of all a cell holds; negative before it does.
    var entered = vec2<f32>(-1.0);
    for (var step = 0u; step < REFLECTION_STEPS; step++) {
        if t > 1.0 || any(point.xy < vec2<f32>(0.0)) || any(point.xy >= post.size.xy) {
            break;
        }
        let size = exp2(f32(mip));
        let cell = floor(point.xy / size);
        let surface = nearest_depth_at(vec2<i32>(cell), mip);
        let edges = select(vec2<f32>(FLOAT_MAX), ((cell + ahead) * size + past - origin.xy) / rate,
            moving);
        let leave = min(edges.x, edges.y);
        if surface < point.z {
            // In front of everything in the cell (reversed depth). Back in
            // front past objects it lay behind by more than their bodies,
            // the screen cannot tell what it met behind them.
            if entered.x >= 0.0 {
                break;
            }
            // Advance to where it leaves the cell (coarser next) or, heading
            // away from the eye, meets its nearest depth (finer next; in a
            // single pixel, the surface the ray meets).
            var meet = FLOAT_MAX;
            if ray.z < 0.0 {
                meet = (surface - origin.z) / ray.z;
            }
            t = min(leave, meet);
            point = origin + ray * t;
            walk.clear = min(t, 1.0);
            if leave < meet {
                mip = min(mip + 1, top);
            } else if mip > 0 {
                mip -= 1;
            } else {
                walk.met = t <= 1.0 && surface > 0.0;
                walk.point = point;
                return walk;
            }
            continue;
        }
        if mip > 0 {
            // Behind the cell's nearest depth: finer, down to single pixels.
            mip -= 1;
            continue;
        }
        // Behind the surface a pixel shows. Past the far plane over the sky
        // the screen holds nothing more.
        if surface <= 0.0 {
            break;
        }
        let here = vec2<i32>(cell);
        let body = body_at(here);
        if body == BODY_WATER {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        let ray_point = world_point(point.xy, point.z);
        let shown = world_point(point.xy, surface);
        // The ray's own length across the pixel, and the depths the surface
        // spans across it the way the ray came in (the edge it lies
        // nearest), which a ray crossing a sloping surface lies behind it by.
        let stride = distance(world_point(point.xy + ray.xy / along, point.z + ray.z / along),
            ray_point);
        let inside = select(1.0 - fract(point.xy), fract(point.xy), forward);
        let axis = select(vec2<i32>(0, 1), vec2<i32>(1, 0), inside.x < inside.y);
        let spans = stride + surface_span(here, axis, surface, point.xy, shown);
        var thickness = 0.0;
        if body == BODY_THIN {
            thickness = REFLECTION_THIN;
        }
        if body == BODY_SOLID {
            if entered.x < 0.0 {
                entered = point.xy;
            }
            thickness = distance(point.xy, entered) * distance(shown, post.eye.xyz) / post.eye.w;
        }
        if distance(ray_point, shown) <= thickness + spans {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        // Behind a sheet or the ground by more than it spans: the eye cannot
        // see what the ray meets there.
        if body != BODY_SOLID {
            break;
        }
        t = leave;
        point = origin + ray * t;
    }
    return walk;
}

// The meters along a ray from `start` (x and z), moving `rate` a meter, over
// which it lies within the box `low` to `high`, within `nearest` to
// `farthest`; empty (x not below y) where it misses it.
fn box_span(start: vec2<f32>, rate: vec2<f32>, low: vec2<f32>, high: vec2<f32>, nearest: f32,
        farthest: f32) -> vec2<f32> {
    let moving = abs(rate) > vec2<f32>(1.0e-12);
    let inverse = 1.0 / select(vec2<f32>(1.0), rate, moving);
    let a = (low - start) * inverse;
    let b = (high - start) * inverse;
    // Still along an axis, inside the box's span of it at every meter or at
    // none.
    let inside = (start >= low) & (start <= high);
    let enters = select(select(vec2<f32>(FLOAT_MAX), vec2<f32>(-FLOAT_MAX), inside), min(a, b),
        moving);
    let leaves = select(select(vec2<f32>(-FLOAT_MAX), vec2<f32>(FLOAT_MAX), inside), max(a, b),
        moving);
    return vec2<f32>(max(nearest, max(enters.x, enters.y)),
        min(farthest, min(leaves.x, leaves.y)));
}

// The meters along a ray at which its point `start + rate s` (in cells,
// `rate` of them a meter) leaves the square of `width` cells from `first`
// ahead of it, WORLD_PAST of a cell past the edge; past every meter where
// it does not move.
fn square_exit(start: vec2<f32>, rate: vec2<f32>, first: vec2<f32>, width: f32) -> f32 {
    let forward = rate >= vec2<f32>(0.0);
    let moving = abs(rate) > vec2<f32>(1.0e-12);
    let edge = first + select(vec2<f32>(-WORLD_PAST), vec2<f32>(width + WORLD_PAST), forward);
    let exits = select(vec2<f32>(FLOAT_MAX), (edge - start) / select(vec2<f32>(1.0), rate, moving),
        moving);
    return min(exits.x, exits.y);
}

// Terrain `info`'s extent from its origin (x and z): its level-0 tiles.
fn terrain_extent(info: TerrainInfo) -> vec2<f32> {
    return vec2<f32>(f32(directory[info.levels.x + 1u]), f32(directory[info.levels.x + 2u]))
        * (f32(TILE_QUADS) * info.origin.w);
}

// Whether terrain-local `local` lies in `cutout` (least x and z, then
// greatest; none where they are equal), the ground a terrain's surroundings
// leave to the terrain they surround.
fn in_cutout(local: vec2<f32>, cutout: vec4<f32>) -> bool {
    return cutout.z > cutout.x && all(local >= cutout.xy) && all(local <= cutout.zw);
}

// The finest level of `terrain` resident at terrain-local `local` from
// level `first` on, and its tile there, whose layer is NOT_RESIDENT where
// none is.
struct FinestTile {
    level: u32,
    tile: LevelTile,
}

fn finest_tile(terrain: GroundTerrain, local: vec2<f32>, first: u32) -> FinestTile {
    var found: FinestTile;
    found.tile.layer = NOT_RESIDENT;
    for (var level = first; level < terrain.levels; level++) {
        let tile = level_tile(terrain, level, local / (terrain.spacing * f32(1u << level)));
        if tile.layer != NOT_RESIDENT {
            found.level = level;
            found.tile = tile;
            return found;
        }
    }
    return found;
}

// What the world trace met: whether it met anything, the meters from the
// reflecting point it met it at (the ray's reach until it does), the
// surface's normal there; whether it is the ground, of which terrain (the
// trace's index) and the level of the tile holding it; and for a canopy,
// its objects' color.
struct WorldHit {
    met: bool,
    s: f32,
    normal: vec3<f32>,
    ground: bool,
    terrain: u32,
    level: u32,
    color: vec3<f32>,
}

// Where a ray from `o` (terrain-local) along `direction` meets the ground
// of `tile` over its 2x2 quads from sample `first` (its level's, `spacing`
// meters apart) within the meters `span`, on the two triangles every
// consumer splits a quad into: the normal of the smooth surface through the
// quad's corners there, and the meters (w; negative where it meets none). A
// ray starting under the ground meets it where it starts. Ground inside
// `cutout` is another terrain's.
fn ground_meet(tile: LevelTile, first: vec2<u32>, o: vec3<f32>, direction: vec3<f32>,
        span: vec2<f32>, spacing: f32, cutout: vec4<f32>) -> vec4<f32> {
    var h: array<f32, 9>;
    for (var z = 0u; z < 3u; z++) {
        for (var x = 0u; x < 3u; x++) {
            h[z * 3u + x] = tile_height(tile, first + vec2<u32>(x, z));
        }
    }
    let start = o.xz / spacing - vec2<f32>(first);
    let rate = direction.xz / spacing;
    var s = span.x;
    // A line crosses three of a 2x2 block's quads at most.
    for (var crossed = 0u; crossed < 3u; crossed++) {
        if s >= span.y {
            break;
        }
        let quad = min(vec2<u32>(max(floor(start + rate * s), vec2<f32>(0.0))), vec2<u32>(1u));
        let leave = min(square_exit(start, rate, vec2<f32>(quad), 1.0), span.y);
        let corner = quad.y * 3u + quad.x;
        let a = h[corner];
        let b = h[corner + 1u];
        let c = h[corner + 3u];
        let d = h[corner + 4u];
        let f0 = start - vec2<f32>(quad);
        // Where it crosses the diagonal from the quad's (0, 1) corner to its
        // (1, 0) one, between its triangles, each plane along the ray.
        let across = rate.x + rate.y;
        var split = leave;
        if abs(across) > 1.0e-12 {
            split = clamp((1.0 - f0.x - f0.y) / across, s, leave);
        }
        var low = s;
        for (var piece = 0u; piece < 2u; piece++) {
            let high = select(leave, split, piece == 0u);
            if high > low {
                let fa = clamp(f0 + rate * low, vec2<f32>(0.0), vec2<f32>(1.0));
                let fb = clamp(f0 + rate * high, vec2<f32>(0.0), vec2<f32>(1.0));
                let above_low = o.y + direction.y * low - triangle_height(a, b, c, d, fa);
                let above_high = o.y + direction.y * high - triangle_height(a, b, c, d, fb);
                if above_low <= 0.0 || above_high <= 0.0 {
                    var meet = low;
                    if above_low > 0.0 {
                        meet = low + (high - low) * above_low / (above_low - above_high);
                    }
                    let f = clamp(f0 + rate * meet, vec2<f32>(0.0), vec2<f32>(1.0));
                    let local = (vec2<f32>(first + quad) + f) * spacing;
                    if !in_cutout(local, cutout) {
                        let rise = vec2<f32>(mix(b - a, d - c, f.y), mix(c - a, d - b, f.x))
                            / spacing;
                        return vec4<f32>(normalize(vec3<f32>(-rise.x, 1.0, -rise.y)), meet);
                    }
                }
            }
            low = max(high, low);
        }
        s = leave;
    }
    return vec4<f32>(0.0, 1.0, 0.0, -1.0);
}

// Where the ray from `p` along `direction` (world) meets the ground of the
// trace's terrain `index` within the meters `bounds`, nearer than `hit`
// holds, which it then holds: on the triangles of the finest level resident
// at each point, as the ground draws it, walked through that level's tile's
// peaks. Coarser levels' tiles are resident wherever finer ones are, so a
// level the finest at a point is so over the finer level's whole tile
// there, a quarter of its own (all of it at level 0), within which the walk
// skips each cell the ray passes above the peak of, coarser next, and
// refines each it may meet, down to cells of 2x2 quads, whose triangles it
// tests.
fn trace_ground(index: u32, p: vec3<f32>, direction: vec3<f32>, bounds: vec2<f32>,
        hit: ptr<function, WorldHit>) {
    let terrain = world_terrains[index];
    let info = terrain.info;
    let ground = GroundTerrain(info.levels.x, info.levels.y, info.origin.w);
    let o = p - info.origin.xyz;
    let span = box_span(o.xz, direction.xz, vec2<f32>(0.0), terrain_extent(info), bounds.x,
        min(bounds.y, (*hit).s));
    var s = span.x;
    var steps = 0u;
    loop {
        let end = min(span.y, (*hit).s);
        if s >= end || steps >= WORLD_STEPS {
            break;
        }
        steps++;
        let at = o + direction * s;
        let found = finest_tile(ground, at.xz, 0u);
        if found.tile.layer == NOT_RESIDENT {
            // No level holds ground here: on past the coarsest level's tile.
            let size = f32(TILE_QUADS) * ground.spacing * f32(1u << (ground.levels - 1u));
            s = square_exit(o.xz / size, direction.xz / size, floor(at.xz / size), 1.0);
            continue;
        }
        let spacing = ground.spacing * f32(1u << found.level);
        // The tile's samples it is the finest over, a peak cell's of the
        // mip `top` (2^(top + 1) samples a side), from `region`.
        let quads = select(TILE_QUADS, TILE_QUADS / 2u, found.level > 0u);
        let top = select(PEAK_MIPS - 1u, PEAK_MIPS - 2u, found.level > 0u);
        let region = floor(at.xz / spacing / f32(quads)) * f32(quads);
        let offset = vec2<u32>(region) - found.tile.first;
        // The ray in the region's samples, and where it leaves the region.
        let start = o.xz / spacing - region;
        let rate = direction.xz / spacing;
        let leaves = min(square_exit(start, rate, vec2<f32>(0.0), f32(quads)), end);
        var mip = top;
        loop {
            if s >= leaves || steps >= WORLD_STEPS {
                break;
            }
            steps++;
            let width = f32(2u << mip);
            let cells = quads >> (mip + 1u);
            let cell = min(vec2<u32>(max(floor((start + rate * s) / width), vec2<f32>(0.0))),
                vec2<u32>(cells - 1u));
            let first = vec2<f32>(cell) * width;
            let peak = textureLoad(peaks, offset / (2u << mip) + cell, found.tile.layer, mip).r;
            let exit = min(square_exit(start, rate, first, width), leaves);
            if min(o.y + direction.y * s, o.y + direction.y * exit) > peak {
                s = exit;
                mip = min(mip + 1u, top);
                continue;
            }
            if mip > 0u {
                mip -= 1u;
                continue;
            }
            let met = ground_meet(found.tile, vec2<u32>(region + first), o, direction,
                vec2<f32>(s, exit), spacing, terrain.cutout);
            if met.w >= 0.0 {
                (*hit).met = true;
                (*hit).s = met.w;
                (*hit).normal = met.xyz;
                (*hit).ground = true;
                (*hit).terrain = index;
                (*hit).level = found.level;
                return;
            }
            s = exit;
            mip = min(mip + 1u, top);
        }
    }
}

// A draw group's canopy, from its cover's eight words in the placements
// (`occupancy::Cover`): its footprint's radius, the bottom and top of what
// it covers above its objects' origin (mesh units), and its objects' linear
// color.
struct Canopy {
    footprint: f32,
    bottom: f32,
    top: f32,
    color: vec3<f32>,
}

fn canopy_of(group: u32) -> Canopy {
    let at = world_placed[0] + 8u * group;
    var canopy: Canopy;
    canopy.footprint = bitcast<f32>(world_placed[at]);
    canopy.bottom = bitcast<f32>(world_placed[at + 1u]);
    canopy.top = bitcast<f32>(world_placed[at + 2u]);
    canopy.color = vec3<f32>(bitcast<f32>(world_placed[at + 4u]),
        bitcast<f32>(world_placed[at + 5u]), bitcast<f32>(world_placed[at + 6u]));
    return canopy;
}

// Where the ray from `p` along `direction` (world) meets the canopy of the
// placed object in slot `slot`, from `nearest` meters on, nearer than `hit`
// holds, which it then holds as terrain `index`'s: the spheroid its cover's
// footprint and heights bound about its origin at its scale, standing on
// the ground as the scatter pass set it this frame; a tree's crown, other
// objects' bounds. Dead slots and objects hidden while tiles holding them
// instead are resident meet nothing. A ray starting inside one meets it
// where it starts.
fn test_canopy(slot: u32, p: vec3<f32>, direction: vec3<f32>, nearest: f32,
        hit: ptr<function, WorldHit>, index: u32) {
    let record = placed_records[slot];
    if record.group == DEAD || (record.terrain & HIDDEN) != 0u {
        return;
    }
    let canopy = canopy_of(record.group);
    let radius = canopy.footprint * record.scale;
    let half = 0.5 * (canopy.top - canopy.bottom) * record.scale;
    if radius <= 0.0 || half <= 0.0 {
        return;
    }
    let pose = placed_poses[slot];
    let center = vec3<f32>(pose.r0.w,
        pose.r1.w + 0.5 * (canopy.bottom + canopy.top) * record.scale, pose.r2.w);
    // The spheroid as a sphere of `radius` once its heights scale by
    // `radius` over `half`.
    let squash = vec3<f32>(1.0, radius / half, 1.0);
    let from_center = (p - center) * squash;
    let way = direction * squash;
    let a = dot(way, way);
    let b = dot(from_center, way);
    let c = dot(from_center, from_center) - radius * radius;
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return;
    }
    let root = sqrt(discriminant);
    if (root - b) / a < nearest {
        return;
    }
    let s = max((-b - root) / a, nearest);
    if s >= (*hit).s {
        return;
    }
    (*hit).met = true;
    (*hit).s = s;
    (*hit).normal = normalize((p + direction * s - center) * squash * squash);
    (*hit).ground = false;
    (*hit).terrain = index;
    (*hit).color = canopy.color;
}

// Where the ray from `p` along `direction` (world; `o` terrain-local) meets
// a canopy of the grid of placed objects at word `block` of the placements,
// a tile's from `first` (terrain-local) `size` meters a side, within the
// meters `bounds`, nearer than `hit` holds (as terrain `index`'s): its
// square widened by its apron each way, in `cells` cells a side, each
// listing the objects whose canopies reach into it, below the highest of
// their tops (world heights), which a pyramid of cells, each the highest of
// the four below it, gathers. The walk skips each cell the ray passes
// above, coarser next, refines each it may meet, down to single cells,
// whose objects it tests, and ends once the nearest canopy met lies within
// the cell tested, or once the walks of the grids of its level took
// WORLD_STEPS steps (`steps`).
fn walk_block(block: u32, first: vec2<f32>, size: f32, o: vec3<f32>, p: vec3<f32>,
        direction: vec3<f32>, bounds: vec2<f32>, hit: ptr<function, WorldHit>, index: u32,
        steps: ptr<function, u32>) {
    let cells = world_placed[block];
    let apron = bitcast<f32>(world_placed[block + 1u]);
    let entries = block + world_placed[block + 2u];
    let width = (size + 2.0 * apron) / f32(cells);
    // The ray in cells from the grid's first corner.
    let start = (o.xz - (first - vec2<f32>(apron))) / width;
    let rate = direction.xz / width;
    let span = box_span(start, rate, vec2<f32>(0.0), vec2<f32>(f32(cells)), bounds.x,
        min(bounds.y, (*hit).s));
    let top = firstTrailingBit(cells);
    let ends = block + 4u + (4u * cells * cells - 1u) / 3u;
    var s = span.x;
    var mip = top;
    loop {
        if s >= min(span.y, (*hit).s) || *steps >= WORLD_STEPS {
            break;
        }
        *steps += 1u;
        let side = cells >> mip;
        let wide = f32(1u << mip);
        let cell = min(vec2<u32>(max(floor((start + rate * s) / wide), vec2<f32>(0.0))),
            vec2<u32>(side - 1u));
        let tops = block + 4u + 4u * (cells * cells - side * side) / 3u;
        let peak = bitcast<f32>(world_placed[tops + cell.y * side + cell.x]);
        let leave = min(square_exit(start, rate, vec2<f32>(cell) * wide, wide), span.y);
        if min(p.y + direction.y * s, p.y + direction.y * leave) > peak {
            s = leave;
            mip = min(mip + 1u, top);
            continue;
        }
        if mip > 0u {
            mip -= 1u;
            continue;
        }
        let at = cell.y * cells + cell.x;
        let listed = select(world_placed[ends + at - 1u], 0u, at == 0u);
        let listed_end = world_placed[ends + at];
        for (var entry = listed; entry < listed_end; entry++) {
            test_canopy(world_placed[entries + entry], p, direction, bounds.x, hit, index);
        }
        if (*hit).s <= leave {
            return;
        }
        s = leave;
        mip = min(mip + 1u, top);
    }
}

// How far across the ray its line `line` lies, as a share of the apron:
// its own (0), then the apron either way across it.
fn line_shift(line: u32) -> f32 {
    return select(select(1.0, -1.0, line == 1u), 0.0, line == 0u);
}

// Where the ray from `p` along `direction` (world) meets the canopy of a
// placed object of the trace's terrain `index` within the meters `bounds`,
// nearer than `hit` holds: at each level whose tiles hold placed objects,
// finest first, the grids of the tiles whose squares widened by the widest
// apron the ray passes through (`walk_block`), each once, in WORLD_STEPS
// steps a level. Those are the tiles one of three lines across the ground
// enters, the ray's own and the ray moved the apron either way across it:
// a tile, far wider than the apron, that the band between them reaches
// into, one of them crosses.
fn trace_canopies(index: u32, p: vec3<f32>, direction: vec3<f32>, bounds: vec2<f32>,
        hit: ptr<function, WorldHit>) {
    let terrain = world_terrains[index];
    let held = terrain.maps.y;
    if held == 0u {
        return;
    }
    let info = terrain.info;
    let o = p - info.origin.xyz;
    let apron = terrain.reach.x;
    let span = box_span(o.xz, direction.xz, vec2<f32>(-apron), terrain_extent(info) + apron,
        bounds.x, min(bounds.y, (*hit).s));
    if span.x >= span.y {
        return;
    }
    let flat = length(direction.xz);
    let side = select(vec2<f32>(1.0, 0.0), vec2<f32>(-direction.z, direction.x) / flat,
        flat > 1.0e-6);
    for (var level = 0u; level < info.levels.y; level++) {
        if ((held >> level) & 1u) == 0u {
            continue;
        }
        var steps = 0u;
        let size = f32(TILE_QUADS) * info.origin.w * f32(1u << level);
        let header = info.levels.x + 4u * level;
        let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
        let rate = direction.xz / size;
        for (var line = 0u; line < 3u; line++) {
            let start = (o.xz + side * (apron * line_shift(line))) / size;
            var s = span.x;
            loop {
                if s >= min(span.y, (*hit).s) || steps >= WORLD_STEPS {
                    break;
                }
                steps++;
                let tile = floor(start + rate * s);
                let leave = square_exit(start, rate, tile, 1.0);
                // Each tile once, by the first line entering it.
                var fresh = all(tile >= vec2<f32>(0.0)) && all(tile < vec2<f32>(tiles));
                for (var earlier = 0u; earlier < line; earlier++) {
                    let before = (o.xz + side * (apron * line_shift(earlier))) / size;
                    let entered = box_span(before, rate, tile, tile + 1.0, span.x, span.y);
                    fresh = fresh && entered.x >= entered.y;
                }
                if fresh {
                    let layer = directory[directory[header] + u32(tile.y) * tiles.x
                        + u32(tile.x)];
                    if layer != NOT_RESIDENT && world_placed[1u + layer] != NO_PLACED {
                        walk_block(world_placed[1u + layer], tile * size, size, o, p,
                            direction, span, hit, index, &steps);
                    }
                }
                s = leave;
            }
        }
    }
}

// What the ray from `p` along `direction` (world) meets from `nearest` to
// `farthest` meters on: the nearest ground or canopy of the terrains the
// trace walks, each walked only while a ray rising lies below the highest
// its ground or a canopy stands.
fn world_trace(p: vec3<f32>, direction: vec3<f32>, nearest: f32, farthest: f32) -> WorldHit {
    var hit: WorldHit;
    hit.s = farthest;
    let count = u32(post.reflections.w);
    for (var index = 0u; index < count; index++) {
        var below = hit.s;
        if direction.y > 0.0 {
            below = min(below, (world_terrains[index].reach.z - p.y) / direction.y);
        }
        if below > nearest {
            trace_ground(index, p, direction, vec2<f32>(nearest, below), &hit);
            trace_canopies(index, p, direction, vec2<f32>(nearest, below), &hit);
        }
    }
    return hit;
}

// Terrain `terrain`'s macro colors at terrain-local `local` at the mip whose
// texels match a footprint `footprint` meters wide, interpolated between
// its texels; ground as it is before they arrive (`macro_tint`).
fn ground_macro(terrain: TraceTerrain, local: vec2<f32>, footprint: f32) -> vec4<f32> {
    let layer = terrain.info.levels.w;
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let extent = terrain_extent(terrain.info);
    let finest = vec2<f32>(textureDimensions(macro_maps, 0));
    let texel = max(extent.x / finest.x, extent.y / finest.y);
    let coarsest = f32(textureNumLevels(macro_maps) - 1u);
    let mip = u32(clamp(floor(log2(max(footprint / texel, 1.0))), 0.0, coarsest));
    let size = vec2<i32>(textureDimensions(macro_maps, mip));
    let at = local / extent * vec2<f32>(size) - 0.5;
    let corner = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = size - vec2<i32>(1);
    let a = textureLoad(macro_maps, clamp(corner, vec2<i32>(0), last), layer - 1u, mip);
    let b = textureLoad(macro_maps, clamp(corner + vec2<i32>(1, 0), vec2<i32>(0), last),
        layer - 1u, mip);
    let c = textureLoad(macro_maps, clamp(corner + vec2<i32>(0, 1), vec2<i32>(0), last),
        layer - 1u, mip);
    let d = textureLoad(macro_maps, clamp(corner + vec2<i32>(1, 1), vec2<i32>(0), last),
        layer - 1u, mip);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Terrain `terrain`'s sky map at terrain-local `local` once it arrived
// (`meshes/terrain_sky.wgsl`'s `terrain_sky_map`): the share of the sky its
// ground sees (red), and how its normal turns toward the middle of the sky
// it sees (green, blue and alpha, from 0.5); before, all the sky and no
// turn.
fn ground_sky(terrain: TraceTerrain, local: vec2<f32>) -> vec4<f32> {
    let layer = terrain.maps.x;
    if layer == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 0.5);
    }
    return textureSampleLevel(terrain_sky, linear, local / terrain_extent(terrain.info), layer - 1u,
        0.0);
}

// The ground's color at terrain-local `local` of `terrain`, facing
// `normal`, met along `toward` by a footprint `footprint` meters wide, its
// finest resident level there `finest`: at the level whose samples match
// the footprint, or the finest resident from there, the nearest sample's
// two strongest materials' mean colors blended by their shares, each with
// its cover's canopy as the ground shows it where tufts retire, as many
// leaves as lush ground grows, which a view crossing them at its angle
// meets (`terrain_variety.wgsl`'s `cover_seen`), tinted by the macro colors
// as the ground is.
fn ground_albedo(terrain: TraceTerrain, local: vec2<f32>, normal: vec3<f32>, toward: vec3<f32>,
        footprint: f32, finest: u32) -> vec3<f32> {
    let info = terrain.info;
    let ground = GroundTerrain(info.levels.x, info.levels.y, info.origin.w);
    let asked = u32(clamp(floor(log2(max(footprint / ground.spacing, 1.0))), 0.0,
        f32(ground.levels - 1u)));
    var found = finest_tile(ground, local, max(finest, asked));
    if found.tile.layer == NOT_RESIDENT {
        found = finest_tile(ground, local, finest);
    }
    let samples = local / (ground.spacing * f32(1u << found.level));
    let texel = min(vec2<u32>(round(max(samples, vec2<f32>(0.0)))) - found.tile.first,
        vec2<u32>(TILE_QUADS)) + vec2<u32>(1u);
    let value = textureLoad(control, texel, found.tile.layer, 0).x;
    let share = f32((value >> 16u) & 0xffu) / 255.0;
    let colors = ground_macro(terrain, local, footprint);
    let growth = cover_growth(colors.a);
    let sine = max(abs(dot(toward, normal)), 0.03);
    var albedo = vec3<f32>(0.0);
    var vegetated = 0.0;
    for (var slot = 0u; slot < 2u; slot++) {
        let weight = select(share, 1.0 - share, slot == 0u);
        let material = info.levels.z + ((value >> (8u * slot)) & 0xffu);
        let entry = palette[min(material, PALETTE_ENTRIES - 1u)];
        var color = entry.far.rgb;
        if entry.canopy.w > 0.0 {
            // As the ground's canopy shows (`meshes/terrain.wgsl`'s
            // `terrain_canopy_depth`): G = 0.5 of the leaves across the
            // view's way, over the sine of its angle to the ground.
            let depth = 0.5 * entry.canopy.w * growth.height * growth.density * terrain.reach.y
                / sine;
            if depth > 0.0 {
                let look = mix(color, entry.canopy.rgb, cover_seen(depth).tip);
                color = mix(color, look, 1.0 - exp(-depth));
            }
            vegetated += weight;
        }
        albedo += weight * color;
    }
    return albedo * macro_tint(colors, vegetated);
}

// The light the world hit `hit` of a ray from `p` along `direction` sends
// back along it, a footprint `footprint` meters wide, as the scene's own
// shading lights it (`meshes/shading.wgsl`'s `finish_reflecting`): its
// color (the ground's, or a canopy's objects'); the sky's light about its
// normal, the ground's turned toward the sky it sees and by the share of it
// it sees (its sky map), with the light the ground around sends where it
// hides the sky (terrains' bounce probes); and the sun through its slope,
// wrapped as the ground's own light is, past the terrains' horizons, the
// cheapest of the sun's shadows, where shadows run, and the clouds'
// shadows. The haze covers it and the grade follows exactly as the scene's
// own shading shows that point to the eye (`finish`'s `atmospheric` from
// the camera, then `grade`), as the screen's hits beside it show theirs.
fn world_light(hit: WorldHit, p: vec3<f32>, direction: vec3<f32>, footprint: f32) -> vec3<f32> {
    let terrain = world_terrains[hit.terrain];
    let world = p + direction * hit.s;
    let normal = hit.normal;
    var albedo = hit.color;
    var ambient: vec3<f32>;
    var bounce: vec3<f32>;
    if hit.ground {
        let local = world.xz - terrain.info.origin.xz;
        albedo = ground_albedo(terrain, local, normal, direction, footprint, hit.level);
        let sky = ground_sky(terrain, local);
        ambient = sky_average(normalize(normal + (sky.gba * 2.0 - 1.0))) * sky.r;
        bounce = bounce_light(world, normal, true).light;
    } else {
        let around = bounce_light(world, normal, false);
        ambient = mix(sky_average(normal), sky_above(normal) * around.sky, around.weight);
        bounce = around.light * around.weight;
    }
    let sky_light = albedo * (ambient * lighting.zenith.w + bounce);
    let sun = lighting.sun.xyz;
    let wrapped = max((dot(normal, sun) + 0.25) / 1.25, 0.0);
    var sun_light = vec3<f32>(0.0);
    if wrapped > 0.0 {
        var visible = cloud_passing(world);
        if SHADOW_MAPS && shadow.state.x != 0.0 {
            visible *= terrain_visibility(world, hit.ground);
        }
        sun_light = albedo * lighting.sun_color.rgb * (wrapped * visible);
    }
    let eye = camera.position.xyz;
    return grade(atmospheric(sky_light + sun_light, eye, world, stable_normalize(world - eye)).rgb);
}

// How much of a reflection whose ray went `travelled` of its reach `reach`
// shows: all of it, fading out over the last REFLECTION_END of the reach.
fn reach_fade(travelled: f32, reach: f32) -> f32 {
    return saturate((1.0 - min(travelled / max(reach, 1.0e-6), 1.0)) / REFLECTION_END);
}

// Reflections at every pixel, every frame: from each reflective pixel (the
// glow target's second channel), its own ray along the mirror direction of
// its own shading normal reaches the reflections' distance. Its segment on
// screen, ending nine tenths of the way to the eye plane where it heads
// toward the camera, is walked over the depth pyramid (`screen_walk`); a
// hit there takes the scene over the lobe its own roughness spreads the
// ray into (`lobe_light`): long along the ray's way on screen, short
// across it, so grazing views draw it out into streaks. Every other ray
// goes on through the world from where the walk last knew the way clear to
// its whole reach (`world_trace`), behind the camera too, and takes the
// light the world sends back from what it meets (`world_light`). A hit
// keeps its whole weight wherever it lands, however near, fading only
// toward the end of the ray's reach; a screen hit on the surface reflecting
// itself (`reflects_itself`) weighs nothing, as does a ray meeting nothing,
// leaving the sky the surface's own shading mirrors (`finish`'s
// `mirrored_sky`, exact where it is smooth). The world trace holds no
// water, so a world hit is never the surface itself. It writes, both
// premultiplied by how surely the ray met the scene, the light the
// surface's reflectivity mirrors, and the share of the surface's own light
// its mirrored environment made up, which that light replaces
// (`reflected`).
@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let reflectivity = textureLoad(glow, pixel, 0).g;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    // The normal the lighting shaded (with normal maps and waves) and the
    // roughness, bent back toward the eye where bumps turn it past the view
    // so the ray leaves the surface: to REFLECTION_FACING or, where the
    // surface the depth buffer holds (`plane`, its normal) faces the eye less
    // squarely, as squarely as it does, so surfaces seen grazing keep their
    // own mirror image.
    let mirroring = mirroring_at(pixel);
    let to_eye = normalize(post.eye.xyz - p);
    let plane = depth_normal(pixel, p);
    let facing = min(REFLECTION_FACING, max(dot(plane, to_eye), 0.0));
    var normal = mirroring.normal;
    normal = normalize(normal + max(facing - dot(normal, to_eye), 0.0) * to_eye);
    let direction = reflect(-to_eye, normal);
    let reach = post.reflections.x;
    let start = screen_point(p);
    let rows = transpose(post.view_projection);
    let w_rate = dot(rows[3].xyz, direction);
    var shown = reach;
    if w_rate < 0.0 {
        shown = min(reach, 0.9 * start.w / -w_rate);
    }
    let end = screen_point(p + direction * shown);
    if !(start.w > 0.0 && end.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let origin = start.xyz;
    let ray = end.xyz - start.xyz;
    let alphas = mirroring.roughness * mirroring.roughness;
    let walk = screen_walk(origin, ray);
    if walk.met {
        if reflects_itself(pixel, p, plane, walk.point.xy) {
            return vec4<f32>(0.0);
        }
        let reached = world_point(walk.point.xy, walk.point.z);
        let travelled = distance(reached, p);
        let weight = reach_fade(travelled, reach);
        // The light reflected over the roughness lobe: its half-lengths
        // where it lands, in pixels, along the plane of incidence and across
        // it (REFLECTION_CONE), the cosine of the view with the normal
        // narrowing it across, so grazing views draw it out along the ray's
        // way.
        let pixels = travelled * post.eye.w / max(distance(reached, post.eye.xyz), 1.0e-4);
        let lengths = REFLECTION_CONE * pixels
            * vec2<f32>(alphas.x, alphas.y * max(dot(normal, to_eye), 0.0));
        let color = lobe_light(walk.point.xy, ray.xy, lengths, p, plane);
        return vec4<f32>(color * (reflectivity * weight), mirroring.mirrored * weight);
    }
    let clear = origin + ray * walk.clear;
    let nearest = distance(world_point(clear.xy, clear.z), p);
    if !(nearest < reach) {
        return vec4<f32>(0.0);
    }
    let hit = world_trace(p, direction, nearest, reach);
    if !hit.met {
        return vec4<f32>(0.0);
    }
    let weight = reach_fade(hit.s, reach);
    // The footprint where it lands: as wide as a pixel spans over the way
    // the ray went, or as the roughness's lobe spreads there, whichever is
    // wider.
    let footprint = max((distance(p, post.eye.xyz) + hit.s) / post.eye.w,
        2.0 * REFLECTION_CONE * alphas.x * hit.s);
    let color = world_light(hit, p, direction, footprint);
    return vec4<f32>(color * (reflectivity * weight), mirroring.mirrored * weight);
}
