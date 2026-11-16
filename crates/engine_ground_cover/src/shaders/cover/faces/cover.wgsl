import package::engine::detail::seen_share;
import package::engine::view::{camera, stable_normalize};
import package::face::foliage::foliage;
import package::face::ground_tiles::PALETTE_ENTRIES;
import package::face::sun::sun_direction;
import package::generated::cover::{COVER_CARD_ROW, COVER_HEADER_ROWS, COVER_LIGHT_ROW,
    COVER_MATERIAL_ROWS, COVER_SEEN_ROW, COVER_SHAPE_ROW, CardRows, CoverCell, CoverHeader,
    CoverMaterial, LightRows, SEEN_ABOVE_HIGH, SEEN_ABOVE_LOW, SEEN_ABOVE_MIDDLE, SKY_COSINE,
    SeenRows, ShapeRows};
import package::place::cover::{CoverFar, CoverSite};

struct CoverSeen {
    tip: f32,
    occlusion: f32,
}

const COVER_NYQUIST: f32 = 0.5;
const COVER_EDGE_ON: f32 = 1.0e-3;
const COVER_GRAZING: f32 = 0.03;
const COVER_SALTS: u32 = 4u;
const COVER_ROUNDING: f32 = 1.0e-3;
const COVER_UNBOUND: f32 = 3.4028233e38;

const COVER_HEADER: u32 = COVER_MATERIAL_ROWS * PALETTE_ENTRIES;
const COVER_SHAPE: u32 = COVER_HEADER + COVER_SHAPE_ROW;
const COVER_LIGHT: u32 = COVER_HEADER + COVER_LIGHT_ROW;
const COVER_SEEN: u32 = COVER_HEADER + COVER_SEEN_ROW;
const COVER_CARD: u32 = COVER_HEADER + COVER_CARD_ROW;
const COVER_FIRST_TERRAIN: u32 = COVER_HEADER + COVER_HEADER_ROWS;
const COVER_PATCHES: u32 = 4u;
const COVER_EXTRA: u32 = 5u;
const COVER_KINDS: u32 = 6u;
const COVER_TUFT: u32 = 7u;
const COVER_FANS: u32 = 8u;
const COVER_LIMITS: u32 = COVER_MATERIAL_ROWS - 1u;
const COVER_NO_LAYER: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> cover_table: array<vec4<u32>>;

fn cover_header() -> CoverHeader {
    var header: CoverHeader;
    header.lush = bitcast<vec4<f32>>(cover_table[COVER_HEADER]);
    header.shading = bitcast<vec4<f32>>(cover_table[COVER_HEADER + 1u]);
    header.layers = bitcast<vec4<f32>>(cover_table[COVER_HEADER + 2u]);
    return header;
}

fn cover_shape() -> ShapeRows {
    var shape: ShapeRows;
    shape.bends = bitcast<vec4<f32>>(cover_table[COVER_SHAPE]);
    shape.arc = bitcast<vec4<f32>>(cover_table[COVER_SHAPE + 1u]);
    shape.tips = bitcast<vec4<f32>>(cover_table[COVER_SHAPE + 2u]);
    shape.steps = bitcast<vec4<f32>>(cover_table[COVER_SHAPE + 3u]);
    return shape;
}

fn cover_light() -> LightRows {
    var light: LightRows;
    light.glow = bitcast<vec4<f32>>(cover_table[COVER_LIGHT]);
    return light;
}

fn cover_merged(light: LightRows, away: f32) -> f32 {
    return smoothstep(light.glow.z, light.glow.w, away);
}

fn cover_bands() -> SeenRows {
    var bands: SeenRows;
    bands.along = bitcast<vec4<f32>>(cover_table[COVER_SEEN]);
    bands.tip = bitcast<vec4<f32>>(cover_table[COVER_SEEN + 1u]);
    return bands;
}

fn cover_card_look() -> CardRows {
    var cards: CardRows;
    cards.form = bitcast<vec4<f32>>(cover_table[COVER_CARD]);
    return cards;
}

fn cover_growth(lush: f32) -> f32 {
    return pow(bitcast<f32>(cover_table[COVER_HEADER].w), saturate(lush) - 0.5);
}

fn cover_tint(lush: f32, vegetated: f32) -> vec3<f32> {
    let tint = pow(bitcast<vec4<f32>>(cover_table[COVER_HEADER]).rgb,
        vec3<f32>(saturate(lush) - 0.5));
    return mix(vec3<f32>(1.0), tint, vegetated);
}

fn cover_hash(cell: vec2<i32>, salt: u32) -> u32 {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u)
        ^ ((salt + 1u) * 0x2c1b3c6du);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn cover_kept(size: f32, footprint: f32) -> f32 {
    return seen_share(COVER_NYQUIST * size / max(footprint, 1.0e-6));
}

fn cover_noise(at: vec2<f32>, size: f32, salt: u32) -> vec4<f32> {
    let p = at / size;
    let cell = vec2<i32>(floor(p));
    let f = p - floor(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = unpack4x8unorm(cover_hash(cell, salt));
    let b = unpack4x8unorm(cover_hash(cell + vec2<i32>(1, 0), salt));
    let c = unpack4x8unorm(cover_hash(cell + vec2<i32>(0, 1), salt));
    let d = unpack4x8unorm(cover_hash(cell + vec2<i32>(1, 1), salt));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y) * 2.0 - 1.0;
}

struct CoverPatches {
    density: f32,
    height: f32,
}

fn cover_patch(at: vec2<f32>, spec: CoverMaterial, footprint: f32, layer: u32)
        -> CoverPatches {
    var patches = CoverPatches(1.0, 1.0);
    let size = spec.patches.x;
    let coarse = cover_kept(size, footprint);
    if coarse <= 0.0 || all(spec.patches.yz == vec2<f32>(0.0)) {
        return patches;
    }
    let salt = layer * COVER_SALTS;
    var field = cover_noise(at, size, salt) * (coarse * (2.0 / 3.0));
    let fine = cover_kept(0.5 * size, footprint);
    if fine > 0.0 {
        field += cover_noise(at, 0.5 * size, salt + 1u) * (fine / 3.0);
    }
    patches.density = max(1.0 + spec.patches.y * field.x, 0.0);
    patches.height = max(1.0 + spec.patches.z * field.y, 0.0);
    return patches;
}

struct CoverPatchRange {
    density: f32,
    low: f32,
    high: f32,
}

fn cover_patch_point(at: u32, low: vec2<f32>, high: vec2<f32>, layer: u32, octave: u32,
        point: u32) -> vec4<f32> {
    let size = bitcast<f32>(cover_table[at + COVER_PATCHES].x) / f32(1u + octave);
    let first = floor(low / size);
    let last = floor(high / size);
    if any(last - first > vec2<f32>(1.0)) {
        return vec4<f32>(-1.0, -1.0, 1.0, 1.0);
    }
    let pick = vec2<u32>(point % 3u, point / 3u);
    let world = select(select(low, high, pick == vec2<u32>(1u)), last * size,
        (pick == vec2<u32>(2u)) & (last > first));
    let value = cover_noise(world, size, layer * COVER_SALTS + octave).xy;
    return vec4<f32>(value, value);
}

fn cover_patch_range(at: u32, coarse: vec4<f32>, fine: vec4<f32>) -> CoverPatchRange {
    let patches = bitcast<vec4<f32>>(cover_table[at + COVER_PATCHES]);
    let least = (2.0 * coarse.xy + fine.xy) / 3.0 - COVER_ROUNDING;
    let most = (2.0 * coarse.zw + fine.zw) / 3.0 + COVER_ROUNDING;
    return CoverPatchRange(max(1.0 + patches.y * most.x, 0.0),
        max(1.0 + patches.z * least.y, 0.0), max(1.0 + patches.z * most.y, 0.0));
}

fn cover_patch_reach(at: u32) -> CoverPatchRange {
    let patches = bitcast<vec4<f32>>(cover_table[at + COVER_PATCHES]);
    return CoverPatchRange(1.0 + patches.y, max(1.0 - patches.z, 0.0), 1.0 + patches.z);
}

struct CoverClump {
    center: vec2<f32>,
    seed: u32,
    cell: vec2<i32>,
    inside: f32,
}

fn cover_clump_seed(id: vec2<i32>, layer: u32) -> u32 {
    return cover_hash(id, layer * COVER_SALTS + 3u);
}

fn cover_clump_point(id: vec2<i32>, size: f32, layer: u32) -> vec2<f32> {
    return (vec2<f32>(id) + unpack4x8unorm(cover_hash(id, layer * COVER_SALTS + 2u)).xy) * size;
}

fn cover_clump(at: vec2<f32>, size: f32, layer: u32) -> CoverClump {
    let p = at / size;
    let cell = vec2<i32>(floor(p));
    let salt = layer * COVER_SALTS;
    var clump: CoverClump;
    var nearest = 8.0;
    var next = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let id = cell + vec2<i32>(x, y);
            let point = vec2<f32>(id) + unpack4x8unorm(cover_hash(id, salt + 2u)).xy;
            let gap = dot(p - point, p - point);
            if gap < nearest {
                next = nearest;
                nearest = gap;
                clump.center = point * size;
                clump.cell = id;
            } else {
                next = min(next, gap);
            }
        }
    }
    clump.seed = select(0u, cover_clump_seed(clump.cell, layer), nearest < 8.0);
    clump.inside = 0.5 * (sqrt(next) - sqrt(nearest)) * size;
    return clump;
}

struct CoverClumped {
    height: f32,
    tint: vec3<f32>,
}

fn cover_tall(clump: CoverClump, spec: CoverMaterial, scale: f32, footprint: f32) -> f32 {
    let soft = max(spec.patches.w * scale + footprint, COVER_ROUNDING);
    return (unpack4x8unorm(clump.seed).x * 2.0 - 1.0) * smoothstep(0.0, soft, clump.inside)
        * cover_kept(spec.clump.x, footprint);
}

fn cover_clumped(spec: CoverMaterial, tall: f32) -> CoverClumped {
    return CoverClumped(1.0 + spec.clump.y * tall, spec.shade.rgb + spec.drift.rgb * tall);
}

fn cover_view(normal: vec3<f32>, toward: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(abs(dot(normal, toward)),
        max(sqrt(max(1.0 - toward.y * toward.y, 0.0)), COVER_EDGE_ON));
}

fn cover_drawn(spec: CoverMaterial, scale: f32, error: f32, dense: f32, view: vec2<f32>,
        layers: f32) -> vec2<f32> {
    let height = 0.5 * (spec.shape.y + spec.shape.z) * scale;
    let stacked = dense * spec.shape.w * scale * height * view.y;
    let layered = select(1.0, layers * view.x / stacked, stacked > 0.0);
    return vec2<f32>(seen_share(height / error), layered);
}

fn cover_reached(depth: f32) -> vec4<f32> {
    let passed = exp(-max(depth, 1.0e-3)
        * vec4<f32>(1.0, SEEN_ABOVE_LOW, SEEN_ABOVE_MIDDLE, SEEN_ABOVE_HIGH));
    return (vec4<f32>(passed.yzw, 1.0) - passed) / (1.0 - passed.x);
}

fn cover_lit(depth: f32, fall: f32) -> f32 {
    let seen = max(depth, COVER_ROUNDING);
    let both = seen + fall;
    return seen * (1.0 - exp(-both)) / (both * (1.0 - exp(-seen)));
}

fn cover_sunward() -> f32 {
    return SKY_COSINE / max(sun_direction().y, COVER_ROUNDING);
}

fn cover_seen(depth: f32, fall: f32) -> CoverSeen {
    var seen: CoverSeen;
    seen.tip = dot(cover_reached(depth), bitcast<vec4<f32>>(cover_table[COVER_SEEN + 1u]));
    seen.occlusion = cover_lit(depth, fall);
    return seen;
}

fn cover_depth(area: f32, facing: f32) -> f32 {
    let taper = bitcast<f32>(cover_table[COVER_SHAPE + 1u].w);
    let risen = bitcast<f32>(cover_table[COVER_SHAPE + 3u].w);
    return 0.5 * taper / (1.0 + taper) * risen * area / max(facing, COVER_GRAZING);
}

fn cover_fall(leaves: f32) -> f32 {
    return cover_depth(leaves, 1.0) / SKY_COSINE;
}

fn cover_leaves(at: u32) -> f32 {
    return bitcast<f32>(cover_table[at].x) * bitcast<f32>(cover_table[at + COVER_EXTRA].x);
}

fn cover_look(at: u32) -> CoverMaterial {
    var spec: CoverMaterial;
    spec.shape = bitcast<vec4<f32>>(cover_table[at]);
    spec.tint = bitcast<vec4<f32>>(cover_table[at + 1u]);
    spec.clump = bitcast<vec4<f32>>(cover_table[at + 2u]);
    spec.shade = bitcast<vec4<f32>>(cover_table[at + 3u]);
    spec.patches = bitcast<vec4<f32>>(cover_table[at + 4u]);
    spec.extra = bitcast<vec4<f32>>(cover_table[at + COVER_EXTRA]);
    spec.accents = cover_table[at + COVER_KINDS];
    spec.tuft = bitcast<vec4<f32>>(cover_table[at + COVER_TUFT]);
    spec.fans = bitcast<vec4<f32>>(cover_table[at + COVER_FANS]);
    spec.form = bitcast<vec4<f32>>(cover_table[at + 9u]);
    spec.drift = bitcast<vec4<f32>>(cover_table[at + 10u]);
    spec.stand = bitcast<vec4<f32>>(cover_table[at + 11u]);
    spec.cells = cover_table[at + 12u];
    spec.lean = bitcast<vec4<f32>>(cover_table[at + 13u]);
    return spec;
}

fn cover_cell(at: u32) -> CoverCell {
    var cell: CoverCell;
    cell.axis = bitcast<vec4<f32>>(cover_table[at]);
    cell.span = bitcast<vec4<f32>>(cover_table[at + 1u]);
    return cell;
}

fn cover_row(base: u32, material: u32) -> u32 {
    return COVER_MATERIAL_ROWS * min(base + material, PALETTE_ENTRIES - 1u);
}

fn cover_limits(at: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(cover_table[at + COVER_LIMITS]);
}

fn cover_densest(at: u32) -> f32 {
    return bitcast<f32>(cover_table[at + COVER_FANS].w);
}

fn cover_share(at: u32, grown: f32, error: f32) -> f32 {
    let shape = bitcast<vec4<f32>>(cover_table[at]);
    let finer = max(error, 1.0e-6) * bitcast<f32>(cover_table[at + COVER_TUFT].w);
    return min(shape.w * grown / finer, 1.0)
        * seen_share(0.5 * (shape.y + shape.z) * grown / finer);
}

struct CoverBounds {
    dense: f32,
    room: f32,
    base: f32,
    share: f32,
}

fn cover_bound(at: u32, growth: vec4<f32>, patches: CoverPatchRange, error: f32) -> vec2<f32> {
    let shape = bitcast<vec4<f32>>(cover_table[at]);
    let tuft = bitcast<vec4<f32>>(cover_table[at + COVER_TUFT]);
    let pulled = tuft.z > 0.0 || bitcast<f32>(cover_table[at + COVER_FANS].y) > 0.0;
    let grown = select(growth.xy, growth.zw, pulled);
    let finer = max(error, 1.0e-6) * tuft.w;
    let least = grown.x * patches.low;
    let mean = 0.5 * (shape.y + shape.z);
    return vec2<f32>(patches.density * grown.y * cover_share(at, grown.y * patches.high, error),
        max(mean * least, 0.5 * finer) * max(shape.w * least, finer));
}

fn cover_bounds(at: u32, growth: vec4<f32>, patches: CoverPatchRange, error: f32)
        -> CoverBounds {
    var bounds = CoverBounds(0.0, COVER_UNBOUND, 0.0, 0.0);
    let dense = cover_dense(at);
    if dense <= 0.0 {
        return bounds;
    }
    let kinds = cover_table[at + COVER_KINDS];
    for (var index = 0u; index <= kinds.y; index++) {
        let kind = select(kinds.x + index * COVER_MATERIAL_ROWS, at, index == kinds.y);
        let bound = cover_bound(kind, growth, patches, error);
        if bound.x <= 0.0 {
            continue;
        }
        bounds.dense = max(bounds.dense, dense * bound.x);
        bounds.room = min(bounds.room, bound.y);
        if index == kinds.y {
            bounds.base = dense * bound.x;
        } else {
            bounds.share += bitcast<f32>(cover_table[kind + COVER_EXTRA].z);
        }
    }
    return bounds;
}

fn cover_dense(at: u32) -> f32 {
    return bitcast<f32>(cover_table[at].x);
}

fn cover_clump_size(at: u32) -> f32 {
    return bitcast<f32>(cover_table[at + 2u].x);
}

fn cover_stand(at: u32) -> f32 {
    return bitcast<f32>(cover_table[at + COVER_TUFT].x);
}

fn cover_layer(at: u32, layer: u32) -> u32 {
    let kinds = cover_table[at + COVER_KINDS];
    return select(kinds.z + layer * COVER_MATERIAL_ROWS, COVER_NO_LAYER, layer >= kinds.w);
}

fn cover_accent_row(kinds: vec4<u32>, pick: f32) -> u32 {
    var below = 0.0;
    for (var index = 0u; index < kinds.y; index++) {
        let at = kinds.x + index * COVER_MATERIAL_ROWS;
        below += bitcast<f32>(cover_table[at + COVER_EXTRA].z);
        if pick < below {
            return at;
        }
    }
    return COVER_NO_LAYER;
}

fn cover_beside(spec: CoverMaterial, index: u32) -> CoverMaterial {
    return cover_look(spec.accents.z + index * COVER_MATERIAL_ROWS);
}

fn cover_hidden(spec: CoverMaterial, weight: f32, dense: f32, scale: f32, error: f32,
        view: vec2<f32>, layers: f32, thin: f32) -> vec4<f32> {
    var shown = 0.0;
    if thin > 0.0 {
        let kept = cover_drawn(spec, scale, error * spec.tuft.w, dense, view, layers);
        shown = min(kept.x, kept.y) * thin;
    }
    let area = weight * (spec.shape.y + spec.shape.z) * spec.shape.w * scale * scale
        * (1.0 - shown);
    return vec4<f32>(area * spec.tint.rgb, area);
}

struct CoverCanopy {
    far: CoverFar,
    reached: vec4<f32>,
    height: f32,
    tilt: f32,
    depth: f32,
    fall: f32,
}

fn cover_faces(toward: vec3<f32>, tilt: f32) -> vec3<f32> {
    let level = 0.5 * (1.0 - tilt);
    let seen = toward * vec3<f32>(level, tilt, level);
    return seen * inverseSqrt(max(dot(toward, seen), COVER_ROUNDING));
}

fn cover_canopy(site: CoverSite, drawn: bool) -> CoverCanopy {
    var canopy: CoverCanopy;
    canopy.far.color = site.ground;
    canopy.far.occlusion = 1.0;
    canopy.far.sun = 1.0;
    let terrain = cover_table[COVER_FIRST_TERRAIN + site.terrain];
    let at = cover_row(terrain.x, site.material);
    if bitcast<f32>(cover_table[at].x) <= 0.0 {
        return canopy;
    }
    canopy.far.vegetated = 1.0;
    let header = cover_header();
    let first = cover_look(at);
    let thin = select(0.0, header.layers.w, drawn && bitcast<f32>(terrain.y) > 0.0);
    let growth = pow(header.lush.w, saturate(site.lush) - 0.5);
    let view = cover_view(site.normal, site.toward);
    let layers = header.layers.x;
    let densest = header.shading.x * growth * growth * growth * first.extra.w * view.y;
    if thin >= 1.0 && first.shade.w * growth >= site.error && layers * view.x >= densest {
        return canopy;
    }
    let patches = cover_patch(site.world.xz, first, site.footprint, 0u);
    let scale = growth * patches.height;
    let thick = header.shading.x * growth * patches.density;
    let dense = thick * first.shape.x;
    let accents = first.accents;
    var share = 1.0;
    var hidden = vec4<f32>(0.0);
    var shade = vec3<f32>(0.0);
    var drift = vec3<f32>(0.0);
    var lean = vec4<f32>(0.0);
    for (var index = 0u; index <= accents.y; index++) {
        let accent = index < accents.y;
        let spec = cover_look(select(at, accents.x + index * COVER_MATERIAL_ROWS, accent));
        var weight = share;
        if accent {
            weight = spec.extra.z;
            share -= weight;
        }
        let base = cover_look(at);
        let part = cover_hidden(spec, weight, dense, scale, site.error, view, layers, thin);
        hidden += part;
        shade += part.rgb * (spec.shade.rgb - base.shade.rgb);
        drift += part.rgb * (spec.drift.rgb - base.drift.rgb);
        lean += part.w * spec.lean;
    }
    for (var index = 0u; index < accents.w; index++) {
        let base = cover_look(at);
        let spec = cover_beside(base, index);
        let part = cover_hidden(spec, spec.shape.x / base.shape.x, thick * spec.shape.x, scale,
            site.error, view, layers, thin);
        hidden += part;
        shade += part.rgb * (spec.shade.rgb - base.shade.rgb);
        drift += part.rgb * (spec.drift.rgb - base.drift.rgb);
        lean += part.w * spec.lean;
    }
    if hidden.w <= 0.0 {
        return canopy;
    }
    let base = cover_look(at);
    var tall = 0.0;
    if cover_kept(base.clump.x, site.footprint) > 0.0 {
        tall = cover_tall(cover_clump(site.world.xz, base.clump.x, 0u), base, scale,
            site.footprint);
    }
    let clumped = cover_clumped(base, tall);
    let leaves = 0.5 * dense * clumped.height * clumped.height;
    canopy.depth = cover_depth(leaves * hidden.w, view.x);
    canopy.fall = cover_fall(leaves * base.extra.x * scale * scale);
    canopy.reached = cover_reached(canopy.depth);
    canopy.height = 0.5 * (base.shape.y + base.shape.z) * scale * clumped.height;
    canopy.tilt = dot(canopy.reached, lean) / hidden.w;
    let tips = max(hidden.rgb / hidden.w * clumped.tint + (shade + drift * tall) / hidden.w,
        vec3<f32>(0.0));
    canopy.far.color = site.ground
        * mix(vec3<f32>(1.0), tips, dot(canopy.reached, cover_bands().tip));
    canopy.far.occlusion = cover_lit(canopy.depth, canopy.fall);
    canopy.far.shown = 1.0 - exp(-canopy.depth);
    return canopy;
}

fn cover_far(site: CoverSite) -> CoverFar {
    let canopy = cover_canopy(site, true);
    var far = canopy.far;
    if far.shown <= 0.0 {
        return far;
    }
    let header = cover_header();
    let light = cover_light();
    let bands = cover_bands();
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let along = dot(canopy.reached, bands.along);
    let faces = cover_faces(site.toward, canopy.tilt);
    let bent = mix(faces, up, header.shading.y);
    let upright = stable_normalize(bent);
    var held = vec4<f32>(1.0);
    let blend = foliage().w;
    if blend > 0.0 {
        held = smoothstep(vec4<f32>(0.0), vec4<f32>(blend), bands.along * canopy.height);
    }
    let risen = dot(canopy.reached, held);
    far.normal = stable_normalize(mix(site.normal, upright, risen));
    let apart = upright - site.normal;
    let swing = (1.0 - header.shading.y) * (1.0 - header.shading.y)
        / max(dot(bent, bent), COVER_ROUNDING);
    let spread = (dot(canopy.reached, held * held) - risen * risen) * dot(apart, apart)
        + risen * risen * swing * (1.0 - dot(faces, faces));
    let near = 1.0 - cover_merged(light, distance(site.world, camera.position.xyz));
    let rough = header.shading.w * mix(1.0, header.shading.z, near * along);
    let alpha = rough * rough;
    far.roughness = sqrt(sqrt(min(alpha * alpha + spread, 1.0)));
    far.through = mix(light.glow.x, light.glow.y, along);
    far.sun = cover_lit(canopy.depth, canopy.fall * cover_sunward());
    return far;
}

fn cover_ground(site: CoverSite) -> CoverFar {
    return cover_canopy(site, false).far;
}
