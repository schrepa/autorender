// Terrain bounce light (`terrain_bounce.rs`), after the scene's
// declarations (`scene.wgsl`) and ground variety's (`terrain_variety.wgsl`).
// Every frame each terrain's ground is lit as terrain shades its own, the
// sun through the horizon maps and the clouds' shadows, the sky through its
// sky map, in its materials' colors tinted by its macro colors; the
// light's levels follow, each texel the mean of the four below it; then
// each level of probes around the detail eye, the finest first, gathers
// through its bins (`engine_terrain::bounce`) the light the ground sends it
// into its region of the horizon maps' bounce layers, which `bounce_light`
// reads. A coarser probe over four of the finer level's takes their mean.

// Each bin's moments: the integral over it of the direction (x, y and z)
// and its solid angle (w).
struct BounceFrame {
    bins: array<vec4<f32>, BOUNCE_BINS>,
}

// One terrain's bounce light at one level of its probes this frame
// (`gpu_layouts::bounce`): its first corner (world x and z) and its extent
// (meters along x and z); its finest level's probe columns along a side,
// its ground's texels along a side (0: it relights none), where its
// materials start in the palette and its macro colors' layer plus one (0
// without them); its ground's first layer, its light's layer, its probes'
// first layer among the bounce layers and its sky map's layer plus one (0
// without one); where its blocks' words, its probes and this level's mean
// probes (from the third level on) start among every terrain's, and its
// place among the horizon records; the level, its window's columns along
// a side (0: it relights none at this level) and the window's first column
// (x and z); the window's first texel (x and y) in the bounce layers and
// where its probes start in the scratch; and the finer level's window: its
// first column (x and z), its columns along a side (0: none relit) and
// where its probes start in the scratch.
struct BounceTerrain {
    place: vec4<f32>,
    grid: vec4<u32>,
    layers: vec4<u32>,
    sources: vec4<u32>,
    level: vec4<u32>,
    kept: vec4<u32>,
    finer: vec4<u32>,
}

// A surface material as the terrain palette holds it (`meshes/terrain.wgsl`):
// its maps' mean color, factors applied, in `far`, and its cover's tip
// color and leaves per square meter in `canopy`.
struct PaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    relief: vec4<f32>,
    canopy: vec4<f32>,
    placed: vec4<f32>,
    placed_size: vec4<f32>,
}

// A probe's light: the light reaching it alike from every way (red, green
// and blue) with the share of the sky it sees (alpha), then how both turn
// along x, y and z; light reaching a surface of normal `n` there is the
// first plus the others' dot with `n`.
struct BounceProbe {
    mean: vec4<f32>,
    x: vec4<f32>,
    y: vec4<f32>,
    z: vec4<f32>,
}

const BOUNCE_BINS: u32 = 26u;
const BOUNCE_PROBE_LAYERS: u32 = 2u;
const BOUNCE_RECORD_WORDS: u32 = 26u;
// A bin's distance code `c` stands for BOUNCE_NEAREST times 2 to the `c`
// over BOUNCE_STEPS meters.
const BOUNCE_NEAREST: f32 = 0.5;
const BOUNCE_STEPS: f32 = 16.0;
// As many as fit 16 KiB (`terrain::PALETTE_ENTRIES`).
const PALETTE_ENTRIES: u32 = 128u;

@group(0) @binding(60) var<uniform> bounce_frame: BounceFrame;
// Every slot's record at the level a dispatch relights.
@group(0) @binding(61) var<storage, read> bounce_terrains: array<BounceTerrain>;
// Each terrain's sky map, a layer each (`meshes/terrain_sky.wgsl`): the
// share of the sky its ground sees (red) and its normal's turn toward it
// (green, blue and alpha, from 0.5).
@group(0) @binding(62) var bounce_sky: texture_2d_array<f32>;
// Each terrain's ground, two layers in its first texels: its strongest
// material, the second and the second's share (red, green, blue); its
// normal's x and z from 128 (red, green).
@group(0) @binding(63) var bounce_surface: texture_2d_array<u32>;
// Each block's first probe, times two, plus one where its columns hold
// probes of their own; each probe's bins, two to a word: the share the
// ground fills (low byte) and its distance's code; after each terrain's
// probes, the means of those each probe of its third level on stands for.
@group(0) @binding(64) var<storage, read> bounce_blocks: array<u32>;
@group(0) @binding(65) var<storage, read> bounce_records: array<u32>;
// The ground's light, a layer per terrain in its first texels, read as a
// level or all of them and written a level at a time.
@group(0) @binding(66) var bounce_light_in: texture_2d_array<f32>;
@group(0) @binding(67) var bounce_light_out: texture_storage_2d_array<rgba16float, write>;
// The horizon maps' bounce layers (`scene.wgsl`'s `bounce_light`).
@group(0) @binding(68) var bounce_probes: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(69) var bounce_sampler: sampler;
@group(0) @binding(70) var<uniform> bounce_palette: array<PaletteEntry, PALETTE_ENTRIES>;
// Each terrain's macro colors with their mips (`meshes/terrain.wgsl`'s
// `terrain_macro`).
@group(0) @binding(71) var bounce_macro: texture_2d_array<f32>;
// Each relit level's probes this frame, in rows of its window, each
// layer's four vectors after the other's, which the next coarser level
// averages where it lies over them.
@group(0) @binding(72) var<storage, read_write> bounce_scratch: array<vec4<f32>>;

// Surface material `material` of the palette from `first`: its maps' mean
// color, and ground under cover its canopy, as it shows from afar: its
// blades' colors seen from above (`cover_seen`), half of them from their
// far sides, in the light their occlusion leaves.
fn bounce_albedo(first: u32, material: u32) -> vec3<f32> {
    let entry = bounce_palette[min(first + material, PALETTE_ENTRIES - 1u)];
    var color = entry.far.rgb;
    if entry.canopy.w > 0.0 {
        let seen = cover_seen(0.5 * entry.canopy.w);
        let canopy = mix(color, entry.canopy.rgb, seen.tip) * mix(1.0, COVER_FAR_SIDE, 0.5)
            * seen.occlusion;
        color = mix(color, canopy, 1.0 - exp(-entry.canopy.w));
    }
    return color;
}

// The share of surface material `material`'s ground, of the palette from
// `first`, that grows plants, as its canopy covers it from afar.
fn bounce_vegetated(first: u32, material: u32) -> f32 {
    let entry = bounce_palette[min(first + material, PALETTE_ENTRIES - 1u)];
    return select(0.0, 1.0 - exp(-entry.canopy.w), entry.canopy.w > 0.0);
}

// Each texel of each relighting terrain's ground: its materials' color,
// tinted by its macro colors as terrain's own ground where a pixel spans
// the texel (`macro_strength`), times the sun's light on it, wrapped about
// the terminator as terrain's own (`finish`), through its horizons and the
// clouds' shadows, and the sky's around its normal turned toward the sky it
// sees, by the share it sees.
@compute @workgroup_size(8, 8, 1)
fn bounce_ground(@builtin(global_invocation_id) id: vec3<u32>) {
    let terrain = bounce_terrains[id.z];
    let side = terrain.grid.y;
    if id.x >= side || id.y >= side {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(side);
    let shape = terrain_horizon_records[HORIZON_RECORD_VECTORS * terrain.sources.w + 1u];
    let height = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
        u32(shape.x) + HORIZON_LAYERS - 1u, 0.0).r;
    let across = terrain.place.xy + uv * terrain.place.zw;
    let world = vec3<f32>(across.x, shape.w + height, across.y);
    let surface = textureLoad(bounce_surface, id.xy, terrain.layers.x, 0);
    let tilt = vec2<f32>(textureLoad(bounce_surface, id.xy, terrain.layers.x + 1u, 0).xy)
        * (2.0 / 255.0) - 1.0;
    let normal = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
    let first = terrain.grid.z;
    let blend = f32(surface.z) / 255.0;
    var albedo = mix(bounce_albedo(first, surface.x), bounce_albedo(first, surface.y), blend);
    if terrain.grid.w > 0u {
        let vegetated = mix(bounce_vegetated(first, surface.x),
            bounce_vegetated(first, surface.y), blend);
        // The macro colors' level whose texels span the ground's.
        let lod = max(log2(f32(textureDimensions(bounce_macro).x) / f32(side)), 0.0);
        let colors = textureSampleLevel(bounce_macro, bounce_sampler, uv,
            terrain.grid.w - 1u, lod);
        let texel = max(terrain.place.z, terrain.place.w) / f32(side);
        albedo *= macro_tint(colors, vegetated, macro_strength(texel));
    }
    var seen = 1.0;
    var turn = vec3<f32>(0.0);
    if terrain.layers.w > 0u {
        let sky_texel = textureSampleLevel(bounce_sky, bounce_sampler, uv, terrain.layers.w - 1u,
            0.0);
        seen = sky_texel.r;
        turn = sky_texel.gba * 2.0 - 1.0;
    }
    var light = sky_average(normalize(normal + turn)) * (lighting.zenith.w * seen);
    let wrapped = max((dot(normal, lighting.sun.xyz) + 0.25) / 1.25, 0.0);
    if wrapped > 0.0 {
        light += lighting.sun_color.rgb
            * (wrapped * terrain_visibility(world, true) * cloud_passing(world));
    }
    textureStore(bounce_light_out, id.xy, terrain.layers.y, vec4<f32>(albedo * light, 1.0));
}

// Each texel of a level of the light: the mean of the four below it.
@compute @workgroup_size(8, 8, 1)
fn bounce_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(bounce_light_out);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let last = textureDimensions(bounce_light_in) - vec2<u32>(1u);
    let first = id.xy * 2u;
    var light = vec4<f32>(0.0);
    for (var corner = 0u; corner < 4u; corner++) {
        let at = min(first + vec2<u32>(corner & 1u, corner >> 1u), last);
        light += textureLoad(bounce_light_in, at, id.z, 0);
    }
    textureStore(bounce_light_out, id.xy, id.z, light * 0.25);
}

// The light `terrain`'s ground sends from terrain-local `local`, from the
// level whose texels span `footprint` meters, held within the terrain's own
// texels so no other's filter in.
fn bounce_ground_light(terrain: BounceTerrain, local: vec2<f32>, footprint: f32) -> vec3<f32> {
    let side = f32(terrain.grid.y);
    let texel = max(terrain.place.z, terrain.place.w) / side;
    let level = clamp(log2(max(footprint / texel, 1.0)), 0.0, log2(side));
    // Half the coarser level's texel inside the terrain's edges.
    let inside = 0.5 * exp2(ceil(level)) / side;
    let uv = clamp(local / terrain.place.zw, vec2<f32>(inside), vec2<f32>(1.0 - inside));
    let share = side / f32(textureDimensions(bounce_light_in).x);
    return textureSampleLevel(bounce_light_in, bounce_sampler, uv * share, terrain.layers.y,
        level).rgb;
}

// Layer `layer` of probe `record` of `terrain`, standing over
// terrain-local `center`: each bin's ground light, as much as the ground
// fills of it, and the sky's share of the rest, gathered by the bins'
// moments into the light reaching the probe around a normal. Each bin's
// ground is read over at least `least` meters, the spacing of the probes
// the record stands for.
fn bounce_probe(terrain: BounceTerrain, record: u32, layer: u32, center: vec2<f32>,
        least: f32) -> BounceProbe {
    var probe: BounceProbe;
    let first = record * BOUNCE_RECORD_WORDS + layer * (BOUNCE_BINS / 2u);
    for (var pair = 0u; pair < BOUNCE_BINS / 2u; pair++) {
        let word = bounce_records[first + pair];
        for (var side = 0u; side < 2u; side++) {
            let bits = word >> (16u * side);
            let moment = bounce_frame.bins[2u * pair + side];
            let filled = f32(bits & 0xffu) / 255.0;
            var light = vec3<f32>(0.0);
            if filled > 0.0 {
                // The ground lies that far along the bin's level direction.
                let distance = BOUNCE_NEAREST * exp2(f32((bits >> 8u) & 0xffu) / BOUNCE_STEPS);
                let level = length(moment.xz);
                let toward = select(vec2<f32>(0.0), moment.xz / max(level, 1.0e-6),
                    level > 1.0e-6);
                light = filled * bounce_ground_light(terrain, center + toward * distance,
                    max(distance * sqrt(moment.w), least));
            }
            let value = vec4<f32>(light, 1.0 - filled);
            probe.mean += value * (moment.w * (0.25 / PI));
            probe.x += value * (moment.x * (0.5 / PI));
            probe.y += value * (moment.y * (0.5 / PI));
            probe.z += value * (moment.z * (0.5 / PI));
        }
    }
    return probe;
}

// The light of probes `a` and `b` together.
fn bounce_sum(a: BounceProbe, b: BounceProbe) -> BounceProbe {
    return BounceProbe(a.mean + b.mean, a.x + b.x, a.y + b.y, a.z + b.z);
}

// A quarter of probe `a`'s light: four probes' sum, their mean.
fn bounce_quarter(a: BounceProbe) -> BounceProbe {
    return BounceProbe(a.mean * 0.25, a.x * 0.25, a.y * 0.25, a.z * 0.25);
}

// Where layer `layer` of the probe at column `local` of a window of
// `columns` columns a side whose probes start at `start` lies in the
// scratch.
fn bounce_slot(start: u32, columns: u32, local: vec2<u32>, layer: u32) -> u32 {
    return ((start + local.y * columns + local.x) * BOUNCE_PROBE_LAYERS + layer) * 4u;
}

// Writes `probe` at texel `texel` of the bounce layers from `first`.
fn bounce_store(texel: vec2<u32>, first: u32, probe: BounceProbe) {
    textureStore(bounce_probes, texel, first, probe.mean);
    textureStore(bounce_probes, texel, first + 1u, probe.x);
    textureStore(bounce_probes, texel, first + 2u, probe.y);
    textureStore(bounce_probes, texel, first + 3u, probe.z);
}

// Keeps layer `layer` of the probe at column `local` of `terrain`'s
// window, `probe`: in the scratch, for the next coarser level, and in the
// level's region of the bounce layers.
fn bounce_keep(terrain: BounceTerrain, local: vec2<u32>, layer: u32, probe: BounceProbe) {
    let at = bounce_slot(terrain.kept.z, terrain.level.y, local, layer);
    bounce_scratch[at] = probe.mean;
    bounce_scratch[at + 1u] = probe.x;
    bounce_scratch[at + 2u] = probe.y;
    bounce_scratch[at + 3u] = probe.z;
    bounce_store(terrain.kept.xy + local, terrain.layers.z + layer * 4u, probe);
}

// Layer `layer` of the finest level's probes of block `block`, the one at
// its middle (`word`'s low bit clear) or each column's own, at the
// columns of the block from `local` in the window.
fn bounce_block(terrain: BounceTerrain, block: vec2<u32>, local: vec2<u32>, layer: u32) {
    let blocks = terrain.grid.x / 2u;
    let word = bounce_blocks[terrain.sources.x + block.y * blocks + block.x];
    let first = terrain.sources.y + (word >> 1u);
    let spacing = terrain.place.zw / f32(terrain.grid.x);
    if (word & 1u) == 0u {
        let middle = vec2<f32>(block * 2u + 1u) * spacing;
        let probe = bounce_probe(terrain, first, layer, middle, 0.0);
        for (var corner = 0u; corner < 4u; corner++) {
            bounce_keep(terrain, local + vec2<u32>(corner & 1u, corner >> 1u), layer, probe);
        }
    } else {
        for (var corner = 0u; corner < 4u; corner++) {
            let offset = vec2<u32>(corner & 1u, corner >> 1u);
            let center = (vec2<f32>(block * 2u + offset) + 0.5) * spacing;
            bounce_keep(terrain, local + offset, layer,
                bounce_probe(terrain, first + corner, layer, center, 0.0));
        }
    }
}

// Layer `layer` of the second level's probe over block `block`, standing at
// its middle: the finest level's probe there, or the mean of its columns'
// own.
fn bounce_block_mean(terrain: BounceTerrain, block: vec2<u32>, layer: u32) -> BounceProbe {
    let blocks = terrain.grid.x / 2u;
    let word = bounce_blocks[terrain.sources.x + block.y * blocks + block.x];
    let first = terrain.sources.y + (word >> 1u);
    let spacing = terrain.place.zw / f32(terrain.grid.x);
    if (word & 1u) == 0u {
        return bounce_probe(terrain, first, layer, vec2<f32>(block * 2u + 1u) * spacing, 0.0);
    }
    var probe: BounceProbe;
    for (var corner = 0u; corner < 4u; corner++) {
        let center = (vec2<f32>(block * 2u + vec2<u32>(corner & 1u, corner >> 1u)) + 0.5)
            * spacing;
        probe = bounce_sum(probe, bounce_probe(terrain, first + corner, layer, center, 0.0));
    }
    return bounce_quarter(probe);
}

// Layer `layer` of the mean of the finer level's four probes from column
// `finer` of its window, as the scratch holds them.
fn bounce_finer_mean(terrain: BounceTerrain, finer: vec2<u32>, layer: u32) -> BounceProbe {
    var probe: BounceProbe;
    for (var corner = 0u; corner < 4u; corner++) {
        let at = bounce_slot(terrain.finer.w, terrain.finer.z,
            finer + vec2<u32>(corner & 1u, corner >> 1u), layer);
        probe = bounce_sum(probe, BounceProbe(bounce_scratch[at], bounce_scratch[at + 1u],
            bounce_scratch[at + 2u], bounce_scratch[at + 3u]));
    }
    return bounce_quarter(probe);
}

// Each probe of each relighting terrain's window at the level its records
// hold: the finest level's a block of four columns at a time, each gathered
// once where one probe at the block's middle stands for its four columns
// and written to all four, or each column's own. A coarser level's probe
// takes the mean of the finer level's four under it where the finer
// window holds them all; elsewhere the second level's gathers its block as
// the finest level would and takes their mean, and those from the third
// level on gather the means of the probes they stand for, each bin's ground
// read over at least their spacing.
@compute @workgroup_size(8, 8, 1)
fn bounce_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let terrain = bounce_terrains[id.z];
    let columns = terrain.level.y;
    let level = terrain.level.x;
    if level == 0u {
        if id.x >= columns / 2u || id.y >= columns / 2u {
            return;
        }
        let block = terrain.level.zw / 2u + id.xy;
        for (var layer = 0u; layer < BOUNCE_PROBE_LAYERS; layer++) {
            bounce_block(terrain, block, id.xy * 2u, layer);
        }
        return;
    }
    if id.x >= columns || id.y >= columns {
        return;
    }
    let column = terrain.level.zw + id.xy;
    let finer = vec2<i32>(column * 2u) - vec2<i32>(terrain.finer.xy);
    let held = terrain.finer.z > 0u && all(finer >= vec2<i32>(0))
        && all(finer + vec2<i32>(1) < vec2<i32>(i32(terrain.finer.z)));
    let spacing = terrain.place.zw / f32(terrain.grid.x >> level);
    let center = (vec2<f32>(column) + 0.5) * spacing;
    let record = terrain.sources.z + column.y * (terrain.grid.x >> level) + column.x;
    for (var layer = 0u; layer < BOUNCE_PROBE_LAYERS; layer++) {
        var probe: BounceProbe;
        if held {
            probe = bounce_finer_mean(terrain, vec2<u32>(finer), layer);
        } else if level == 1u {
            probe = bounce_block_mean(terrain, column, layer);
        } else {
            probe = bounce_probe(terrain, record, layer, center, max(spacing.x, spacing.y));
        }
        bounce_keep(terrain, id.xy, layer, probe);
    }
}
