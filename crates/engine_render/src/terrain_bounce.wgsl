// Terrain bounce light (`terrain_bounce.rs`), after the scene's
// declarations (`scene.wgsl`) and ground variety's (`terrain_variety.wgsl`).
// Every frame each terrain's ground is lit as terrain shades its own, the
// sun through the horizon maps and the clouds' shadows, the sky through its
// sky map, in its materials' colors tinted by its macro colors; the
// light's levels follow, each texel the mean of the four below it; then
// each probe around the camera gathers through its bins
// (`engine_terrain::bounce`) the light the ground sends it into the horizon
// maps' bounce layers, which `bounce_light` reads.

// Each bin's moments: the integral over it of the direction (x, y and z)
// and its solid angle (w).
struct BounceFrame {
    bins: array<vec4<f32>, BOUNCE_BINS>,
}

// One terrain's bounce light this frame (`gpu_layouts::bounce`): its first
// corner (world x and z) and its extent (meters along x and z); its probe
// columns along a side, its ground's texels along a side and the first
// column (x and z) of the window relighting; the window's columns along a
// side (0: it relights none), where its blocks' words and its probes start
// among every terrain's, and its place among the horizon records; its
// ground's first layer, its light's layer, its probes' first layer among
// the bounce layers and its sky map's layer plus one (0 without
// one); and where its materials start in the palette and its macro colors'
// layer plus one (0 without them).
struct BounceTerrain {
    place: vec4<f32>,
    grid: vec4<u32>,
    window: vec4<u32>,
    layers: vec4<u32>,
    materials: vec4<u32>,
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
// ground fills (low byte) and its distance's code.
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
    if terrain.window.x == 0u || id.x >= side || id.y >= side {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(side);
    let shape = terrain_horizon_records[HORIZON_RECORD_VECTORS * terrain.window.w + 1u];
    let height = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
        u32(shape.x) + HORIZON_LAYERS - 1u, 0.0).r;
    let across = terrain.place.xy + uv * terrain.place.zw;
    let world = vec3<f32>(across.x, shape.w + height, across.y);
    let surface = textureLoad(bounce_surface, id.xy, terrain.layers.x, 0);
    let tilt = vec2<f32>(textureLoad(bounce_surface, id.xy, terrain.layers.x + 1u, 0).xy)
        * (2.0 / 255.0) - 1.0;
    let normal = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
    let first = terrain.materials.x;
    let blend = f32(surface.z) / 255.0;
    var albedo = mix(bounce_albedo(first, surface.x), bounce_albedo(first, surface.y), blend);
    if terrain.materials.y > 0u {
        let vegetated = mix(bounce_vegetated(first, surface.x),
            bounce_vegetated(first, surface.y), blend);
        // The macro colors' level whose texels span the ground's.
        let lod = max(log2(f32(textureDimensions(bounce_macro).x) / f32(side)), 0.0);
        let colors = textureSampleLevel(bounce_macro, bounce_sampler, uv,
            terrain.materials.y - 1u, lod);
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
// moments into the light reaching the probe around a normal.
fn bounce_probe(terrain: BounceTerrain, record: u32, layer: u32, center: vec2<f32>)
        -> BounceProbe {
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
                    distance * sqrt(moment.w));
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

// Writes `probe` at texel `texel` of the bounce layers from `first`.
fn bounce_store(texel: vec2<u32>, first: u32, probe: BounceProbe) {
    textureStore(bounce_probes, texel, first, probe.mean);
    textureStore(bounce_probes, texel, first + 1u, probe.x);
    textureStore(bounce_probes, texel, first + 2u, probe.y);
    textureStore(bounce_probes, texel, first + 3u, probe.z);
}

// Each block of each relighting terrain's window: its probes' layers, each
// gathered once where one probe at the block's middle stands for its four
// columns and written to all four, or each column's own.
@compute @workgroup_size(8, 8, 1)
fn bounce_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let terrain = bounce_terrains[id.z];
    let span = terrain.window.x / 2u;
    if id.x >= span || id.y >= span {
        return;
    }
    let blocks = terrain.grid.x / 2u;
    let block = terrain.grid.zw / 2u + id.xy;
    let word = bounce_blocks[terrain.window.y + block.y * blocks + block.x];
    let first = terrain.window.z + (word >> 1u);
    let spacing = terrain.place.zw / f32(terrain.grid.x);
    let texel = id.xy * 2u;
    for (var layer = 0u; layer < BOUNCE_PROBE_LAYERS; layer++) {
        let layers = terrain.layers.z + layer * 4u;
        if (word & 1u) == 0u {
            let middle = vec2<f32>(block * 2u + 1u) * spacing;
            let probe = bounce_probe(terrain, first, layer, middle);
            for (var corner = 0u; corner < 4u; corner++) {
                bounce_store(texel + vec2<u32>(corner & 1u, corner >> 1u), layers, probe);
            }
        } else {
            for (var corner = 0u; corner < 4u; corner++) {
                let offset = vec2<u32>(corner & 1u, corner >> 1u);
                let center = (vec2<f32>(block * 2u + offset) + 0.5) * spacing;
                bounce_store(texel + offset, layers,
                    bounce_probe(terrain, first + corner, layer, center));
            }
        }
    }
}
