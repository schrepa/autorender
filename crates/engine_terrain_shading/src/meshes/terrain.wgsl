import package::face::dither::{encoded, stored};
import package::face::heightfield::{COVER_FAR_SIDE, COVER_ROUGHNESS, CoverLaw, SURFACE_MESHLET_LEVEL,
    WATER_HELD, WaterCell, cover_clump_layer, cover_clump_spacing, cover_clumped, cover_growth,
    cover_seen, cover_share, cover_shown, macro_tint, spread_slopes, surface_vertex, terrain,
    terrain_directory, terrain_level_height, terrain_view, water_cell, water_level};
import package::face::horizons::bounce_light;
import package::face::lighting::{finish_ground, local_light, widened};
import package::face::shadows::{KeptLift, keep_lift, kept_texels};
import package::face::sun::sun_direction;
import package::face::surface_point::{BODY_GROUND, SurfacePoint};
import package::frame::view::{DETAIL_EDGE_ON, PI, camera, detail_across, detail_distance, detail_level,
    detail_resolved, detail_span, eye_world, fade_noise, material_detail, stable_normalize};
import package::meshes::declarations::{DepthSceneTargets, DepthTemporalTargets, SceneTargets,
    TemporalTargets};
import package::meshes::surface::{Shaded, scene_targets, temporal_targets, world_plane};
import package::terrain::draw_shared::{terrain_hash2, terrain_value, terrain_value2};
import package::terrain::sky::terrain_sky_map;
import package::terrain::surface::{surface_leaf, surface_relative, terrain_true_height};

@group(1) @binding(20) var<storage, read> terrain_holes: array<u32>;

override TERRAIN_HOLES: bool = false;

@group(1) @binding(21) var terrain_normals: texture_2d_array<u32>;
@group(1) @binding(24) var terrain_control: texture_2d_array<u32>;
@group(1) @binding(32) var terrain_control_sampler: sampler;

struct TerrainPaletteEntry {
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
const TERRAIN_PALETTE_ENTRIES: u32 = 113u;
@group(1) @binding(25) var<uniform> terrain_palette: array<TerrainPaletteEntry, TERRAIN_PALETTE_ENTRIES>;
@group(1) @binding(26) var terrain_albedo: texture_2d_array<f32>;
@group(1) @binding(27) var terrain_material_data: texture_2d_array<f32>;
@group(1) @binding(29) var terrain_material_sampler: sampler;
@group(1) @binding(35) var terrain_macro: texture_2d_array<f32>;

struct TerrainFootprint {
    resolved: f32,
    along: f32,
}

fn terrain_footprint(span: vec4<f32>) -> TerrainFootprint {
    return TerrainFootprint(detail_resolved(span.xy), max(span.x, span.y));
}

fn terrain_octave(size: f32, resolved: f32) -> f32 {
    return saturate(2.0 - 2.0 * resolved / size);
}

fn terrain_macro_colors(site: TerrainSite) -> vec4<f32> {
    let layer = terrain.macro_map.x;
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    let size = vec2<f32>(textureDimensions(terrain_macro));
    let step = max(extent.x / size.x, extent.y / size.y);
    let lod = max(log2(max(site.footprint.resolved, 1.0e-6) / step), 0.0);
    let edge = 0.5 * exp2(ceil(lod)) / size;
    let uv = clamp((site.world.xz - terrain.origin.xz) / extent, edge, vec2<f32>(1.0) - edge);
    return textureSampleGrad(terrain_macro, terrain_material_sampler, uv, layer - 1u,
        site.world_dx.xz / extent, site.world_dy.xz / extent);
}

fn terrain_tint(world: vec3<f32>, ground: TerrainGround, site: TerrainSite) -> vec3<f32> {
    let resolved = site.footprint.resolved;
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, resolved);
    var plain = 1.0;
    if ground.mapped < 1.0 {
        let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, resolved);
        plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
    }
    var mapped = 1.0;
    if ground.mapped > 0.0 {
        let broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, resolved);
        mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
    }
    return mix(plain, mapped, ground.mapped) * macro_tint(site.colors, ground.vegetated);
}

struct TerrainOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) relative: vec4<f32>,
    @location(2) cell: vec2<f32>,
}

struct TerrainTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) relative: vec4<f32>,
    @location(2) cell: vec2<f32>,
    @location(4) previous: vec3<f32>,
}

struct TerrainSurfaceVertex {
    relative: vec3<f32>,
    local: vec2<f32>,
    spacing: f32,
}

fn terrain_surface_vertex(leaf: u32, vertex: u32) -> TerrainSurfaceVertex {
    let triangle = surface_leaf(leaf);
    var point: TerrainSurfaceVertex;
    point.local = surface_vertex(triangle, vertex);
    point.relative = surface_relative(point.local, terrain_true_height(point.local));
    point.relative += terrain_displacement(eye_world(point.relative), point.local, 0.0).moved;
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

fn terrain_detail(p: vec2<f32>, size: f32, resolved: f32) -> f32 {
    let kept = terrain_octave(size, resolved);
    if kept == 0.0 {
        return 0.5;
    }
    return 0.5 + (terrain_value(p / size) - 0.5) * kept;
}

fn terrain_layer_fades(layer: u32) -> vec2<f32> {
    let half = (terrain_holes[layer / 2u] >> ((layer % 2u) * 16u)) & 0xffffu;
    return vec2<f32>(f32(half & 0xffu), f32(half >> 8u)) * (1.0 / 255.0);
}

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
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texels = textureGather(0, terrain_normals, terrain_control_sampler,
            (vec2<f32>(corner) + vec2<f32>(2.0)) / 259.0, layer);
        let fade = terrain_layer_fades(layer).x;
        gradient += terrain_gradient_between(texels, f) * (left * (1.0 - fade));
        left *= fade;
        if left <= 0.0 {
            return gradient;
        }
    }
    return gradient;
}

struct TerrainShadingLevel {
    level: u32,
    blend: f32,
}

fn terrain_shading_level(footprint: f32) -> TerrainShadingLevel {
    let levels = terrain.directory.y;
    let over = terrain.pixel_error.x * footprint / terrain.anchor.w;
    var shading: TerrainShadingLevel;
    shading.level = u32(clamp(ceil(log2(max(over, 1.0e-6))) - 1.0, 0.0, f32(levels - 1u)));
    shading.blend = select(0.0, saturate(over / f32(1u << shading.level) - 1.0),
        shading.level + 1u < levels);
    return shading;
}

fn terrain_pixel_shading(world: vec3<f32>) -> TerrainShadingLevel {
    return terrain_shading_level(detail_across(world));
}

const KEPT_GROUND_TEXELS: f32 = 2.0;

fn terrain_kept_lift(world: vec3<f32>, local: vec2<f32>) -> KeptLift {
    let kept = kept_texels(world);
    var lift: KeptLift;
    lift.first = kept.first;
    let count = kept.count;
    let levels = terrain.directory.y;
    for (var index = 0u; index < 4u && lift.first + index < count; index++) {
        let texel = kept.texels[index];
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

fn terrain_gradient_between(texels: vec4<u32>, f: vec2<f32>) -> vec2<f32> {
    let low = mix(unpack2x16float(texels.w), unpack2x16float(texels.z), f.x);
    let high = mix(unpack2x16float(texels.x), unpack2x16float(texels.y), f.x);
    return mix(low, high, f.y);
}

fn terrain_surface_normal(local: vec2<f32>, shading: TerrainShadingLevel) -> vec3<f32> {
    var gradient = terrain_level_gradient(local, shading.level);
    if shading.blend > 0.0 {
        gradient = mix(gradient, terrain_level_gradient(local, shading.level + 1u),
            shading.blend);
    }
    return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
}

fn terrain_material(material: u32) -> TerrainPaletteEntry {
    return terrain_palette[min(terrain.directory.z + material, TERRAIN_PALETTE_ENTRIES - 1u)];
}

struct TerrainBlend {
    ids: vec4<u32>,
    weights: vec4<f32>,
}

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

fn terrain_controls_blend(blend: ptr<function, TerrainBlend>, controls: vec4<u32>,
        f: vec2<f32>, weight: f32) {
    let weights = vec4<f32>(weight * (1.0 - f.x) * (1.0 - f.y), weight * f.x * (1.0 - f.y),
        weight * (1.0 - f.x) * f.y, weight * f.x * f.y);
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

fn terrain_averaged() -> bool {
    return terrain.steep.w > 0.5;
}

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
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let own = weight * (1.0 - fade);
        let controls = textureGather(0, terrain_control, terrain_control_sampler,
            (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer);
        terrain_controls_blend(blend, controls, f, own);
    }
    return weight * fade;
}

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

const TERRAIN_BORDER_TURN: f32 = 4.0;

fn terrain_border_noise(local: vec2<f32>, level: u32) -> vec2<f32> {
    let turned = vec2<f32>(0.8 * local.x - 0.6 * local.y, 0.6 * local.x + 0.8 * local.y);
    return terrain_value2(turned / (2.5 * terrain.anchor.w * f32(1u << level))
        + vec2<f32>(11.3, -6.7));
}

fn terrain_border_blend(blend: ptr<function, TerrainBlend>, local: vec2<f32>,
        shading: TerrainShadingLevel) {
    let weights = (*blend).weights;
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

fn terrain_blend(local: vec2<f32>, normal: vec3<f32>, shading: TerrainShadingLevel)
        -> TerrainBlend {
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    let averaged = terrain_averaged();
    var near = 1.0 - shading.blend;
    var level = shading.level;
    if averaged && level == 0u {
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

const TERRAIN_WET_SPACING: f32 = 2.0;

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

fn terrain_wetness(local: vec2<f32>, shading: TerrainShadingLevel) -> f32 {
    var wet = terrain_level_wetness(local, shading.level);
    if shading.blend > 0.0 {
        wet = mix(wet, terrain_level_wetness(local, shading.level + 1u), shading.blend);
    }
    return wet;
}

const TERRAIN_FRINGE: f32 = 0.5;

struct TerrainWaterline {
    above: f32,
    rise: vec2<f32>,
}

fn terrain_waterline_at(cell: WaterCell, height: f32, rise: vec2<f32>) -> TerrainWaterline {
    var waterline = TerrainWaterline(TERRAIN_FRINGE, vec2<f32>(0.0));
    let level = water_level(cell);
    if level.held <= 0.0 {
        return waterline;
    }
    waterline.above = height - level.surface;
    waterline.rise = rise + level.slope;
    let ending = (1.0 - level.held / WATER_HELD) * TERRAIN_FRINGE;
    if ending > waterline.above {
        waterline.above = ending;
        waterline.rise = level.held_rate * (-TERRAIN_FRINGE / WATER_HELD);
    }
    return waterline;
}

fn terrain_fringe_wetted(above: f32) -> f32 {
    let within = min(above, TERRAIN_FRINGE);
    let drying = max(within, 0.0);
    return within - drying * drying * (0.5 / TERRAIN_FRINGE);
}

fn terrain_waterline(drawn: vec3<f32>, site: TerrainSite) -> f32 {
    let local = drawn.xz - terrain.origin.xz;
    let height = drawn.y - terrain.origin.y;
    let rise = -site.geometric.xz / max(site.geometric.y, 1.0e-3);
    let picked = detail_level(site.footprint.resolved, terrain.anchor.w, terrain.directory.y);
    let cell = water_cell(local, picked.level);
    var waterline = terrain_waterline_at(cell, height, rise);
    if picked.blend > 0.0 && cell.level == picked.level {
        let coarser = terrain_waterline_at(water_cell(local, picked.level + 1u), height, rise);
        waterline.above = mix(waterline.above, coarser.above, picked.blend);
        waterline.rise = mix(waterline.rise, coarser.rise, picked.blend);
    }
    let half = 0.5 * length(waterline.rise) * sqrt(site.across * site.footprint.along);
    if half < 1.0e-3 * TERRAIN_FRINGE {
        return saturate(1.0 - waterline.above / TERRAIN_FRINGE);
    }
    let low = max(waterline.above, -half);
    return (terrain_fringe_wetted(low + half) - terrain_fringe_wetted(low - half)) / (2.0 * half);
}

const TERRAIN_POROUS: f32 = 0.5;
const TERRAIN_POROUS_RANGE: f32 = 0.4;
const TERRAIN_SOAKED: f32 = 0.2;

fn terrain_wet_ground(roughness: f32, wet: f32) -> vec2<f32> {
    let gloss = 1.0 - roughness;
    let porosity = saturate(((1.0 - gloss) - TERRAIN_POROUS) / TERRAIN_POROUS_RANGE);
    let factor = mix(1.0, TERRAIN_SOAKED, porosity);
    return vec2<f32>(mix(1.0, factor, wet), 1.0 - mix(1.0, gloss, mix(1.0, factor, 0.5 * wet)));
}

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
}

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
        let shares = max(sharp / (sharp.x + sharp.y + sharp.z) - 0.03, vec3<f32>(0.0));
        mapping.planes = shares / (shares.x + shares.y + shares.z);
    }
    return mapping;
}

const TERRAIN_TURN: vec2<f32> = vec2<f32>(0.5, 0.8660254);
const TERRAIN_STRETCH: f32 = 1.31;

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

fn terrain_lattice_at(world: vec3<f32>, mapping: TerrainMapping, axis: u32, lattice: u32)
        -> vec2<f32> {
    return terrain_lattice(world_plane(world, axis) * mapping.scale, axis, lattice)
        + vec2<f32>(0.37, 0.71) * f32(lattice);
}

fn terrain_plane_texel(map: texture_2d_array<f32>, layer: u32, mapping: TerrainMapping,
        world: vec3<f32>, world_dx: vec3<f32>, world_dy: vec3<f32>, axis: u32, lattice: u32)
        -> vec4<f32> {
    return textureSampleGrad(map, terrain_material_sampler,
        terrain_lattice_at(world, mapping, axis, lattice), layer,
        terrain_lattice(world_plane(world_dx, axis) * mapping.scale, axis, lattice),
        terrain_lattice(world_plane(world_dy, axis) * mapping.scale, axis, lattice));
}

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
        if axis == 1u {
            n = vec3<f32>(TERRAIN_TURN.x * n.x - TERRAIN_TURN.y * n.y,
                TERRAIN_TURN.y * n.x + TERRAIN_TURN.x * n.y, n.z);
        } else {
            n.x *= TERRAIN_STRETCH;
        }
    }
    if axis == 0u {
        return vec4<f32>(stable_normalize(vec3<f32>(abs(n.z) * normal.x, n.y + normal.y,
            n.x + normal.z)), lost);
    }
    if axis == 2u {
        return vec4<f32>(stable_normalize(vec3<f32>(n.x + normal.x, n.y + normal.y,
            abs(n.z) * normal.z)), lost);
    }
    return vec4<f32>(stable_normalize(vec3<f32>(n.x + normal.x, abs(n.z) * normal.y,
        normal.z - n.y)), lost);
}

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

fn terrain_second_lattice(world: vec3<f32>, tiling: f32) -> f32 {
    let p = (world.xz + world.y * vec2<f32>(0.71, -0.43)) / (tiling * 3.1);
    return smoothstep(0.4, 0.6, terrain_value(p + vec2<f32>(7.3, -2.9)));
}

fn terrain_mapped(entry: TerrainPaletteEntry, site: TerrainSite) -> bool {
    return entry.params.w > 0.0 && site.footprint.resolved < entry.params.x;
}

const TERRAIN_HEIGHT_BAND: f32 = 0.08;

fn terrain_lattice_share(second: f32, first: f32, raised: f32, contrast: f32) -> f32 {
    let a = 1.0 - second + contrast * first;
    let b = second + contrast * raised;
    let top = max(a, b) - TERRAIN_HEIGHT_BAND;
    let lifted = max(vec2<f32>(a, b) - top, vec2<f32>(0.0));
    return mix(second, lifted.y / max(lifted.x + lifted.y, 1.0e-6), saturate(contrast));
}

fn terrain_lattice_mix(pattern: f32, first: f32, other: f32, contrast: f32) -> f32 {
    if pattern > 0.0 && pattern < 1.0 {
        return terrain_lattice_share(pattern, first, other, contrast);
    }
    return pattern;
}

struct TerrainSite {
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    footprint: TerrainFootprint,
    geometric: vec3<f32>,
    toward: vec3<f32>,
    colors: vec4<f32>,
    thinned: f32,
    grown: f32,
    clump: vec4<f32>,
    across: f32,
}

fn terrain_thinned(world: vec3<f32>) -> f32 {
    let cover = terrain.cover;
    if cover.z <= 0.0 {
        return 0.0;
    }
    if cover.x <= 0.0 {
        return 1.0;
    }
    let law = CoverLaw(cover.x);
    let away = detail_distance(world);
    return 1.0 - cover_shown(away, cover_share(away, law), law);
}

fn terrain_grown(colors: vec4<f32>) -> f32 {
    let growth = cover_growth(colors.a);
    return growth.height * growth.density;
}

fn terrain_canopy_clump(world: vec3<f32>, resolved: f32) -> vec4<f32> {
    var look = vec4<f32>(0.0);
    for (var layer = 0u; layer < 2u; layer++) {
        let kept = terrain_octave(cover_clump_spacing(layer), resolved);
        if kept > 0.0 {
            look += cover_clump_layer(world.xz, layer) * (0.5 * kept);
        }
    }
    return look;
}

fn terrain_across(world: vec3<f32>) -> f32 {
    return detail_across(world);
}

fn terrain_blade_normal(ground: vec3<f32>) -> vec3<f32> {
    let sun = sun_direction();
    let across = vec3<f32>(sun.x, 0.0, sun.z);
    let side = across * (0.6366 / max(length(across), 1.0e-4));
    return stable_normalize(mix(side, ground, 0.6));
}

fn terrain_canopy_depth(leaves: f32, site: TerrainSite) -> f32 {
    let sine = max(abs(dot(site.toward, site.geometric)), 0.03);
    return 0.5 * leaves * terrain.cover.z / sine;
}

fn terrain_canopy(leaves: f32, site: TerrainSite) -> f32 {
    if site.thinned <= 0.0 || leaves <= 0.0 {
        return 0.0;
    }
    return site.thinned * (1.0 - exp(-terrain_canopy_depth(leaves, site)));
}

fn terrain_canopy_through(site: TerrainSite) -> f32 {
    let sun = sun_direction();
    let lengths = length(site.toward.xz) * length(sun.xz);
    var through = 0.5;
    if lengths > 1.0e-4 {
        through = acos(clamp(dot(site.toward.xz, sun.xz) / lengths, -1.0, 1.0)) / PI;
    }
    return mix(1.0, COVER_FAR_SIDE, through);
}

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

fn terrain_placed_share(entry: TerrainPaletteEntry, site: TerrainSite) -> f32 {
    let covered = entry.placed.w;
    if covered <= 0.0 || site.across <= 0.0 {
        return 0.0;
    }
    let first = entry.placed_size.x;
    let octaves = max(first - entry.placed_size.y, 0.0);
    let kept = saturate((1.0 + first - log2(site.across)) / (1.0 + octaves));
    var shown = kept;
    if octaves > 1.0e-3 {
        shown = (1.0 - exp2(-2.0 * octaves * kept)) / (1.0 - exp2(-2.0 * octaves));
    }
    return covered * (1.0 - shown);
}

struct TerrainSlot {
    mapped: bool,
    texel: vec4<f32>,
    second: f32,
    raised: f32,
    contrast: f32,
}

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
        slot.second = terrain_lattice_mix(pattern, first.a, other.a, sharpness);
        slot.texel = mix(first, other, slot.second);
        height = slot.texel.a;
    }
    slot.raised = weight + sharpness * height;
    slot.contrast = weight * sharpness;
    return slot;
}

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
        let clumped = cover_clumped(site.clump);
        let tip = entry.canopy.rgb * clumped.tint;
        let leaves = entry.canopy.w * site.grown * clumped.height;
        let canopy = terrain_canopy(leaves, site);
        if canopy > 0.0 {
            let look = terrain_canopy_look(color, tip, leaves, site);
            color = mix(color, look.color, canopy);
            roughness = mix(roughness, COVER_ROUGHNESS, canopy);
            occlusion = mix(occlusion, look.occlusion, canopy);
            turned = mix(turned, terrain_blade_normal(site.geometric), canopy);
        }
        (*ground).vegetated += weight;
        (*ground).canopy += weight * canopy;
    }
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

struct TerrainShares {
    ids: vec4<u32>,
    weights: vec4<f32>,
    first: TerrainSlot,
    second: TerrainSlot,
    third: TerrainSlot,
    fourth: TerrainSlot,
}

fn terrain_ground(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainGround {
    return terrain_shared_ground(terrain_shares(site, shading), site);
}

fn terrain_shares(site: TerrainSite, shading: TerrainShadingLevel) -> TerrainShares {
    let blend = terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, shading);
    let first = terrain_slot(blend.ids.x, blend.weights.x, site);
    let second = terrain_slot(blend.ids.y, blend.weights.y, site);
    let third = terrain_slot(blend.ids.z, blend.weights.z, site);
    let fourth = terrain_slot(blend.ids.w, blend.weights.w, site);
    let raised = vec4<f32>(first.raised, second.raised, third.raised, fourth.raised);
    let contrast = vec4<f32>(first.contrast, second.contrast, third.contrast, fourth.contrast);
    let highest = max(max(raised.x, raised.y), max(raised.z, raised.w));
    let lifted = max(raised - vec4<f32>(highest - TERRAIN_HEIGHT_BAND), vec4<f32>(0.0));
    let shares = lifted / max(dot(lifted, vec4<f32>(1.0)), 1.0e-6);
    let weights = mix(blend.weights, shares,
        saturate(contrast.x + contrast.y + contrast.z + contrast.w));
    return TerrainShares(blend.ids, weights, first, second, third, fourth);
}

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

const TERRAIN_MESH_CARRIES: f32 = 2.0;

const TERRAIN_BAND_PIXELS: f32 = 1.0;

struct TerrainHeights {
    mapping: TerrainMapping,
    pattern: f32,
    contrast: f32,
}

fn terrain_material_heights(entry: TerrainPaletteEntry, world: vec3<f32>,
        normal: vec3<f32>) -> TerrainHeights {
    return TerrainHeights(terrain_mapping(entry, normal),
        terrain_second_lattice(world, entry.params.x), entry.params.y);
}

fn terrain_lattice_height(mapping: TerrainMapping, world: vec3<f32>, lattice: u32, level: f32)
        -> f32 {
    var height = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = mapping.planes[axis];
        if share > 0.0 {
            height += share * textureSampleLevel(terrain_material_data, terrain_material_sampler,
                terrain_lattice_at(world, mapping, axis, lattice), 2u * mapping.layer + 1u,
                level).a;
        }
    }
    return height;
}

fn terrain_band_height(heights: TerrainHeights, world: vec3<f32>, level: f32) -> f32 {
    var first = 0.0;
    var other = 0.0;
    if heights.pattern < 1.0 {
        first = terrain_lattice_height(heights.mapping, world, 0u, level);
    }
    if heights.pattern > 0.0 {
        other = terrain_lattice_height(heights.mapping, world, 1u, level);
    }
    return mix(first, other, terrain_lattice_mix(heights.pattern, first, other, heights.contrast));
}

fn terrain_pixel_height(heights: TerrainHeights, world: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    let layer = 2u * heights.mapping.layer + 1u;
    var first = 0.0;
    var other = 0.0;
    if heights.pattern < 1.0 {
        first = terrain_map(terrain_material_data, layer, heights.mapping, world, dx, dy, 0u).a;
    }
    if heights.pattern > 0.0 {
        other = terrain_map(terrain_material_data, layer, heights.mapping, world, dx, dy, 1u).a;
    }
    return mix(first, other, terrain_lattice_mix(heights.pattern, first, other, heights.contrast));
}

fn terrain_band_level(mapping: TerrainMapping, meters: f32) -> f32 {
    let texels = f32(textureDimensions(terrain_material_data).x) * mapping.scale;
    return max(log2(meters * texels), 0.0);
}

fn terrain_carried(across: f32) -> f32 {
    return TERRAIN_MESH_CARRIES * max(terrain.relief.z, across);
}

struct TerrainReliefLayer {
    depth: f32,
    heights: TerrainHeights,
    mean: f32,
}

fn terrain_relief_layer(material: u32, weight: f32, site: TerrainSite) -> TerrainReliefLayer {
    var layer: TerrainReliefLayer;
    layer.depth = 0.0;
    if weight <= 0.0 || material == 0xffffffffu {
        return layer;
    }
    let entry = terrain_material(material);
    if entry.relief.x <= 0.0 || !terrain_mapped(entry, site) {
        return layer;
    }
    layer.heights = terrain_material_heights(entry, site.world, site.geometric);
    layer.mean = entry.heights.x;
    layer.depth = entry.relief.x * weight;
    return layer;
}

fn terrain_relief_rise(layer: TerrainReliefLayer, point: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    if layer.depth <= 0.0 {
        return 0.0;
    }
    return layer.depth * (terrain_pixel_height(layer.heights, point, dx, dy) - layer.mean);
}

struct TerrainLift {
    moved: vec3<f32>,
    kept: vec3<f32>,
}

fn terrain_material_rises(entry: TerrainPaletteEntry) -> bool {
    return entry.relief.x > 0.0 && entry.params.w > 0.0;
}

fn terrain_band_rise(blend: TerrainBlend, world: vec3<f32>, normal: vec3<f32>,
        meters: vec2<f32>) -> vec2<f32> {
    var rise = vec2<f32>(0.0);
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = blend.weights[slot];
        if weight <= 0.0 {
            continue;
        }
        let entry = terrain_material(blend.ids[slot]);
        if !terrain_material_rises(entry) {
            continue;
        }
        let heights = terrain_material_heights(entry, world, normal);
        let depth = weight * entry.relief.x;
        let band = terrain_band_height(heights, world,
            terrain_band_level(heights.mapping, meters.x));
        var coarse = band;
        if meters.y > meters.x {
            coarse = terrain_band_height(heights, world,
                terrain_band_level(heights.mapping, meters.y));
        }
        rise += depth * (vec2<f32>(band, coarse) - entry.heights.x);
    }
    return rise;
}

fn terrain_displacement(world: vec3<f32>, local: vec2<f32>, pieces: f32) -> TerrainLift {
    var lift = TerrainLift(vec3<f32>(0.0), vec3<f32>(0.0));
    let across = max(detail_across(world), 1.0e-6);
    if terrain.relief.x < TERRAIN_BAND_PIXELS * across {
        return lift;
    }
    let carried = terrain_carried(across);
    let spaced = max(carried, TERRAIN_MESH_CARRIES * pieces);
    let shading = terrain_pixel_shading(world);
    let normal = terrain_surface_normal(local, shading);
    let blend = terrain_blend(local, normal, shading);
    let rise = terrain_band_rise(blend, world, normal, vec2<f32>(carried, spaced));
    lift.moved = normal * rise.x;
    lift.kept = normal * rise.y;
    return lift;
}

struct TerrainRelief {
    world: vec3<f32>,
    beyond: f32,
}

struct TerrainReliefLayers {
    a: TerrainReliefLayer,
    b: TerrainReliefLayer,
    c: TerrainReliefLayer,
    d: TerrainReliefLayer,
    lifted: f32,
    dx: vec3<f32>,
    dy: vec3<f32>,
}

fn terrain_sink(layers: TerrainReliefLayers, point: vec3<f32>, fade: f32) -> f32 {
    let ground = terrain_relief_rise(layers.a, point, layers.dx, layers.dy)
        + terrain_relief_rise(layers.b, point, layers.dx, layers.dy)
        + terrain_relief_rise(layers.c, point, layers.dx, layers.dy)
        + terrain_relief_rise(layers.d, point, layers.dx, layers.dy);
    return (layers.lifted - ground) * fade;
}

fn terrain_relief_reach(layer: TerrainReliefLayer) -> vec2<f32> {
    return layer.depth * vec2<f32>(1.0 - layer.mean, layer.mean);
}

fn terrain_relief_sideways(layer: TerrainReliefLayer) -> bool {
    return layer.depth > 0.0 && layer.heights.mapping.planes.y < 1.0;
}

const TERRAIN_RELIEF_STRIDE: f32 = 1.0 / 64.0;

fn terrain_relief_repeat(layer: TerrainReliefLayer) -> f32 {
    if layer.depth <= 0.0 {
        return 1.0e6;
    }
    return 1.0 / layer.heights.mapping.scale;
}

const TERRAIN_RELIEF_REFINE: u32 = 2u;

fn terrain_relief_across(way: vec3<f32>, normal: vec3<f32>) -> f32 {
    let ray = way / max(length(way), 1.0e-6);
    let travel = ray / max(-dot(ray, normal), DETAIL_EDGE_ON);
    return min(length(travel.xz), 3.0);
}

fn terrain_parallax(site: TerrainSite, shading: TerrainShadingLevel, dither: f32,
        lifted: f32) -> TerrainRelief {
    var relief: TerrainRelief;
    relief.world = site.world;
    relief.beyond = 0.0;
    let to_point = site.world - camera.position.xyz;
    let ray = to_point / max(length(to_point), 1.0e-6);
    let into = -dot(ray, site.geometric);
    let grazing = smoothstep(0.0, DETAIL_EDGE_ON, into);
    let pixels = terrain_relief_across(site.world - camera.detail.xyz, site.geometric)
        / site.footprint.along;
    let shows = grazing > 0.0 && terrain.relief.x * pixels >= TERRAIN_BAND_PIXELS;
    if !shows {
        return relief;
    }
    let blend = terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, shading);
    let weights = blend.weights;
    var layers: TerrainReliefLayers;
    layers.a = terrain_relief_layer(blend.ids.x, weights.x, site);
    layers.b = terrain_relief_layer(blend.ids.y, weights.y, site);
    layers.c = terrain_relief_layer(blend.ids.z, weights.z, site);
    layers.d = terrain_relief_layer(blend.ids.w, weights.w, site);
    layers.lifted = lifted;
    layers.dx = site.world_dx;
    layers.dy = site.world_dy;
    let deep = layers.a.depth + layers.b.depth + layers.c.depth + layers.d.depth;
    if deep * pixels < TERRAIN_BAND_PIXELS {
        return relief;
    }
    let reach = terrain_relief_reach(layers.a) + terrain_relief_reach(layers.b)
        + terrain_relief_reach(layers.c) + terrain_relief_reach(layers.d);
    let top = min(lifted - reach.x, 0.0) * grazing;
    let depth = max(lifted + reach.y, 0.0) * grazing;
    let travel = ray / max(into, DETAIL_EDGE_ON);
    let across = length(travel.xz);
    let along = travel.xz * min(1.0, 3.0 / max(across, 1.0e-6));
    let rise = -site.geometric.xz / max(site.geometric.y, 0.05);
    let march = vec3<f32>(along.x, dot(rise, along), along.y);
    var start = site.world;
    if terrain_relief_sideways(layers.a) || terrain_relief_sideways(layers.b)
            || terrain_relief_sideways(layers.c) || terrain_relief_sideways(layers.d) {
        let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
        if held.found {
            start.y = held.height + terrain.origin.y;
        }
    }
    let crossed = (depth - top) * length(along);
    let stride = TERRAIN_RELIEF_STRIDE * min(
        min(terrain_relief_repeat(layers.a), terrain_relief_repeat(layers.b)),
        min(terrain_relief_repeat(layers.c), terrain_relief_repeat(layers.d)));
    let wanted = max((depth - top) * pixels, crossed / max(stride, 1.0e-4));
    let count = clamp(wanted, 2.0, terrain.relief.y);
    let steps = max(u32(floor(count + dither)), 2u);
    var previous = top;
    var gap = top - terrain_sink(layers, start + march * top, grazing);
    var hit = top;
    if gap < 0.0 {
        hit = depth;
        for (var step = 1u; step <= steps; step++) {
            let at = top + (depth - top) * f32(step) / f32(steps);
            let point = start + march * at;
            let next_gap = at - terrain_sink(layers, point, grazing);
            if next_gap >= 0.0 {
                var low = vec2<f32>(previous, gap);
                var high = vec2<f32>(at, next_gap);
                for (var refine = 0u; refine < TERRAIN_RELIEF_REFINE; refine++) {
                    let middle = mix(low.x, high.x, low.y / min(low.y - high.y, -1.0e-6));
                    let middle_gap = middle - terrain_sink(layers, start + march * middle, grazing);
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
    relief.beyond = hit * length(travel);
    return relief;
}

fn terrain_point(relative: vec3<f32>) -> SurfacePoint {
    let look = terrain_forward_look(relative);
    let point = terrain_point_seen(relative, look, terrain_seen(relative, look), relative);
    return widened(point, dpdx(point.normal), dpdy(point.normal));
}

struct TerrainLook {
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    ray_dx: vec3<f32>,
    ray_dy: vec3<f32>,
    raw: vec2<bool>,
    dither: f32,
}

fn terrain_forward_look(relative: vec3<f32>) -> TerrainLook {
    let world = eye_world(relative);
    let ray = relative / max(length(relative), 1.0e-6);
    var look: TerrainLook;
    look.world_dx = dpdx(world);
    look.world_dy = dpdy(world);
    look.ray_dx = dpdx(ray);
    look.ray_dy = dpdy(ray);
    look.raw = vec2<bool>(true);
    look.dither = 0.5;
    return look;
}

struct TerrainSeen {
    site: TerrainSite,
    shading: TerrainShadingLevel,
    beyond: f32,
}

fn terrain_seen(relative: vec3<f32>, look: TerrainLook) -> TerrainSeen {
    let drawn = eye_world(relative);
    let lift = terrain_displacement(drawn, drawn.xz - terrain.origin.xz, 0.0).moved;
    let world = drawn - lift;
    let away = length(relative);
    let ray = relative / max(away, 1.0e-6);
    let ray_dx = look.ray_dx;
    let ray_dy = look.ray_dy;
    let local = world.xz - terrain.origin.xz;
    let shading = terrain_pixel_shading(world);
    let geometric = terrain_surface_normal(local, shading);
    let facing = min(dot(geometric, ray), -0.015625);
    let detail = material_detail();
    let smooth_dx = away * detail * (ray_dx - ray * (dot(geometric, ray_dx) / facing));
    let smooth_dy = away * detail * (ray_dy - ray * (dot(geometric, ray_dy) / facing));
    var site: TerrainSite;
    site.world = world;
    site.world_dx = smooth_dx;
    site.world_dy = smooth_dy;
    site.footprint = terrain_footprint(detail_span(world, geometric));
    site.geometric = geometric;
    site.toward = -ray;
    site.colors = terrain_macro_colors(site);
    site.thinned = terrain_thinned(world);
    site.grown = terrain_grown(site.colors);
    site.across = terrain_across(world);
    let relief = terrain_parallax(site, shading, look.dither, dot(lift, geometric));
    site.world = relief.world;
    return TerrainSeen(site, shading, relief.beyond);
}

fn terrain_point_seen(relative: vec3<f32>, look: TerrainLook, seen: TerrainSeen,
        depth_point: vec3<f32>) -> SurfacePoint {
    let site = seen.site;
    let world = eye_world(relative);
    let local = world.xz - terrain.origin.xz;
    let geometric = site.geometric;
    keep_lift(terrain_kept_lift(world, local));
    var surface = site;
    let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
    if held.found {
        surface.world.y = held.height + terrain.origin.y;
    }
    if surface.thinned > 0.0 {
        surface.clump = terrain_canopy_clump(surface.world, site.footprint.resolved);
    }
    var ground = terrain_ground(surface, seen.shading);
    ground.albedo *= terrain_tint(surface.world, ground, site);
    var albedo = ground.albedo;
    let normal = ground.normal;
    let occlusion = ground.occlusion;
    var roughness = ground.roughness;
    let wet = max(terrain_wetness(local, seen.shading),
        terrain_waterline(eye_world(depth_point), site)) * (1.0 - ground.canopy);
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
    let detail = material_detail();
    point.world_dx = select(site.world_dx / detail, look.world_dx, look.raw.x);
    point.world_dy = select(site.world_dy / detail, look.world_dy, look.raw.y);
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = true;
    point.body = BODY_GROUND;
    point.leaf = ground.canopy;
    return point;
}

@group(2) @binding(18) var terrain_ids_texture: texture_2d<u32>;

@fragment
fn fs_terrain_ids(input: TerrainOutput) -> @location(0) vec2<u32> {
    terrain_holes_test(input.cell);
    return vec2<u32>(terrain.directory.x + 1u, bitcast<u32>(input.clip_position.z));
}

fn terrain_relative_at(pixel: vec2<f32>, depth: f32, size: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(pixel.x * 2.0 / size.x - 1.0, 1.0 - pixel.y * 2.0 / size.y);
    let point = terrain_view.inverse_relative_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w;
}

fn terrain_ray_at(pixel: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let relative = terrain_relative_at(pixel, 1.0, size);
    return relative / max(length(relative), 1.0e-30);
}

struct TerrainResolved {
    point: SurfacePoint,
    previous: vec3<f32>,
    depth: f32,
}

fn terrain_resolved(position: vec4<f32>, dither: f32) -> TerrainResolved {
    let pixel = vec2<i32>(position.xy);
    let size = camera.region.xy;
    let texel = textureLoad(terrain_ids_texture, pixel, 0);
    let mine = texel.x == terrain.directory.x + 1u;
    if !mine {
        discard;
    }
    let depth = bitcast<f32>(texel.y);
    let relative = terrain_relative_at(position.xy, depth, size);
    let away = length(relative);
    let here = select(0.0, 1.0, mine);
    let reach = 0.05 * away + 0.5;
    let others = vec2<f32>(dpdxFine(here), dpdyFine(here));
    let farther = vec2<f32>(dpdxFine(away), dpdyFine(away));
    let same = (others == vec2<f32>(0.0)) & (abs(farther) < vec2<f32>(reach));
    let ray = terrain_ray_at(position.xy, size);
    var look: TerrainLook;
    look.world_dx = dpdxFine(relative);
    look.world_dy = dpdyFine(relative);
    look.ray_dx = dpdxCoarse(ray);
    look.ray_dy = dpdyCoarse(ray);
    look.raw = same;
    look.dither = dither;
    let seen = terrain_seen(relative, look);
    let met = relative + relative * (seen.beyond / max(away, 1.0e-6));
    let shown = terrain_point_seen(relative, look, seen, met);
    var resolved: TerrainResolved;
    resolved.point = widened(shown, select(vec3<f32>(0.0), dpdxFine(shown.normal), same.x),
        select(vec3<f32>(0.0), dpdyFine(shown.normal), same.y));
    resolved.previous = (camera.previous_view_projection * vec4<f32>(eye_world(met), 1.0)).xyw;
    resolved.depth = depth;
    if seen.beyond != 0.0 {
        let clip = terrain_view.relative_view_projection * vec4<f32>(met, 1.0);
        resolved.depth = select(1.0, saturate(clip.z / clip.w), clip.w > 0.0);
    }
    return resolved;
}

fn terrain_scene_targets(shaded: Shaded, resolved: TerrainResolved, position: vec4<f32>)
        -> DepthSceneTargets {
    let targets = scene_targets(shaded, stored(shaded.color, position.xy));
    return DepthSceneTargets(targets.color, targets.glow, targets.surface, resolved.depth);
}

fn terrain_temporal_targets(shaded: Shaded, resolved: TerrainResolved, position: vec4<f32>)
        -> DepthTemporalTargets {
    let targets = temporal_targets(shaded, stored(shaded.color, position.xy), position,
        resolved.previous);
    return DepthTemporalTargets(targets.color, targets.motion, targets.glow, targets.surface,
        resolved.depth);
}

@fragment
fn fs_terrain_resolve_post(@builtin(position) position: vec4<f32>) -> DepthSceneTargets {
    let resolved = terrain_resolved(position, 0.5);
    let point = resolved.point;
    return terrain_scene_targets(finish_ground(point, local_light(point, position.xy)), resolved,
        position);
}

@fragment
fn fs_terrain_resolve_temporal(@builtin(position) position: vec4<f32>)
        -> DepthTemporalTargets {
    let resolved = terrain_resolved(position, fade_noise(position.xy));
    let point = resolved.point;
    return terrain_temporal_targets(finish_ground(point, local_light(point, position.xy)),
        resolved, position);
}

fn shade_terrain(input: TerrainOutput) -> Shaded {
    let point = terrain_point(input.relative.xyz);
    terrain_holes_test(input.cell);
    return finish_ground(point, local_light(point, input.clip_position.xy));
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
fn fs_terrain_post(input: TerrainOutput) -> SceneTargets {
    let shaded = shade_terrain(input);
    return scene_targets(shaded, stored(shaded.color, input.clip_position.xy));
}

@fragment
fn fs_terrain_temporal(input: TerrainTemporalOutput) -> TemporalTargets {
    let shaded = shade_terrain(terrain_temporal_output(input));
    return temporal_targets(shaded, stored(shaded.color, input.clip_position.xy),
        input.clip_position, input.previous);
}

@fragment
fn fs_terrain_probe(input: TerrainOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade_terrain(input).linear, 1.0);
}
