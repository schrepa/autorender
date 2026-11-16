import package::face::cover::{CoverSite, cover_far, cover_tint};
import package::face::far_cover::{FarGround, GroundAt, far_cover};
import package::face::ground_tiles::macro_color;
import package::face::bounce::bounce_light;
import package::face::foliage::foliage;
import package::face::shadows::kept_texels;
import package::face::sun::sun_direction;
import package::engine::math::PI;
import package::engine::view::{camera, detail_across, detail_level, eye_world,
    material_detail, stable_normalize, view_pixel};
import package::engine::shading::{SurfacePoint, ViewPixel};
import package::engine::slots::{derived, set_added_indirect, set_ambient_bend,
    set_direct_blocked, set_direct_share, set_ignore_face, set_metalness, set_mirrored,
    set_occlusion,
    set_receiver_lift_first, set_receiver_lifts, set_roughness, set_tangled, set_transmission,
    standard_point};
import package::engine::surfaces::SURFACE_GROUND;
import package::face::water::{WATER_HELD, WaterCell, water_cell, water_level};
import package::terrain_shading::draw_shared::{terrain_hash2, terrain_site_gain,
    terrain_site_lush, terrain_value};
import package::terrain_shading::faces::ground_tiles::{DETAIL_EDGE_ON, detail_resolved,
    detail_span, macro_maps};
import package::terrain_shading::faces::heightfield::{surface_drawn, surface_vertex, terrain, terrain_index, terrain_level_height, terrain_look, terrain_view, use_terrain};
import package::terrain_shading::field::spread_slopes;
import package::terrain_shading::holes::terrain_holed;
import package::engine::kinds::KindPoint;
import package::generated::kinds::KIND_TERRAIN;
import package::terrain_shading::sky::terrain_sky_map;
import package::terrain_shading::surface::{surface_leaf, surface_seen};
import package::terrain_shading::tables::{table_fades, table_floats, terrain_tables};
import package::generated::terrain_shading::{CONTROL_PLANE, NORMAL_PLANE, PaletteEntry,
    ROUGH_SUN, ROUGH_TURN, ROUGH_UNIT, ROUGH_VIEW, ROUGH_WORDS,
    RUNG_WORDS, SITE_PLANE, SURFACE_MESHLET_LEVEL, SURFACE_VERTEX_STRIDE, TABLE_RUNGS,
    TILE_PLANES,
    control_a, control_b, control_blend, control_wet};
import package::place::ground_tiles::PALETTE_ENTRIES;

@group(0) @binding(21) var terrain_tile_maps: texture_2d_array<u32>;
@group(0) @binding(32) var terrain_control_sampler: sampler;

@group(0) @binding(25) var<uniform> terrain_palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(26) var terrain_albedo: texture_2d_array<f32>;
@group(0) @binding(27) var terrain_material_data: texture_2d_array<f32>;
@group(0) @binding(29) var terrain_material_sampler: sampler;

const TERRAIN_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

struct TerrainFootprint {
    resolved: f32,
    along: f32,
}

fn terrain_footprint(span: vec4<f32>) -> TerrainFootprint {
    return TerrainFootprint(detail_resolved(span.xy), max(span.x, span.y));
}

fn terrain_macro_colors(site: TerrainSite) -> vec4<f32> {
    let layer = terrain.macro_map.x;
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_tables[header + 1u]),
        f32(terrain_tables[header + 2u])) * 256.0 * terrain.anchor.w;
    let size = vec2<f32>(textureDimensions(macro_maps));
    let step = max(extent.x / size.x, extent.y / size.y);
    let lod = max(log2(max(site.footprint.resolved, 1.0e-6) / step), 0.0);
    let edge = 0.5 * exp2(ceil(lod)) / size;
    let uv = clamp((site.world.xz - terrain.origin.xz) / extent, edge, vec2<f32>(1.0) - edge);
    return textureSampleGrad(macro_maps, terrain_material_sampler, uv, layer - 1u,
        site.world_dx.xz / extent, site.world_dy.xz / extent);
}

fn terrain_tint(ground: TerrainGround, site: TerrainSite) -> vec3<f32> {
    return macro_color(site.colors) * cover_tint(site.colors.a, ground.vegetated);
}

struct TerrainOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(1) relative: vec4<f32>,
    @location(2) cell: vec2<f32>,
    @location(3) @interpolate(flat) terrain: u32,
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
    point.relative = surface_drawn(triangle, point.local);
    point.spacing = length(triangle.c - triangle.a) / f32(1u << SURFACE_MESHLET_LEVEL);
    return point;
}

fn terrain_output(leaf: u32, vertex: u32, index: u32) -> TerrainOutput {
    var output: TerrainOutput;
    if !surface_seen(surface_leaf(leaf)) {
        output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        output.terrain = index;
        return output;
    }
    let point = terrain_surface_vertex(leaf, vertex);
    let relative = point.relative + terrain_view.eye_offset.xyz;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = vec4<f32>(relative, point.spacing);
    output.cell = point.local / terrain.anchor.w;
    output.terrain = index;
    return output;
}

fn terrain_holes_test(cell: vec2<f32>, holes: bool) {
    if holes && terrain_holed(terrain, cell) {
        discard;
    }
}

@vertex
fn vs_terrain(@builtin(instance_index) leaf: u32, @builtin(vertex_index) vertex: u32)
        -> TerrainOutput {
    let index = vertex / SURFACE_VERTEX_STRIDE;
    use_terrain(index);
    return terrain_output(leaf, vertex % SURFACE_VERTEX_STRIDE, index);
}

struct TerrainSurfaceOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

@vertex
fn vs_terrain_surface(@builtin(instance_index) leaf: u32, @builtin(vertex_index) vertex: u32)
        -> TerrainSurfaceOutput {
    let index = vertex / SURFACE_VERTEX_STRIDE;
    use_terrain(index);
    let drawn = terrain_output(leaf, vertex % SURFACE_VERTEX_STRIDE, index);
    var output: TerrainSurfaceOutput;
    output.clip_position = drawn.clip_position;
    output.normal_u = vec4<f32>(0.0, 1.0, 0.0, drawn.cell.x);
    output.world_v = vec4<f32>(eye_world(drawn.relative.xyz), drawn.cell.y);
    output.material = vec4<u32>(0xffffffffu, 0xffffffffu, 0u, 0u);
    output.previous = drawn.clip_position.xyw;
    output.object = vec4<u32>(0u, index, 0u, KIND_TERRAIN);
    return output;
}

fn terrain_layer_fades(layer: u32) -> vec2<f32> {
    return table_fades(layer, terrain_look.shore.z);
}

fn terrain_level_gradient(local: vec2<f32>, level: u32) -> vec2<f32> {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var gradient = vec2<f32>(0.0);
    var left = 1.0;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texels = textureGather(0, terrain_tile_maps, terrain_control_sampler,
            (vec2<f32>(corner) + vec2<f32>(2.0)) / 259.0, layer * TILE_PLANES + NORMAL_PLANE);
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
    let over = terrain.pixel_error.x * camera.pixel_error * footprint / terrain.anchor.w;
    var shading: TerrainShadingLevel;
    shading.level = u32(clamp(ceil(log2(max(over, 1.0e-6))) - 1.0, 0.0, f32(levels - 1u)));
    shading.blend = select(0.0, saturate(over / f32(1u << shading.level) - 1.0),
        shading.level + 1u < levels);
    return shading;
}

fn terrain_pixel_shading(world: vec3<f32>) -> TerrainShadingLevel {
    return terrain_shading_level(detail_across(world));
}

fn terrain_gradient_between(texels: vec4<u32>, f: vec2<f32>) -> vec2<f32> {
    let low = mix(unpack2x16float(texels.w), unpack2x16float(texels.z), f.x);
    let high = mix(unpack2x16float(texels.x), unpack2x16float(texels.y), f.x);
    return mix(low, high, f.y);
}

fn terrain_surface_normal(local: vec2<f32>, shading: TerrainShadingLevel) -> vec3<f32> {
    var gradient = vec2<f32>(0.0);
    let last = select(1u, 2u, shading.blend > 0.0);
    for (var walk = 0u; walk < last; walk++) {
        let found = terrain_level_gradient(local, shading.level + walk);
        gradient = select(found, mix(gradient, found, shading.blend), walk == 1u);
    }
    return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
}

fn terrain_material(material: u32) -> PaletteEntry {
    return terrain_palette[min(terrain.directory.z + material, PALETTE_ENTRIES - 1u)];
}

struct TerrainBlend {
    ids: vec4<u32>,
    weights: vec4<f32>,
    wet: f32,
    site: vec4<f32>,
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
    let share = f32(control_blend(control)) * (1.0 / 255.0);
    let first = weight * (1.0 - share);
    if first != 0.0 {
        terrain_blend_add(blend, control_a(control), first);
    }
    let second = weight * share;
    if second != 0.0 {
        terrain_blend_add(blend, control_b(control), second);
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

fn terrain_controls_wet(controls: vec4<u32>, f: vec2<f32>, level: u32) -> f32 {
    let wets = vec4<f32>(vec4<u32>(control_wet(controls.x), control_wet(controls.y),
        control_wet(controls.z), control_wet(controls.w))) * (1.0 / 255.0);
    let low = mix(wets.w, wets.z, f.x);
    let high = mix(wets.x, wets.y, f.x);
    let spacing = terrain.anchor.w * f32(1u << level);
    let kept = select(saturate(terrain_look.shore.y / spacing), 1.0, terrain_averaged());
    return mix(low, high, f.y) * kept;
}

fn terrain_sites_between(sites: vec4<u32>, f: vec2<f32>) -> vec4<f32> {
    let low = mix(unpack4x8unorm(sites.w), unpack4x8unorm(sites.z), f.x);
    let high = mix(unpack4x8unorm(sites.x), unpack4x8unorm(sites.y), f.x);
    return mix(low, high, f.y);
}

fn terrain_level_own(blend: ptr<function, TerrainBlend>, local: vec2<f32>, level: u32,
        weight: f32) -> f32 {
    let header = terrain.directory.x + level * 4u;
    let samples = max(local / (terrain.anchor.w * f32(1u << level)), vec2<f32>(0.0));
    let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
    let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
    let layer = terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
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
        let controls = textureGather(0, terrain_tile_maps, terrain_control_sampler,
            (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer * TILE_PLANES + CONTROL_PLANE);
        terrain_controls_blend(blend, controls, f, own);
        (*blend).wet += own * terrain_controls_wet(controls, f, level);
        if terrain.macro_map.y != 0u {
            let sites = textureGather(0, terrain_tile_maps, terrain_control_sampler,
                (vec2<f32>(texel) + vec2<f32>(1.0)) / 259.0, layer * TILE_PLANES + SITE_PLANE);
            (*blend).site += own * terrain_sites_between(sites, f);
        }
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
    let held = dot((*blend).weights, vec4<f32>(1.0));
    if held <= 0.0 {
        terrain_blend_add(blend, 0u, left);
        return;
    }
    let grown = (held + left) / held;
    (*blend).weights *= grown;
    (*blend).wet *= grown;
    (*blend).site *= grown;
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
    let last = select(1u, 2u, shading.blend > 0.0);
    for (var walk = select(0u, 1u, !(near > 0.0)); walk < last; walk++) {
        terrain_level_blend(&blend, local, select(level, shading.level + 1u, walk == 1u),
            select(near, shading.blend, walk == 1u));
    }
    if !averaged {
        terrain_steep_blend(&blend, normal);
    }
    return blend;
}

struct TerrainWaterline {
    above: f32,
    rise: vec2<f32>,
}

fn terrain_waterline_at(cell: WaterCell, height: f32, rise: vec2<f32>) -> TerrainWaterline {
    var waterline = TerrainWaterline(terrain_look.shore.x, vec2<f32>(0.0));
    let level = water_level(cell);
    if level.held <= 0.0 {
        return waterline;
    }
    waterline.above = height - level.surface;
    waterline.rise = rise + level.slope;
    let ending = (1.0 - level.held / WATER_HELD) * terrain_look.shore.x;
    if ending > waterline.above {
        waterline.above = ending;
        waterline.rise = level.held_rate * (-terrain_look.shore.x / WATER_HELD);
    }
    return waterline;
}

fn terrain_fringe_wetted(above: f32) -> f32 {
    let within = min(above, terrain_look.shore.x);
    let drying = max(within, 0.0);
    return within - drying * drying * (0.5 / terrain_look.shore.x);
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
    if half < 1.0e-3 * terrain_look.shore.x {
        return saturate(1.0 - waterline.above / terrain_look.shore.x);
    }
    let low = max(waterline.above, -half);
    return (terrain_fringe_wetted(low + half) - terrain_fringe_wetted(low - half)) / (2.0 * half);
}

fn terrain_wet_ground(roughness: f32, wet: f32) -> vec2<f32> {
    let gloss = 1.0 - roughness;
    let porosity = saturate((roughness - terrain_look.wet.x) * terrain_look.wet.y);
    let factor = mix(1.0, terrain_look.wet.z, porosity);
    return vec2<f32>(mix(1.0, factor, wet),
        1.0 - mix(1.0, gloss, mix(1.0, factor, terrain_look.wet.w * wet)));
}

struct TerrainGround {
    albedo: vec3<f32>,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    metalness: f32,
    vegetated: f32,
    canopy: f32,
    through: f32,
    shaded: f32,
    detailed: f32,
    averaged: f32,
    share: vec2<f32>,
}

struct TerrainMapping {
    scale: f32,
    layer: u32,
    planes: vec3<f32>,
}

fn terrain_mapping(entry: PaletteEntry, normal: vec3<f32>) -> TerrainMapping {
    var mapping: TerrainMapping;
    mapping.scale = 1.0 / entry.params.x;
    mapping.layer = u32(entry.params.w) - 1u;
    mapping.planes = vec3<f32>(0.0, 1.0, 0.0);
    if entry.params.z > 0.5 {
        let seen = abs(normal) * vec3<f32>(1.0, max(2.0 - entry.params.z, 1.0e-3), 1.0);
        let sharp = pow(seen, vec3<f32>(terrain_look.planes.x));
        let shares = max(sharp / (sharp.x + sharp.y + sharp.z) - terrain_look.planes.y,
            vec3<f32>(0.0));
        mapping.planes = shares / (shares.x + shares.y + shares.z);
    }
    return mapping;
}

struct TerrainHex {
    turn: vec2<f32>,
    shift: vec2<f32>,
}

struct TerrainHexes {
    open: vec3<f32>,
    turns: mat3x2<f32>,
    shifts: mat3x2<f32>,
}

const TERRAIN_HEX_SKEW: vec2<f32> = vec2<f32>(0.57735027, 1.15470054);

fn terrain_hex(cell: vec2<i32>, salt: u32, turned: f32) -> TerrainHex {
    let seed = (bitcast<u32>(cell.x) * 0x8da6b343u) ^ (bitcast<u32>(cell.y) * 0xd8163841u)
        ^ (salt * 0x9e3779b9u);
    let toward = vec2<f32>(1.0 - turned, 0.0)
        + turned * (terrain_hash2(seed ^ 0x68bc21ebu) * 2.0 - 1.0);
    return TerrainHex(toward * inverseSqrt(max(dot(toward, toward), 1.0e-12)),
        terrain_hash2(seed));
}

fn terrain_hexes(uv: vec2<f32>, axis: u32, layer: u32) -> TerrainHexes {
    let at = uv * terrain_look.tiling.x;
    let skewed = vec2<f32>(at.x - TERRAIN_HEX_SKEW.x * at.y, TERRAIN_HEX_SKEW.y * at.y);
    let base = floor(skewed);
    let f = skewed - base;
    let rest = 1.0 - f.x - f.y;
    let upper = select(0.0, 1.0, rest <= 0.0);
    let flip = 2.0 * upper - 1.0;
    let weights = vec3<f32>(-rest * flip, upper - f.y * flip, upper - f.x * flip);
    let cell = vec2<i32>(base);
    let near = i32(upper);
    let salt = 3u * layer + axis;
    let turned = select(0.0, terrain_look.tiling.z, axis == 1u);
    let first = terrain_hex(cell + vec2<i32>(near, near), salt, turned);
    let second = terrain_hex(cell + vec2<i32>(near, 1 - near), salt, turned);
    let third = terrain_hex(cell + vec2<i32>(1 - near, near), salt, turned);
    let lead = max(weights.x, max(weights.y, weights.z));
    return TerrainHexes(saturate(vec3<f32>(1.0) - (vec3<f32>(lead) - weights)
            * terrain_look.tiling.y),
        mat3x2<f32>(first.turn, second.turn, third.turn),
        mat3x2<f32>(first.shift, second.shift, third.shift));
}

fn terrain_turned(v: vec2<f32>, turn: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(turn.x * v.x - turn.y * v.y, turn.y * v.x + turn.x * v.y);
}

fn terrain_hex_shares(open: vec3<f32>, heights: vec3<f32>) -> vec3<f32> {
    let raised = heights + open - vec3<f32>(1.0);
    let top = max(raised.x, max(raised.y, raised.z)) - terrain_look.tiling.w;
    let lifted = max(raised - vec3<f32>(top), vec3<f32>(0.0));
    return lifted / (lifted.x + lifted.y + lifted.z);
}

fn world_plane(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}

fn terrain_pick2(columns: mat3x2<f32>, at: u32) -> vec2<f32> {
    return select(select(columns[0], columns[1], at == 1u), columns[2], at == 2u);
}

fn terrain_pick3(columns: mat3x3<f32>, at: u32) -> vec3<f32> {
    return select(select(columns[0], columns[1], at == 1u), columns[2], at == 2u);
}

fn terrain_put3(columns: mat3x3<f32>, at: u32, column: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(select(columns[0], column, at == 0u),
        select(columns[1], column, at == 1u), select(columns[2], column, at == 2u));
}

const TERRAIN_LANES: vec4<u32> = vec4<u32>(0u, 1u, 2u, 3u);

fn terrain_lane_set(lanes: vec4<f32>, at: u32, value: f32) -> vec4<f32> {
    return select(lanes, vec4<f32>(value), vec4<u32>(at) == TERRAIN_LANES);
}

fn terrain_lane_set_u(lanes: vec4<u32>, at: u32, value: u32) -> vec4<u32> {
    return select(lanes, vec4<u32>(value), vec4<u32>(at) == TERRAIN_LANES);
}

struct TerrainPlaneAt {
    uv: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
}

fn terrain_plane_at(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, axis: u32, least: f32) -> TerrainPlaneAt {
    let dx = world_plane(world_dx, axis) * mapping.scale;
    let dy = world_plane(world_dy, axis) * mapping.scale;
    return TerrainPlaneAt(world_plane(world, axis) * mapping.scale,
        dx * max(1.0, least * inverseSqrt(max(dot(dx, dx), 1.0e-24))),
        dy * max(1.0, least * inverseSqrt(max(dot(dy, dy), 1.0e-24))));
}

struct TerrainHexed {
    texel: vec4<f32>,
    shares: mat3x3<f32>,
    opened: mat3x3<f32>,
}

fn terrain_hexed(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, level: f32) -> TerrainHexed {
    var hexed: TerrainHexed;
    let layer = 2u * mapping.layer + 1u;
    let span = exp2(level) / f32(textureDimensions(terrain_material_data).x);
    for (var axis = 0u; axis < 3u; axis++) {
        let plane = mapping.planes[axis];
        if plane > 0.0 {
            let at = terrain_plane_at(mapping, world, world_dx, world_dy, axis, 0.0);
            let dx = select(at.dx, vec2<f32>(span, 0.0), level >= 0.0);
            let dy = select(at.dy, vec2<f32>(0.0, span), level >= 0.0);
            let hexes = terrain_hexes(at.uv, axis, mapping.layer);
            var first = vec4<f32>(0.0);
            var second = vec4<f32>(0.0);
            var third = vec4<f32>(0.0);
            if hexes.open.x > 0.0 {
                first = terrain_hex_tap(at.uv, dx, dy, hexes.turns[0], hexes.shifts[0], layer);
            }
            if hexes.open.y > 0.0 {
                second = terrain_hex_tap(at.uv, dx, dy, hexes.turns[1], hexes.shifts[1], layer);
            }
            if hexes.open.z > 0.0 {
                third = terrain_hex_tap(at.uv, dx, dy, hexes.turns[2], hexes.shifts[2], layer);
            }
            let shares = terrain_hex_shares(hexes.open, vec3<f32>(first.a, second.a, third.a));
            hexed.shares = terrain_put3(hexed.shares, axis, shares);
            hexed.opened = terrain_put3(hexed.opened, axis,
                hexes.open / (hexes.open.x + hexes.open.y + hexes.open.z));
            hexed.texel += plane * (shares.x * first + shares.y * second + shares.z * third);
        }
    }
    return hexed;
}

fn terrain_hex_tap(uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, turn: vec2<f32>,
        shift: vec2<f32>, layer: u32) -> vec4<f32> {
    return textureSampleGrad(terrain_material_data, terrain_material_sampler,
        terrain_turned(uv, turn) + shift, layer, terrain_turned(dx, turn), terrain_turned(dy, turn));
}

fn terrain_plane_normal(filtered: vec3<f32>, turn: vec2<f32>, axis: u32, normal: vec3<f32>,
        scale: f32) -> vec4<f32> {
    let shortened = length(filtered);
    let lost = spread_slopes(shortened) * scale * scale;
    let sampled = filtered / max(shortened, 1.0e-4);
    let n = vec3<f32>(terrain_turned(sampled.xy, turn) * scale, sampled.z);
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

struct TerrainNear {
    albedo: vec3<f32>,
    mapped: vec4<f32>,
}

fn terrain_near(mapping: TerrainMapping, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, held: mat3x3<f32>, opened: bool, normal: vec3<f32>, scale: f32,
        least: f32) -> TerrainNear {
    var near: TerrainNear;
    for (var axis = 0u; axis < 3u; axis++) {
        let plane = mapping.planes[axis];
        if plane > 0.0 {
            let at = terrain_plane_at(mapping, world, world_dx, world_dy, axis, least);
            let hexes = terrain_hexes(at.uv, axis, mapping.layer);
            var share = terrain_pick3(held, axis);
            if opened {
                share = hexes.open / (hexes.open.x + hexes.open.y + hexes.open.z);
            }
            for (var hex = 0u; hex < 3u; hex++) {
                if share[hex] > 0.0 {
                    let turn = terrain_pick2(hexes.turns, hex);
                    let uv = terrain_turned(at.uv, turn) + terrain_pick2(hexes.shifts, hex);
                    let dx = terrain_turned(at.dx, turn);
                    let dy = terrain_turned(at.dy, turn);
                    near.albedo += plane * share[hex] * textureSampleGrad(terrain_albedo,
                        terrain_material_sampler, uv, mapping.layer, dx, dy).rgb;
                    if scale > 0.0 {
                        let filtered = textureSampleGrad(terrain_material_data,
                            terrain_material_sampler, uv, 2u * mapping.layer, dx, dy).xyz
                            * 2.0 - 1.0;
                        near.mapped += plane * share[hex]
                            * terrain_plane_normal(filtered, turn, axis, normal, scale);
                    }
                }
            }
        }
    }
    return near;
}

fn terrain_gone(resolved: f32, size: f32, pixels: f32) -> f32 {
    return saturate(log2(max(resolved, 1.0e-6) * pixels / size) + 1.0);
}

struct TerrainRungs {
    tilings: vec2<f32>,
    layers: vec2<u32>,
}

fn terrain_rungs_at(material: u32) -> u32 {
    return TABLE_RUNGS + min(terrain.directory.z + material, PALETTE_ENTRIES - 1u) * RUNG_WORDS;
}

fn terrain_rungs(material: u32) -> TerrainRungs {
    let head = table_floats(terrain_rungs_at(material));
    return TerrainRungs(head.xy, vec2<u32>(head.zw));
}

struct TerrainEnds {
    maps: f32,
    albedo: f32,
    shown: vec2<f32>,
    chain: vec2<f32>,
    rungs: TerrainRungs,
}

fn terrain_ends(entry: PaletteEntry, material: u32, resolved: vec2<f32>) -> TerrainEnds {
    var ends: TerrainEnds;
    ends.maps = 1.0;
    ends.albedo = 1.0;
    if entry.params.w <= 0.0 {
        return ends;
    }
    ends.albedo = terrain_gone(resolved.x, entry.params.x, terrain_look.reach.z);
    ends.maps = max(terrain_gone(resolved.x, entry.params.x, terrain_look.reach.y), ends.albedo);
    ends.rungs = terrain_rungs(material);
    let tilings = ends.rungs.tilings;
    ends.shown = select(vec2<f32>(0.0),
        vec2<f32>(1.0 - terrain_gone(resolved.y, tilings.x, terrain_look.reach.z),
            1.0 - terrain_gone(resolved.y, tilings.y, terrain_look.reach.z)),
        tilings > vec2<f32>(0.0));
    let coarser = select(vec2<f32>(1.0),
        vec2<f32>(terrain_gone(resolved.y, tilings.x, terrain_look.reach.y),
            terrain_gone(resolved.y, tilings.y, terrain_look.reach.y)),
        tilings > vec2<f32>(0.0));
    ends.chain = ends.shown * vec2<f32>(1.0 - coarser.x, coarser.x * (1.0 - coarser.y));
    return ends;
}

fn terrain_fewer_planes(planes: vec3<f32>) -> vec3<f32> {
    let kept = max(planes - terrain_look.far.y, vec3<f32>(0.0));
    return kept / max(kept.x + kept.y + kept.z, 1.0e-6);
}

struct TerrainSite {
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    footprint: TerrainFootprint,
    geometric: vec3<f32>,
    toward: vec3<f32>,
    colors: vec4<f32>,
    across: f32,
    bedding: vec4<f32>,
}

fn terrain_rough_at(normal: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let sun = sun_direction();
    let lit = dot(normal, sun);
    let seen = dot(normal, toward);
    let level = sqrt(max((1.0 - lit * lit) * (1.0 - seen * seen), 1.0e-12));
    let turn = acos(clamp((dot(sun, toward) - lit * seen) / level, -1.0, 1.0));
    let heights = sqrt(asin(saturate(vec2<f32>(lit, seen))) * (2.0 / PI))
        * vec2<f32>(f32(ROUGH_SUN), f32(ROUGH_VIEW)) - 1.0;
    return vec3<f32>(clamp(heights, vec2<f32>(0.0),
        vec2<f32>(f32(ROUGH_SUN - 1u), f32(ROUGH_VIEW - 1u))),
        turn * (f32(ROUGH_TURN - 1u) / PI));
}

fn terrain_rough_word(at: u32) -> vec2<f32> {
    let word = terrain_tables[at];
    return vec2<f32>(vec2<u32>(word & 0xffffu, word >> 16u)) * (1.0 / f32(ROUGH_UNIT));
}

fn terrain_rough(material: u32, at: vec3<f32>) -> vec2<f32> {
    let row = TABLE_RUNGS + PALETTE_ENTRIES * RUNG_WORDS
        + min(terrain.directory.z + material, PALETTE_ENTRIES - 1u) * ROUGH_WORDS;
    let low = min(vec3<u32>(at), vec3<u32>(ROUGH_SUN - 2u, ROUGH_VIEW - 2u, ROUGH_TURN - 2u));
    let f = at - vec3<f32>(low);
    let near = row + (low.z * ROUGH_VIEW + low.y) * ROUGH_SUN + low.x;
    let far = near + ROUGH_VIEW * ROUGH_SUN;
    let first = mix(mix(terrain_rough_word(near), terrain_rough_word(near + 1u), f.x),
        mix(terrain_rough_word(near + ROUGH_SUN), terrain_rough_word(near + ROUGH_SUN + 1u),
            f.x), f.y);
    let second = mix(mix(terrain_rough_word(far), terrain_rough_word(far + 1u), f.x),
        mix(terrain_rough_word(far + ROUGH_SUN), terrain_rough_word(far + ROUGH_SUN + 1u),
            f.x), f.y);
    return mix(first, second, f.z);
}

fn terrain_across(world: vec3<f32>) -> f32 {
    return detail_across(world);
}

struct TerrainSlot {
    far: f32,
    faded: f32,
    shown: vec2<f32>,
    texel: vec4<f32>,
    hexes: mat3x3<f32>,
    chain: vec2<f32>,
    depth: f32,
    buried: f32,
    edge: f32,
    lush: f32,
    textured: bool,
    opened: bool,
    mapping: TerrainMapping,
    rungs: TerrainRungs,
}

fn terrain_rung_mapping(slot: TerrainSlot, rung: u32) -> TerrainMapping {
    var mapping = slot.mapping;
    mapping.scale = 1.0 / slot.rungs.tilings[rung];
    mapping.layer = slot.rungs.layers[rung] - 1u;
    mapping.planes = terrain_fewer_planes(mapping.planes);
    return mapping;
}

fn terrain_rung_blur(rungs: TerrainRungs, fine: f32, rung: u32) -> f32 {
    return 0.5 * terrain_look.reach.w
        * select(rungs.tilings.x, fine, rung == 0u || rungs.tilings.x <= 0.0);
}

fn terrain_maps_gone(material: u32, weight: f32, resolved: f32) -> f32 {
    if weight <= 0.0 {
        return 1.0;
    }
    let entry = terrain_material(material);
    if entry.params.w <= 0.0 {
        return 1.0;
    }
    return max(terrain_gone(resolved, entry.params.x, terrain_look.reach.y),
        terrain_gone(resolved, entry.params.x, terrain_look.reach.z));
}

fn terrain_slot(material: u32, sharing: bool, site: TerrainSite) -> TerrainSlot {
    var slot: TerrainSlot;
    slot.far = 1.0;
    slot.faded = 1.0;
    slot.texel = vec4<f32>(1.0, 1.0, 1.0, 0.5);
    if !sharing {
        return slot;
    }
    let entry = terrain_material(material);
    slot.depth = entry.relief.y;
    if slot.depth <= 0.0 && entry.params.w > 0.0 {
        slot.buried = 1.0 / max(entry.relief.x, 1.0e-3);
    }
    let ends = terrain_ends(entry, material, vec2<f32>(site.footprint.resolved,
        max(site.footprint.resolved, site.footprint.along / terrain_look.march.z)));
    slot.rungs = ends.rungs;
    slot.far = ends.maps;
    slot.faded = ends.albedo;
    slot.shown = ends.shown;
    slot.chain = ends.chain;
    if slot.faded < 1.0 || any(slot.shown > vec2<f32>(0.0)) {
        slot.mapping = terrain_mapping(entry, site.geometric);
        if slot.far > 0.0 {
            slot.mapping.planes = mix(slot.mapping.planes,
                terrain_fewer_planes(slot.mapping.planes), slot.far);
        }
    }
    return slot;
}

fn terrain_height_rank(material: u32, lowest: u32, height: f32) -> f32 {
    let at = terrain_rungs_at(material);
    return saturate((height - bitcast<f32>(terrain_tables[at + lowest]))
        * bitcast<f32>(terrain_tables[at + 11u]));
}

fn terrain_slot_pinned(slot: TerrainSlot) -> f32 {
    return slot.far * (1.0 - slot.chain.x - slot.chain.y);
}

fn terrain_slot_rank(held: TerrainSlot, material: u32, site: TerrainSite, contested: bool)
        -> f32 {
    var rank = 0.5;
    if contested && held.far > 0.0 {
        let fine = terrain_material(material).params.x;
        for (var rung = 0u; rung < 2u; rung++) {
            let part = held.chain[rung];
            if !terrain_detailed || part <= 0.0 {
                continue;
            }
            let mapping = terrain_rung_mapping(held, rung);
            let height = terrain_band_height(mapping, site.world, terrain_band_level(mapping,
                max(terrain_rung_blur(held.rungs, fine, rung), site.footprint.resolved)));
            rank += part * (terrain_height_rank(material, 12u + rung, height) - 0.5);
        }
    }
    if terrain_detailed && held.far < 1.0 {
        let height = terrain_pixel_height(held.mapping, site.world, site.world_dx,
            site.world_dy);
        rank = mix(terrain_height_rank(material, 7u, height), rank, held.far);
    }
    return rank;
}

fn terrain_slot_texels(slot: ptr<function, TerrainSlot>, site: TerrainSite) {
    let held = *slot;
    let hexed = terrain_hexed(held.mapping, site.world, site.world_dx, site.world_dy, -1.0);
    (*slot).texel = hexed.texel;
    (*slot).hexes = hexed.shares * (1.0 - held.far) + hexed.opened * held.far;
}

fn terrain_slot_kept(slot: ptr<function, TerrainSlot>, kept: f32) {
    (*slot).faded = 1.0 - kept * (1.0 - (*slot).faded);
    (*slot).far = 1.0 - kept * (1.0 - (*slot).far);
    (*slot).shown *= kept;
    (*slot).chain *= kept;
    (*slot).textured = kept > 0.0;
}

fn terrain_strata_at(point: vec3<f32>, thickness: f32) -> vec2<f32> {
    return vec2<f32>(dot(point.xz, terrain_look.strike.xy) * terrain_look.bedding.w,
        -dot(point, terrain_look.bedding.xyz) / thickness);
}

fn terrain_bedding(ground: vec3<f32>, geometric: vec3<f32>) -> vec4<f32> {
    let steep = smoothstep(terrain_look.slopes.x, terrain_look.slopes.y, length(geometric.xz));
    if steep <= 0.0 {
        return vec4<f32>(steep, 0.0, 0.0, 0.0);
    }
    let warp = terrain_look.strike.z * (terrain_value(ground.xz * terrain_look.strike.w
        + vec2<f32>(3.1, 8.7)) - 0.5);
    return vec4<f32>(steep, dot(ground.xz, terrain_look.strike.xy) * terrain_look.bedding.w,
        -dot(ground, terrain_look.bedding.xyz), warp);
}

fn terrain_strata(entry: PaletteEntry, site: TerrainSite) -> vec4<f32> {
    let steep = site.bedding.x;
    if steep <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let thickness = entry.heights.w;
    let at = vec2<f32>(site.bedding.y, site.bedding.z / thickness)
        - vec2<f32>(0.0, site.bedding.w / thickness);
    let dx = terrain_strata_at(site.world_dx, thickness);
    let dy = terrain_strata_at(site.world_dy, thickness);
    let layer = u32(entry.params.w) - 1u;
    let color = textureSampleGrad(terrain_albedo, terrain_material_sampler, at, layer, dx, dy).rgb;
    let raised = textureSampleGrad(terrain_material_data, terrain_material_sampler, at,
        2u * layer, dx, dy).g * 2.0 - 1.0;
    let mean = entry.far.rgb / max(entry.color.rgb, vec3<f32>(1.0e-4));
    let light = dot(color, TERRAIN_LUMA) / max(dot(mean, TERRAIN_LUMA), 1.0e-4);
    let tone = max(1.0 + steep * entry.relief.w * (light - 1.0), 0.0);
    let dipped = terrain_look.bedding.xyz;
    let along = dipped - site.geometric * dot(dipped, site.geometric);
    return vec4<f32>(along * (steep * entry.heights.z * raised), tone);
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
    let scale = entry.maps.x;
    let banding = entry.heights.w > 0.0 && slot.textured;
    var banded = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if banding {
        banded = terrain_strata(entry, site);
    }
    if terrain_detailed && slot.faded < 1.0 {
        let mapping = slot.mapping;
        let tiled = terrain_near(mapping, site.world, site.world_dx, site.world_dy,
            slot.hexes, slot.opened, site.geometric, select(0.0, scale, slot.far < 1.0), 0.0);
        let mapped = tiled.mapped;
        color = mix(entry.color.rgb * tiled.albedo, color, slot.faded);
        if slot.far < 1.0 {
            roughness = mix(entry.color.w * slot.texel.g, roughness, slot.far);
            occlusion = mix(1.0 + entry.maps.y * (slot.texel.r - 1.0), occlusion, slot.far);
            metalness *= mix(slot.texel.b, 1.0, slot.far);
            lost = mix(mapped.w, lost, slot.far);
            if scale > 0.0 {
                turned = mix(stable_normalize(mapped.xyz), turned, slot.far);
            }
            let near = weight * (1.0 - slot.far);
            (*ground).detailed += near;
            if entry.relief.z > 0.0 {
                (*ground).averaged += near * saturate(mapped.w / entry.relief.z);
            }
        }
    }
    for (var rung = 0u; rung < 2u; rung++) {
        if !terrain_detailed || slot.shown[rung] <= 0.0 {
            continue;
        }
        let mapping = terrain_rung_mapping(slot, rung);
        let held = table_floats(terrain_rungs_at(material) + 4u + 4u * rung);
        let spans = terrain_march_spans(site.world_dx, site.world_dy, terrain_look.march.z);
        let bent = select(0.0, slot.far * slot.chain[rung], scale > 0.0);
        let wide = terrain_near(mapping, site.world, spans.dx, spans.dy,
            mat3x3<f32>(), true, site.geometric,
            select(0.0, scale, bent > 0.0),
            terrain_rung_blur(slot.rungs, entry.params.x, rung) * mapping.scale);
        color *= mix(vec3<f32>(1.0), wide.albedo * held.rgb, slot.shown[rung]);
        if bent > 0.0 {
            turned = stable_normalize(turned
                + bent * (stable_normalize(wide.mapped.xyz) - site.geometric));
        }
    }
    let rim = select(0.0, slot.edge, slot.depth > 0.0);
    if slot.edge > 0.0 {
        let dimmed = 1.0 - slot.edge * select(terrain_look.far.z, terrain_look.far.w,
            slot.depth > 0.0);
        color *= dimmed;
        occlusion *= dimmed;
    }
    color *= terrain_site_gain(entry.heights.y, slot.lush, terrain_look.lush.rgb,
        terrain_look.bare.rgb);
    if banding {
        color *= banded.w;
        turned = stable_normalize(turned + banded.xyz);
    }
    if lost > 0.0 {
        let alpha = roughness * roughness;
        roughness = sqrt(sqrt(min(alpha * alpha + lost, 1.0)));
    }
    let far = cover_far(CoverSite(terrain_index, material, color, site.world, site.colors.a,
        site.toward, site.geometric, site.across * camera.pixel_error, site.footprint.resolved));
    let grown = far.shown * (1.0 - rim);
    color = mix(color, far.color, grown);
    roughness = mix(roughness, far.roughness, grown);
    occlusion = mix(occlusion, far.occlusion, grown);
    if grown > 0.0 {
        turned = stable_normalize(mix(turned, far.normal, grown));
    }
    (*ground).vegetated += weight * far.vegetated;
    (*ground).through += weight * grown * far.through;
    (*ground).shaded += weight * grown * (1.0 - far.sun);
    (*ground).canopy += weight * grown;
    (*ground).albedo += weight * color;
    (*ground).roughness += weight * roughness;
    (*ground).occlusion += weight * occlusion;
    (*ground).metalness += weight * metalness;
    var share = vec2<f32>(1.0);
    if entry.params.y > 0.5 && grown < 1.0 {
        share = mix(terrain_rough(material, terrain_rough_at(site.geometric, site.toward)),
            share, grown);
    }
    (*ground).share += weight * share;
    *normal += weight * turned;
}

struct TerrainShares {
    ids: vec4<u32>,
    weights: vec4<f32>,
    held: vec4<u32>,
    given: vec4<f32>,
    marks: u32,
    lip: f32,
    lush: f32,
}

const TERRAIN_MARK_REACHED: u32 = 0u;
const TERRAIN_MARK_KEPT: u32 = 4u;
const TERRAIN_MARK_PRESENT: u32 = 8u;
const TERRAIN_MARK_SOFT_LIPS: u32 = 12u;
const TERRAIN_MARK_EDGED: u32 = 13u;
const TERRAIN_MARK_SHARED: u32 = 16u;

fn terrain_marks(lanes: vec4<bool>, at: u32) -> u32 {
    return (select(0u, 1u, lanes.x) | select(0u, 2u, lanes.y) | select(0u, 4u, lanes.z)
        | select(0u, 8u, lanes.w)) << at;
}

fn terrain_marked(marks: u32, at: u32) -> bool {
    return ((marks >> at) & 1u) != 0u;
}

const TERRAIN_ABSENT: vec4<f32> = vec4<f32>(-1.0e9, -1.0e9, -1.0e9, -1.0e9);

fn terrain_cover_depth(own: vec4<f32>, shares: vec4<f32>, soil: f32) -> f32 {
    return dot(own, shares) + soil;
}

fn terrain_highest(heights: vec4<f32>) -> f32 {
    return max(max(heights.x, heights.y), max(heights.z, heights.w));
}

fn terrain_share_offset(share: f32) -> f32 {
    let lower = 0.5 - 0.5 * sqrt(2.0 * min(share, 1.0 - share));
    return select(lower, -lower, share < 0.5);
}

fn terrain_order(weights: vec4<f32>) -> vec4<u32> {
    var order = vec4<u32>(0u);
    for (var slot = 0u; slot < 4u; slot++) {
        var before = 0u;
        for (var other = 0u; other < 4u; other++) {
            if weights[other] > weights[slot]
                    || (weights[other] == weights[slot] && other < slot) {
                before++;
            }
        }
        order = terrain_lane_set_u(order, before, slot);
    }
    return order;
}

fn terrain_reached(base: vec4<f32>, present: vec4<f32>, pinned: vec4<f32>, band: f32)
        -> vec4<f32> {
    let least = terrain_highest(select(TERRAIN_ABSENT, base + 0.5 * pinned,
        present > vec4<f32>(0.5)));
    return present * select(vec4<f32>(0.0), vec4<f32>(1.0),
        base + vec4<f32>(1.0) - 0.5 * pinned >= vec4<f32>(least - band));
}

fn terrain_shares(site: TerrainSite, blend: TerrainBlend) -> TerrainShares {
    let present = select(vec4<f32>(0.0), vec4<f32>(1.0), blend.weights > vec4<f32>(0.0));
    let share = blend.weights / max(dot(blend.weights, vec4<f32>(1.0)), 1.0e-6);
    let resolved = site.footprint.resolved;
    var gone = vec4<f32>(0.0);
    var depths = vec4<f32>(0.0);
    var buried = vec4<f32>(0.0);
    var pinned = vec4<f32>(0.0);
    for (var slot = 0u; slot < 4u; slot++) {
        let material = blend.ids[slot];
        let part = share[slot];
        let held = terrain_slot(material, !(part <= 0.0), site);
        gone = terrain_lane_set(gone, slot, terrain_maps_gone(material, part, resolved));
        depths = terrain_lane_set(depths, slot, held.depth);
        buried = terrain_lane_set(buried, slot, held.buried);
        pinned = terrain_lane_set(pinned, slot, terrain_slot_pinned(held));
    }
    let soft = present * select(vec4<f32>(0.0), vec4<f32>(1.0), depths > vec4<f32>(0.0));
    let hard = present - soft;
    var soil = 0.0;
    var lush = 0.0;
    if terrain.macro_map.y != 0u {
        lush = terrain_site_lush(blend.site);
        soil = (terrain.pixel_error.z + blend.site.z * terrain.pixel_error.w)
            * dot(share, soft)
            * terrain_highest(hard * buried);
    }
    let base =vec4<f32>(terrain_share_offset(share.x), terrain_share_offset(share.y),
        terrain_share_offset(share.z), terrain_share_offset(share.w))
        + soft * terrain_cover_depth(depths, share, soil);
    let band = terrain_look.tiling.w;
    let open = terrain_reached(base, present, pinned, band);
    let contested = dot(open, hard) > 0.5 && dot(open, vec4<f32>(1.0)) > 1.5;
    let reached = select(terrain_reached(base, present, gone, band), open, contested);
    let handed = dot(share, gone);
    var marks = terrain_marks(reached > vec4<f32>(0.5), TERRAIN_MARK_REACHED)
        | terrain_marks(blend.weights > vec4<f32>(0.0), TERRAIN_MARK_PRESENT)
        | terrain_marks(!(share <= vec4<f32>(0.0)), TERRAIN_MARK_SHARED);
    var raised = TERRAIN_ABSENT;
    for (var slot = 0u; slot < 4u; slot++) {
        if terrain_marked(marks, TERRAIN_MARK_REACHED + slot) {
            let material = blend.ids[slot];
            raised = terrain_lane_set(raised, slot, base[slot] + terrain_slot_rank(
                terrain_slot(material, terrain_marked(marks, TERRAIN_MARK_SHARED + slot), site),
                material, site, contested));
        }
    }
    let highest = terrain_highest(raised);
    let lifted = max(raised - vec4<f32>(highest - band), vec4<f32>(0.0));
    let won = lifted / max(dot(lifted, vec4<f32>(1.0)), 1.0e-6);
    var weights = won;
    if handed > 0.0 && dot(soft, vec4<f32>(1.0)) > 1.5 {
        let among = share * soft / dot(share, soft);
        weights = mix(won, won * hard + dot(won, soft) * among, handed);
    }
    let given = weights;
    let covers = terrain_highest(select(TERRAIN_ABSENT, raised, soft > vec4<f32>(0.5)));
    let floors = terrain_highest(select(TERRAIN_ABSENT, raised, hard > vec4<f32>(0.5)));
    marks |= terrain_marks(won > vec4<f32>(0.0), TERRAIN_MARK_KEPT);
    var lip = 0.0;
    if covers > -1.0e8 && floors > -1.0e8 {
        let over = (covers - floors) * terrain_look.slopes.z;
        lip = saturate(1.0 - abs(over));
        marks |= select(0u, 1u << TERRAIN_MARK_SOFT_LIPS, over >= 0.0)
            | (1u << TERRAIN_MARK_EDGED);
    }
    let order = terrain_order(given);
    return TerrainShares(
        vec4<u32>(blend.ids[order.x], blend.ids[order.y], blend.ids[order.z],
            blend.ids[order.w]),
        vec4<f32>(given[order.x], given[order.y], given[order.z], given[order.w]),
        blend.ids, given, marks, lip, lush);
}

fn terrain_shared_ground(shares: TerrainShares, site: TerrainSite) -> TerrainGround {
    var ground: TerrainGround;
    var normal = vec3<f32>(0.0);
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = shares.given[slot];
        if weight <= 0.0 {
            continue;
        }
        let material = shares.held[slot];
        var held = terrain_slot(material,
            terrain_marked(shares.marks, TERRAIN_MARK_SHARED + slot), site);
        held.opened = !(terrain_detailed && terrain_marked(shares.marks, TERRAIN_MARK_REACHED + slot)
            && held.far < 1.0);
        held.lush = shares.lush;
        if !held.opened {
            terrain_slot_texels(&held, site);
        }
        if terrain_marked(shares.marks, TERRAIN_MARK_EDGED) {
            let present = select(0.0, 1.0,
                terrain_marked(shares.marks, TERRAIN_MARK_PRESENT + slot));
            let soft = present * select(0.0, 1.0, held.depth > 0.0);
            let hard = present - soft;
            held.edge = shares.lip
                * select(hard, soft, terrain_marked(shares.marks, TERRAIN_MARK_SOFT_LIPS))
                * (1.0 - held.far);
        }
        terrain_slot_kept(&held, select(0.0, 1.0,
            terrain_marked(shares.marks, TERRAIN_MARK_KEPT + slot)));
        terrain_slot_ground(&ground, &normal, material, weight, held, site);
    }
    ground.normal = stable_normalize(normal);
    return ground;
}

const TERRAIN_MESH_CARRIES: f32 = 2.0;

const TERRAIN_BAND_PIXELS: f32 = 1.0;

var<private> terrain_detailed: bool = true;

fn terrain_band_height(mapping: TerrainMapping, world: vec3<f32>, level: f32) -> f32 {
    return terrain_hexed(mapping, world, vec3<f32>(0.0), vec3<f32>(0.0), level).texel.a;
}

fn terrain_pixel_height(mapping: TerrainMapping, world: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    return terrain_hexed(mapping, world, dx, dy, -1.0).texel.a;
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
    mapping: TerrainMapping,
    mean: f32,
}

fn terrain_relief_layer(material: u32, depth: f32, mean: f32, site: TerrainSite)
        -> TerrainReliefLayer {
    var layer: TerrainReliefLayer;
    layer.depth = depth;
    layer.mean = mean;
    if depth > 0.0 {
        layer.mapping = terrain_mapping(terrain_material(material), site.geometric);
    }
    return layer;
}

fn terrain_relief_rise(layer: TerrainReliefLayer, point: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    if layer.depth <= 0.0 {
        return 0.0;
    }
    return layer.depth * (terrain_pixel_height(layer.mapping, point, dx, dy) - layer.mean);
}

struct TerrainLift {
    moved: vec3<f32>,
    kept: vec3<f32>,
}

fn terrain_material_rises(entry: PaletteEntry) -> bool {
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
        let mapping = terrain_mapping(entry, normal);
        let depth = weight * entry.relief.x;
        let band = terrain_band_height(mapping, world, terrain_band_level(mapping, meters.x));
        var coarse = band;
        if meters.y > meters.x {
            coarse = terrain_band_height(mapping, world,
                terrain_band_level(mapping, meters.y));
        }
        rise += depth * (vec2<f32>(band, coarse) - entry.heights.x);
    }
    return rise;
}

fn terrain_displacement(world: vec3<f32>, local: vec2<f32>, pieces: f32) -> TerrainLift {
    let across = max(detail_across(world), 1.0e-6);
    if terrain.relief.x < TERRAIN_BAND_PIXELS * across {
        return TerrainLift(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let shading = terrain_pixel_shading(world);
    let normal = terrain_surface_normal(local, shading);
    return terrain_lifted(terrain_blend(local, normal, shading), world, normal, across, pieces);
}

fn terrain_lifted(blend: TerrainBlend, world: vec3<f32>, normal: vec3<f32>, across: f32,
        pieces: f32) -> TerrainLift {
    let carried = terrain_carried(across);
    let spaced = max(carried, TERRAIN_MESH_CARRIES * pieces);
    let rise = terrain_band_rise(blend, world, normal, vec2<f32>(carried, spaced));
    return TerrainLift(normal * rise.x, normal * rise.y);
}

struct TerrainSpans {
    dx: vec3<f32>,
    dy: vec3<f32>,
}

fn terrain_march_spans(dx: vec3<f32>, dy: vec3<f32>, limit: f32) -> TerrainSpans {
    let wide = length(dx);
    let tall = length(dy);
    if wide > tall * limit {
        return TerrainSpans(dx, dy * (wide / max(tall * limit, 1.0e-12)));
    }
    if tall > wide * limit {
        return TerrainSpans(dx * (tall / max(wide * limit, 1.0e-12)), dy);
    }
    return TerrainSpans(dx, dy);
}

struct TerrainRelief {
    world: vec3<f32>,
    beyond: f32,
    moved: f32,
}

struct TerrainReliefGround {
    layer: TerrainReliefLayer,
    lifted: f32,
    dx: vec3<f32>,
    dy: vec3<f32>,
}

fn terrain_sink(ground: TerrainReliefGround, point: vec3<f32>, fade: f32) -> f32 {
    return (ground.lifted - terrain_relief_rise(ground.layer, point, ground.dx, ground.dy))
        * fade;
}

fn terrain_relief_sideways(layer: TerrainReliefLayer) -> bool {
    return layer.depth > 0.0 && layer.mapping.planes.y < 1.0;
}

const TERRAIN_RELIEF_REFINE: u32 = 2u;

const TERRAIN_MARCH_STEP: u32 = 0u;
const TERRAIN_MARCH_TOP: u32 = 1u;
const TERRAIN_MARCH_REFINE: u32 = 2u;

const TERRAIN_SLOT_PIXELS: f32 = 0.25;

fn terrain_relief_across(way: vec3<f32>, normal: vec3<f32>) -> f32 {
    let ray = way / max(length(way), 1.0e-6);
    let travel = ray / max(-dot(ray, normal), DETAIL_EDGE_ON);
    return min(length(travel.xz), terrain_look.march.w);
}

struct TerrainReliefSeen {
    depths: vec4<f32>,
    means: vec4<f32>,
    fade: f32,
    pixels: f32,
}

fn terrain_relief_seen(blend: TerrainBlend, world: vec3<f32>, normal: vec3<f32>)
        -> TerrainReliefSeen {
    let footprint = terrain_footprint(detail_span(world, normal));
    let to_point = world - camera.position.xyz;
    let ray = to_point / max(length(to_point), 1.0e-6);
    let grazing = smoothstep(0.0, DETAIL_EDGE_ON, -dot(ray, normal));
    var depths = vec4<f32>(0.0);
    var means = vec4<f32>(0.0);
    var kept = 0.0;
    for (var slot = 0u; slot < 4u; slot++) {
        let weight = blend.weights[slot];
        if weight <= 0.0 {
            continue;
        }
        let entry = terrain_material(blend.ids[slot]);
        if !terrain_material_rises(entry) {
            continue;
        }
        let depth = entry.relief.x * weight;
        depths = terrain_lane_set(depths, slot, depth);
        means = terrain_lane_set(means, slot, entry.heights.x);
        kept += depth
            * (1.0 - terrain_gone(footprint.resolved, entry.params.x, terrain_look.reach.x));
    }
    var seen: TerrainReliefSeen;
    seen.depths = depths;
    seen.means = means;
    seen.fade = grazing * kept / max(dot(depths, vec4<f32>(1.0)), 1.0e-6);
    seen.pixels = terrain_relief_across(world - camera.detail.xyz, normal) / footprint.along;
    return seen;
}

fn terrain_relief_range(depths: vec4<f32>, means: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(dot(depths, vec4<f32>(1.0) - means), dot(depths, means));
}

fn terrain_parallax(site: TerrainSite, blend: TerrainBlend, seen: TerrainReliefSeen,
        dither: f32, lifted: f32) -> TerrainRelief {
    var relief = TerrainRelief(site.world, 0.0, 0.0);
    let fade = seen.fade;
    let pixels = seen.pixels;
    let order = terrain_order(blend.weights);
    var material = blend.ids[order.x];
    var lead = 1.0;
    if blend.weights[order.y] > 0.0 {
        let shares = terrain_shares(site, blend);
        material = shares.ids.x;
        lead = shares.weights.x - shares.weights.y;
    }
    let entry = terrain_material(material);
    let mean = entry.heights.x;
    var risen = 0.0;
    if terrain_material_rises(entry) {
        risen = entry.relief.x * lead;
    }
    if risen * max(mean, 1.0 - mean) * (fade * pixels) < TERRAIN_SLOT_PIXELS {
        risen = 0.0;
    }
    let top = min(lifted - risen * (1.0 - mean), 0.0) * fade;
    let depth = max(lifted + risen * mean, 0.0) * fade;
    if max(-top, depth) * pixels < TERRAIN_BAND_PIXELS {
        return relief;
    }
    var ground: TerrainReliefGround;
    ground.layer = terrain_relief_layer(material, risen, mean, site);
    ground.lifted = lifted;
    let spans = terrain_march_spans(site.world_dx, site.world_dy, terrain_look.march.x);
    ground.dx = spans.dx;
    ground.dy = spans.dy;
    let to_point = site.world - camera.position.xyz;
    let ray = to_point / max(length(to_point), 1.0e-6);
    let travel = ray / max(-dot(ray, site.geometric), DETAIL_EDGE_ON);
    let across = length(travel.xz);
    let along = travel.xz * min(1.0, terrain_look.march.w / max(across, 1.0e-6));
    let rise = -site.geometric.xz / max(site.geometric.y, 0.05);
    let march = vec3<f32>(along.x, dot(rise, along), along.y);
    var start = site.world;
    if terrain_relief_sideways(ground.layer) {
        let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
        if held.found {
            start.y = held.height + terrain.origin.y;
        }
    }
    let shifted = (depth - top) * pixels;
    let count = clamp(shifted, 1.0, terrain.relief.y);
    let steps = max(u32(floor(count + dither)), 1u);
    let refines = select(0u, TERRAIN_RELIEF_REFINE, shifted > terrain_look.march.y * count);
    var previous = top;
    var gap = 0.0;
    var hit = depth;
    var low = vec2<f32>(0.0);
    var high = vec2<f32>(0.0);
    var step = 1u;
    var refine = 0u;
    var stage = TERRAIN_MARCH_STEP;
    var probe = top + (depth - top) * f32(step) / f32(steps);
    loop {
        let probe_gap = probe - terrain_sink(ground, start + march * probe, fade);
        if stage == TERRAIN_MARCH_STEP {
            if !(probe_gap >= 0.0) {
                previous = probe;
                gap = probe_gap;
                step++;
                if step > steps {
                    break;
                }
                probe = top + (depth - top) * f32(step) / f32(steps);
                continue;
            }
            high = vec2<f32>(probe, probe_gap);
            if step == 1u {
                stage = TERRAIN_MARCH_TOP;
                probe = top;
                continue;
            }
        } else if stage == TERRAIN_MARCH_TOP {
            gap = min(probe_gap, 0.0);
        } else {
            if probe_gap >= 0.0 {
                high = vec2<f32>(probe, probe_gap);
            } else {
                low = vec2<f32>(probe, probe_gap);
            }
            let nearer = mix(low.x, high.x, low.y / min(low.y - high.y, -1.0e-6));
            let settled = abs(nearer - probe) * pixels < terrain_look.march.y;
            hit = nearer;
            refine++;
            if settled || refine >= refines {
                break;
            }
            probe = hit;
            continue;
        }
        low = vec2<f32>(previous, gap);
        hit = mix(low.x, high.x, low.y / min(low.y - high.y, -1.0e-6));
        if refines == 0u {
            break;
        }
        stage = TERRAIN_MARCH_REFINE;
        probe = hit;
    }
    relief.world = site.world + march * hit;
    relief.beyond = hit * length(travel);
    relief.moved = abs(hit) * pixels;
    return relief;
}

fn terrain_point(relative: vec3<f32>) -> SurfacePoint {
    var point = terrain_point_seen(relative, terrain_seen(relative,
        terrain_forward_look(relative)), false);
    let world = eye_world(relative);
    point.world_dx = dpdx(world);
    point.world_dy = dpdy(world);
    point.normal_dx = dpdx(point.normal);
    point.normal_dy = dpdy(point.normal);
    return point;
}

struct TerrainLook {
    ray_dx: vec3<f32>,
    ray_dy: vec3<f32>,
    dither: f32,
}

fn terrain_forward_look(relative: vec3<f32>) -> TerrainLook {
    let ray = relative / max(length(relative), 1.0e-6);
    return TerrainLook(dpdx(ray), dpdy(ray), 0.5);
}

struct TerrainSeen {
    site: TerrainSite,
    shading: TerrainShadingLevel,
    blend: TerrainBlend,
    beyond: f32,
    moved: f32,
}

fn terrain_ground_blend(seen: TerrainSeen, site: TerrainSite) -> TerrainBlend {
    if seen.moved < TERRAIN_BAND_PIXELS {
        return seen.blend;
    }
    return terrain_blend(site.world.xz - terrain.origin.xz, site.geometric, seen.shading);
}

fn terrain_seen(relative: vec3<f32>, look: TerrainLook) -> TerrainSeen {
    let drawn = eye_world(relative);
    let away = length(relative);
    let ray = relative / max(away, 1.0e-6);
    let ray_dx = look.ray_dx;
    let ray_dy = look.ray_dy;
    let local = drawn.xz - terrain.origin.xz;
    let reach = max(detail_across(drawn), 1.0e-6);
    let shading = terrain_shading_level(reach);
    let geometric = terrain_surface_normal(local, shading);
    let blend = terrain_blend(local, geometric, shading);
    var rises: TerrainReliefSeen;
    var marches = false;
    var lift = vec3<f32>(0.0);
    if terrain_detailed && terrain.relief.x >= TERRAIN_BAND_PIXELS * reach {
        rises = terrain_relief_seen(blend, drawn, geometric);
        let range = terrain_relief_range(rises.depths, rises.means);
        marches = (range.x + range.y) * rises.fade * rises.pixels >= TERRAIN_BAND_PIXELS;
        if marches || max(range.x, range.y) >= TERRAIN_BAND_PIXELS * reach {
            lift = terrain_lifted(blend, drawn, geometric, reach, 0.0).moved;
        }
    }
    let world = drawn - lift;
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
    var relief = TerrainRelief(world, 0.0, 0.0);
    if marches {
        relief = terrain_parallax(site, blend, rises, look.dither, dot(lift, geometric));
    }
    site.colors = terrain_macro_colors(site);
    site.across = terrain_across(world);
    site.bedding = terrain_bedding(world, geometric);
    site.world = relief.world;
    return TerrainSeen(site, shading, blend, relief.beyond, relief.moved);
}

fn terrain_point_seen(relative: vec3<f32>, seen: TerrainSeen, marched: bool) -> SurfacePoint {
    let site = seen.site;
    var surface = site;
    let held = terrain_level_height(site.world.xz - terrain.origin.xz, 0u);
    if held.found {
        surface.world.y = held.height + terrain.origin.y;
    }
    var ground = terrain_shared_ground(
        terrain_shares(surface, terrain_ground_blend(seen, surface)), surface);
    let world = eye_world(relative);
    let local = world.xz - terrain.origin.xz;
    let geometric = site.geometric;
    ground.albedo *= terrain_tint(ground, site);
    let cover = far_cover(FarGround(ground.albedo, ground.roughness, ground.normal,
        ground.occlusion, 0.0, 0.0, 1.0), GroundAt(world, geometric, site.toward, site.across,
        site.footprint.resolved, 0.0, 0.0));
    var albedo = cover.color;
    let normal = cover.normal;
    let occlusion = cover.occlusion;
    var roughness = cover.roughness;
    let bare = 1.0 - ground.canopy;
    let open = 1.0 - cover.shown;
    var wet = 0.0;
    if bare != 0.0 && open != 0.0 {
        let depth_point = select(relative,
            relative + relative * (seen.beyond / max(length(relative), 1.0e-6)), marched);
        wet = max(seen.blend.wet, terrain_waterline(eye_world(depth_point), site)) * bare * open;
    }
    if wet > 0.0 {
        let soaked = terrain_wet_ground(roughness, wet);
        albedo *= soaked.x;
        roughness = soaked.y;
    }
    roughness = max(roughness, 0.03);
    let sky_map = terrain_sky_map(local);
    let sky = sky_map.r;
    var point = standard_point();
    point.base_color = albedo;
    point.opacity = 1.0;
    point.emissive = vec3<f32>(0.0);
    point = set_added_indirect(point, bounce_light(world, normal, true).light * occlusion);
    point = set_ambient_bend(point, sky_map.gba * 2.0 - 1.0);
    point.geometric = geometric;
    point.normal = normal;
    point.distance = length(relative);
    point.view = stable_normalize(-relative);
    point.world = world;
    let detail = material_detail();
    point.world_dx = site.world_dx / detail;
    point.world_dy = site.world_dy / detail;
    point.kind = SURFACE_GROUND;
    point = set_ignore_face(point, true);
    point = set_receiver_lift_first(point, kept_texels(world).first);
    point = set_receiver_lifts(point, vec4<f32>(cover.lift));
    point = set_roughness(point, roughness);
    point = set_metalness(point, ground.metalness);
    point = set_occlusion(point, occlusion * sky);
    point = set_transmission(point, min(foliage().x * ground.through, ground.canopy));
    point = set_tangled(point, select(0.0, 1.0, ground.canopy > 0.0));
    point = set_mirrored(point, 0.0);
    point = set_direct_blocked(point,
        1.0 - cover.sun * (1.0 - ground.shaded * (1.0 - cover.shown)));
    point = set_direct_share(point, mix(ground.share, vec2<f32>(1.0), cover.shown));
    return derived(point);
}

fn terrain_ids(input: TerrainOutput, holes: bool) -> vec2<u32> {
    use_terrain(input.terrain);
    terrain_holes_test(input.cell, holes);
    return vec2<u32>(terrain.directory.x + 1u, bitcast<u32>(input.clip_position.z));
}

@fragment
fn fs_terrain_ids(input: TerrainOutput) -> @location(0) vec2<u32> {
    return terrain_ids(input, false);
}

@fragment
fn fs_terrain_ids_holes(input: TerrainOutput) -> @location(0) vec2<u32> {
    return terrain_ids(input, true);
}

fn terrain_surfaced(input: TerrainOutput, holes: bool) -> KindPoint {
    use_terrain(input.terrain);
    let point = terrain_point(input.relative.xyz);
    terrain_holes_test(input.cell, holes);
    let world = eye_world(input.relative.xyz);
    let previous = camera.previous_view_projection * vec4<f32>(world, 1.0);
    let at: ViewPixel = view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.w));
    return KindPoint(point, input.clip_position.xy, at);
}

fn terrain_surface(input: TerrainOutput) -> KindPoint {
    return terrain_surfaced(input, false);
}

fn terrain_surface_holes(input: TerrainOutput) -> KindPoint {
    return terrain_surfaced(input, true);
}
