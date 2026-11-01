import package::place::heightfield::{SurfaceTriangle, TerrainGroundRise, TerrainGroundSamples,
    TerrainLevelHeight, TerrainUniform, TerrainView};
import package::engine::view::camera;
import package::terrain_shading::draw::{TerrainBlend, TerrainShadingLevel, terrain_band_rise, terrain_border_blend, terrain_control, terrain_control_sampler, terrain_controls_blend, terrain_gradient_between, terrain_layer_fades, terrain_material, terrain_material_rises, terrain_normals, terrain_steep_blend};
import package::terrain_shading::field::{surface_along, terrain_directory, terrain_load};
import package::generated::terrain_shading::{PATCH_GRID, SUBDIVISION_ARGS, SUBDIVISION_LEAVES, SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES};

@group(0) @binding(0) var<storage, read> terrain_views: array<TerrainView>;

@group(0) @binding(0) var<storage, read> terrains: array<TerrainUniform>;

@group(0) @binding(0) var<storage, read> terrain_surface_leaves: array<SurfaceTriangle>;

@group(0) @binding(0) var<storage, read> terrain_surface_args: array<u32>;

const TERRAIN_GRID: u32 = PATCH_GRID;

const SURFACE_LEAF_INDICES: u32 = SURFACE_MESHLET_INDICES;

var<private> terrain: TerrainUniform;

var<private> terrain_view: TerrainView;

var<private> terrain_index: u32;

fn use_terrain(index: u32) {
    terrain_index = index;
    terrain = terrains[index];
    terrain_view = terrain_views[camera.view.x];
}

fn terrain_uniform() -> TerrainUniform {
    return terrain;
}

fn terrain_view_uniform() -> TerrainView {
    return terrain_view;
}

fn surface_leaf(index: u32) -> SurfaceTriangle {
    return terrain_surface_leaves[terrain.water.z + index];
}

fn surface_vertex(triangle: SurfaceTriangle, vertex: u32) -> vec2<f32> {
    let parts = 1u << SURFACE_MESHLET_LEVEL;
    var row = parts;
    var first = SURFACE_MESHLET_VERTICES - parts - 1u;
    while vertex < first {
        first -= row;
        row--;
    }
    let j = vertex - first;
    let i = row - j;
    var tail = triangle.c;
    var head = triangle.b;
    var steps = j;
    if row == parts {
        tail = triangle.a;
    } else if j == 0u {
        head = triangle.a;
        steps = i;
    } else if i != 0u {
        let share = vec2<f32>(f32(i), f32(j)) / f32(parts);
        return triangle.c + (triangle.a - triangle.c) * share.x
            + (triangle.b - triangle.c) * share.y;
    }
    return surface_along(tail, head, steps);
}

fn terrain_level_height(local: vec2<f32>, level: u32) -> TerrainLevelHeight {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var held: TerrainLevelHeight;
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
        let texel = corner + vec2<u32>(1u);
        let h00 = terrain_load(layer, texel);
        let h10 = terrain_load(layer, texel + vec2<u32>(1u, 0u));
        let h01 = terrain_load(layer, texel + vec2<u32>(0u, 1u));
        let h11 = terrain_load(layer, texel + vec2<u32>(1u, 1u));
        held.found = true;
        if f.x + f.y <= 1.0 {
            held.height = h00 + f.x * (h10 - h00) + f.y * (h01 - h00);
        } else {
            held.height = h11 + (1.0 - f.x) * (h01 - h11) + (1.0 - f.y) * (h10 - h11);
        }
        return held;
    }
    return held;
}

fn terrain_ground_samples(local: vec2<f32>) -> TerrainGroundSamples {
    var samples = TerrainGroundSamples(vec4<u32>(0u), vec4<u32>(0u), vec2<f32>(0.0));
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let at = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(at / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu || terrain_layer_fades(layer).y >= 1.0 {
            continue;
        }
        let within = clamp(at - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let uv = (vec2<f32>(corner) + vec2<f32>(2.0)) / 259.0;
        samples.controls = textureGather(0, terrain_control, terrain_control_sampler, uv, layer);
        samples.gradients = textureGather(0, terrain_normals, terrain_control_sampler, uv, layer);
        samples.within = within - vec2<f32>(corner);
        return samples;
    }
    return samples;
}

fn terrain_ground_rise(samples: TerrainGroundSamples, local: vec2<f32>, height: f32,
        meters: f32) -> TerrainGroundRise {
    let gradient = terrain_gradient_between(samples.gradients, samples.within);
    let normal = normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    terrain_controls_blend(&blend, samples.controls, samples.within, 1.0);
    terrain_steep_blend(&blend, normal);
    terrain_border_blend(&blend, local, TerrainShadingLevel(0u, 0.0));
    let world = terrain.origin.xyz + vec3<f32>(local.x, height, local.y);
    var raised = TerrainGroundRise(0.0, false);
    raised.rise = terrain_band_rise(blend, world, normal, vec2<f32>(meters)).x * normal.y;
    for (var slot = 0u; slot < 4u; slot++) {
        if blend.weights[slot] > 0.0 && terrain_material_rises(terrain_material(blend.ids[slot])) {
            raised.relief = true;
        }
    }
    return raised;
}

fn surface_leaf_count() -> u32 {
    if terrain.water.y == 0u {
        return 0u;
    }
    return terrain_surface_args[(terrain.water.y - 1u) * SUBDIVISION_ARGS + SUBDIVISION_LEAVES];
}

fn bed_height(at: vec2<f32>) -> f32 {
    return terrain_level_height(at, 0u).height;
}
