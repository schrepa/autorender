// Terrain normal maps: a tile's smooth surface at its samples
// (`terrain_tile.wgsl`'s `tile_shape`), once when the tile uploads, into the
// same layer of the normal array: each texel the surface's gradient there,
// its rise along x and z as the two halves of a float, precise on the
// gentlest slopes. Level-0 tiles and surroundings' tiles make theirs here;
// a terrain's coarser tiles upload theirs from their shading, level 0's
// averaged (`engine_terrain::TileShading`), in the same words. Fragments
// blend the gradients around them whatever level of detail the geometry
// draws at, as `engine_terrain::Tile::sample` blends them for gameplay, so
// the ground's light holds still as levels hand over.

struct Job {
    // The layer, and the tile's sample spacing (as bits).
    layer: u32,
    spacing: u32,
    pad: vec2<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> job: Job;
@group(0) @binding(2) var normals: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16)
fn tile_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let gradient = tile_shape(id.x, id.y, bitcast<f32>(job.spacing)).xy;
    textureStore(normals, id.xy, vec4<u32>(pack2x16float(gradient), 0u, 0u, 0u));
}
