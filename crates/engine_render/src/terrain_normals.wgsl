// Terrain normal maps: a tile's normals from its heights,
// once when the tile uploads, into the same layer of the normal array.
// Fragments read them whatever level of detail the geometry draws at, so
// the ground's light holds still as levels hand over.

struct Job {
    // The layer, and the tile's sample spacing (as bits).
    layer: u32,
    spacing: u32,
    pad: vec2<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> job: Job;
@group(0) @binding(2) var normals: texture_storage_2d<rgba8snorm, write>;

// Texels along a layer's side: a tile's samples and their border.
const TILE_STRIDE: u32 = 259u;

fn height(x: u32, z: u32) -> f32 {
    return textureLoad(heights, vec2<u32>(x, z), job.layer, 0).r;
}

@compute @workgroup_size(16, 16)
fn tile_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    // Border texels take their neighbor's normal; only samples 0 to 256 are
    // ever read.
    let x = clamp(id.x, 1u, TILE_STRIDE - 2u);
    let z = clamp(id.y, 1u, TILE_STRIDE - 2u);
    let spacing = bitcast<f32>(job.spacing);
    let normal = normalize(vec3<f32>(height(x - 1u, z) - height(x + 1u, z), 2.0 * spacing,
        height(x, z - 1u) - height(x, z + 1u)));
    textureStore(normals, id.xy, vec4<f32>(normal, 0.0));
}
