import package::face::cover::material_canopy;
import package::face::ground_tiles::PALETTE_ENTRIES;
import package::generated::reflections::{CANOPY_GROUP, TraceTerrain};

@group(1) @binding(0) var<storage, read> world_terrains: array<TraceTerrain>;
@group(1) @binding(1) var<storage, read_write> world_placed: array<u32>;

@compute @workgroup_size(CANOPY_GROUP)
fn cs_canopy(@builtin(global_invocation_id) id: vec3<u32>) {
    let terrain = world_terrains[id.y];
    let at = terrain.maps.z + 4u * id.x;
    if id.x >= PALETTE_ENTRIES || at + 4u > arrayLength(&world_placed) {
        return;
    }
    let canopy = bitcast<vec4<u32>>(material_canopy(terrain.maps.x, id.x));
    world_placed[at] = canopy.x;
    world_placed[at + 1u] = canopy.y;
    world_placed[at + 2u] = canopy.z;
    world_placed[at + 3u] = canopy.w;
}
