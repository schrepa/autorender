// Tile classification of the visibility buffer (decision 045): which 8x8
// tiles hold pixels of each material bin, so each bin's resolve draws only
// those. A tile lists a bin once, by its first pixel of that bin.

struct Classify {
    // The viewport's size in pixels, the tiles across it and each bin's
    // room for tiles (every tile of the viewport).
    size: vec2<u32>,
    tiles_across: u32,
    capacity: u32,
    // Bins drawn this phase.
    bins: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> classify: Classify;
// Each pixel's visible entry plus one (0: none) and triangle | bin << 7.
@group(0) @binding(1) var ids: texture_2d<u32>;
// Each bin's tiles, then its tile list from `bin * capacity`.
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;
// Per bin: vertices, instances, first vertex, first instance.
@group(0) @binding(4) var<storage, read_write> draw_args: array<u32>;

const NONE: u32 = 0xffffffffu;
const TILE_PIXELS: u32 = 64u;

var<workgroup> tile_bins: array<u32, TILE_PIXELS>;

@compute @workgroup_size(8, 8)
fn classify_tiles(@builtin(workgroup_id) tile: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(global_invocation_id) pixel: vec3<u32>) {
    var bin = NONE;
    if all(pixel.xy < classify.size) {
        let id = textureLoad(ids, pixel.xy, 0).xy;
        if id.x != 0u {
            bin = id.y >> 7u;
        }
    }
    tile_bins[lane] = bin;
    workgroupBarrier();
    if bin == NONE || bin >= classify.bins {
        return;
    }
    for (var other = 0u; other < lane; other++) {
        if tile_bins[other] == bin {
            return;
        }
    }
    let slot = atomicAdd(&counts[bin], 1u);
    if slot < classify.capacity {
        tiles[bin * classify.capacity + slot] = tile.y * classify.tiles_across + tile.x;
    }
}

// Each bin's draw: two triangles per listed tile.
@compute @workgroup_size(64)
fn resolve_args(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= classify.bins {
        return;
    }
    draw_args[id.x * 4u] = 6u;
    draw_args[id.x * 4u + 1u] = min(atomicLoad(&counts[id.x]), classify.capacity);
    draw_args[id.x * 4u + 2u] = 0u;
    draw_args[id.x * 4u + 3u] = 0u;
}
