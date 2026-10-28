// The live water's bed (`water_sim.rs`): the ground under the live
// square's cells (`water_sim.wgsl`'s `sim_ground`) raised by the relief of
// the ground's materials, with the heights and the blend the ground draws
// its stones and cobbles with (`terrain_band_rise`), each read at the level
// whose texels span a live cell: a level the cells' spacing sets, not the
// view, so the bed is a function of where it stands alone, its materials
// blended at the finest level the terrain shades with, as the ground near
// the eye shades. The relief lies along the ground's smooth normal, as the
// mesh band moves the drawn ground, and raises each corner as high as it
// raises the drawn ground's point there, its rise times the normal's
// upright part: the stones stand where the drawn ones do, within the
// sideways part of their rise, and no higher than their depth on any
// slope. Each block the live water takes into a slot of its pool rises
// from the ground its window gathered (`engine_core`'s `water_sim.rs`),
// kept for it, in the frame that copies it there, before any step reads
// it, and rises again from it now and then after, so its bed follows the
// ground's materials as they arrive and the tiles under it fade in; a
// workgroup a block, after an empty first group and the terrain group of
// the terrain the window stands on, read as the ground's mesh band reads
// it. A lake holds none of its cells whose raised ground stands at or over
// its level, as a lake holds no cell it does not stand over.

// The blocks whose bed the live water raises this frame, then nothing; and
// the meters across their cells, then nothing.
struct SimBed {
    blocks: vec4<u32>,
    cell: vec4<f32>,
}

@group(2) @binding(41) var<uniform> bed_frame: SimBed;
// Each block whose bed rises: the slot of the live water's pool it stands
// in, the cell of the terrain's lattice of live cells its first corner
// stands at (an `i32`'s bits along x and z), and the height its heights
// stand above (world meters, a float's bits).
@group(2) @binding(42) var<storage, read> bed_taken: array<vec4<u32>>;
// The live water's ground, WATER_BED_GROUND entries a block of its pool:
// the ground at its cells' corners over its datum (meters, BED_SIDE a row,
// rows along z), then the level a lake holds each of its cells at
// (WATER_BED_UNHELD where none does), then what the solve reads past them.
@group(2) @binding(43) var<storage, read_write> bed_ground: array<f32>;
// The ground each slot's block was taken with, which its bed rises from:
// BED_KEPT entries a slot, its ground's first, its corners' and its lakes'.
@group(2) @binding(44) var<storage, read> bed_kept: array<f32>;

// Entries of a block of the live water's ground, and what a cell no lake
// holds holds as its lake's level (`engine_terrain::BLOCK_GROUND`,
// `shallow_water::UNHELD`).
const WATER_BED_GROUND: u32 = 337u;
const WATER_BED_UNHELD: f32 = -1.0e30;
// Corners along a side of a block, a block's corners, and the entries of
// its ground its bed rises from.
const BED_SIDE: u32 = WATER_LIVE_BLOCK + 1u;
const BED_CORNERS: u32 = BED_SIDE * BED_SIDE;
const BED_KEPT: u32 = BED_CORNERS + WATER_LIVE_BLOCK * WATER_LIVE_BLOCK;

// A block's corners as its workgroup raises them.
var<workgroup> bed_corners: array<f32, BED_CORNERS>;

// Corner `corner` (x and z, 0 to WATER_LIVE_BLOCK) of the block `taken`
// lists, whose ground stands `ground` meters over its datum there, raised
// by the relief of the ground's materials (`terrain_band_rise`) at the
// level whose texels span a live cell: the ground's smooth normal, its
// blend and its materials' heights read at the finest level the terrain
// shades with.
fn bed_raised(taken: vec4<u32>, corner: vec2<u32>, ground: f32) -> f32 {
    let cell = bed_frame.cell.x;
    let local = vec2<f32>(bitcast<vec2<i32>>(taken.yz) + vec2<i32>(corner)) * cell;
    let world = vec3<f32>(terrain.origin.x + local.x, bitcast<f32>(taken.w) + ground,
        terrain.origin.z + local.y);
    let shading = TerrainShadingLevel(0u, 0.0);
    let normal = terrain_surface_normal(local, shading);
    let blend = terrain_blend(local, normal, shading);
    return ground + terrain_band_rise(blend, world, normal, vec2<f32>(cell)).x * normal.y;
}

// The live water's blocks whose bed rises this frame, a workgroup each,
// once the copies of those taken landed in their slots: each invocation
// raises the corner its cell starts at from the ground kept for it, those
// along the block's last column and row its far corners too
// (`bed_raised`); then each cell's lake holds it at its level where its
// raised ground, the mean of its corners as the solve reads it
// (`water_sim.wgsl`'s `sim_bed`), stands under that level, and lets it go
// elsewhere.
@compute @workgroup_size(WATER_LIVE_BLOCK, WATER_LIVE_BLOCK)
fn raise_bed(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let block = group.y * count.x + group.x;
    if block >= bed_frame.blocks.x {
        return;
    }
    let taken = bed_taken[block];
    let base = taken.x * WATER_BED_GROUND;
    let kept = taken.x * BED_KEPT;
    let last = vec2<u32>(WATER_LIVE_BLOCK - 1u);
    for (var k = 0u; k < 4u; k++) {
        let far = vec2<u32>(k & 1u, k >> 1u);
        if any((far > vec2<u32>(0u)) & (local.xy < last)) {
            continue;
        }
        let corner = local.xy + far;
        let at = corner.y * BED_SIDE + corner.x;
        let raised = bed_raised(taken, corner, bed_kept[kept + at]);
        bed_ground[base + at] = raised;
        bed_corners[at] = raised;
    }
    workgroupBarrier();
    let first = local.y * BED_SIDE + local.x;
    let raised = 0.5 * (0.5 * (bed_corners[first] + bed_corners[first + BED_SIDE])
        + 0.5 * (bed_corners[first + 1u] + bed_corners[first + BED_SIDE + 1u]));
    let cell = local.y * WATER_LIVE_BLOCK + local.x;
    let level = bed_kept[kept + BED_CORNERS + cell];
    bed_ground[base + BED_CORNERS + cell] = select(WATER_BED_UNHELD, level, level > raised);
}
