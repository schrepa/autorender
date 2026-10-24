// The runtime virtual texture's coarser far pages, each the mean of the
// next finer far mip (`terrain_vt/far.rs`): every texel of such a page, its
// border too, the mean of the 2x2 texels of the finer mip under it, read
// where the finer mip's pages lie in the sheets, so a coarser page is the
// average of the finer one by construction. Reading decodes the sheets'
// sRGB into the values pages hold; the means average those and encode them
// as the sheets store them, into rows the copies into the sheets take.

// Texels along a page's side, the border around it and a slot's side
// (`terrain_vt.rs`), the layers of the sheets a slot spans (base color,
// canopy and normal turn), and the bytes of a slot's row as copies take it.
const VT_PAGE: i32 = 128;
const VT_BORDER: i32 = 8;
const VT_SLOT: i32 = 144;
const VT_SHEET_LAYERS: u32 = 3u;
const VT_ROW_BYTES: u32 = 768u;

// A coarser page to make: its page along x and z, the finer mip's last page
// along x and z, the slots of the finer mip's pages from one before twice
// its page on (four rows of four along x, rows along z, held within the
// finer mip's pages), and its own slot.
struct FarMean {
    page: vec2<i32>,
    last: vec2<i32>,
    fine: array<vec4<u32>, 4>,
    slot: u32,
    pad: array<u32, 3>,
}

// The sheets, the pages to make, and their texels, each page's three layers
// in turn, row by row.
@group(0) @binding(0) var far_sheets: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> far_means: array<FarMean>;
@group(0) @binding(2) var<storage, read_write> far_texels: array<u32>;

// Slot `slot`'s first texel in its sheet (x, y) and the sheet's first layer
// (z): as many slots along a sheet's side as its texels hold.
fn far_place(slot: u32) -> vec3<i32> {
    let per_row = textureDimensions(far_sheets).x / u32(VT_SLOT);
    let per_sheet = per_row * per_row;
    let within = slot % per_sheet;
    let corner = vec2<i32>(vec2<u32>(within % per_row, within / per_row)) * VT_SLOT;
    return vec3<i32>(corner, i32(slot / per_sheet * VT_SHEET_LAYERS));
}

// One texel (x, y) of a slot of coarser page `z`: the mean of the four finer
// texels under it, each layer's.
@compute @workgroup_size(8, 8)
fn far_mean(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(VT_SLOT) || id.y >= u32(VT_SLOT) || id.z >= arrayLength(&far_means) {
        return;
    }
    let mean = far_means[id.z];
    // The coarser mip's texel this one holds, and the finer mip's last.
    let texel = mean.page * VT_PAGE - VT_BORDER + vec2<i32>(id.xy);
    let highest = (mean.last + vec2<i32>(1)) * VT_PAGE - vec2<i32>(1);
    var sums = array<vec4<f32>, VT_SHEET_LAYERS>();
    for (var corner = 0; corner < 4; corner++) {
        // The finer texel, held within the finer mip's pages, and the page
        // holding it.
        let fine = clamp(2 * texel + vec2<i32>(corner & 1, corner >> 1u), vec2<i32>(0),
            highest);
        let page = fine / VT_PAGE;
        let cell = page - (2 * mean.page - vec2<i32>(1));
        let place = far_place(far_means[id.z].fine[cell.y][cell.x]);
        let at = place.xy + vec2<i32>(VT_BORDER) + fine - page * VT_PAGE;
        for (var layer = 0; layer < i32(VT_SHEET_LAYERS); layer++) {
            sums[layer] += textureLoad(far_sheets, at, place.z + layer, 0);
        }
    }
    for (var layer = 0u; layer < VT_SHEET_LAYERS; layer++) {
        let value = sums[layer] * 0.25;
        let row = (id.z * VT_SHEET_LAYERS + layer) * u32(VT_SLOT) + id.y;
        far_texels[row * (VT_ROW_BYTES / 4u) + id.x] =
            pack4x8unorm(vec4<f32>(encode_srgb(value.rgb), value.a));
    }
}
