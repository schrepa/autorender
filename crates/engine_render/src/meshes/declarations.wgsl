// Mesh drawing, after scene.wgsl: these declarations, then vertex stages
// (vertex.wgsl), shading (shading.wgsl), forward fragments (forward.wgsl),
// the visibility resolve (resolve.wgsl), the sky, decals and impostors.
// Opaque entry points write the canvas, the screen effects' scene targets
// (decision 021) or temporal targets (decision 017); `_lit` variants add
// clustered local lights (decision 023); transparent ones accumulate
// weighted blended transparency (decision 026). Unlit entry points never
// touch the lights' bindings.

// Material maps (decision 031): glTF base color, metallic-roughness, normal,
// occlusion and emissive textures, each with its sampler, and the material's
// map parameters. Materials without maps bind white textures.
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_sampler: sampler;
@group(1) @binding(4) var normal_texture: texture_2d<f32>;
@group(1) @binding(5) var normal_sampler: sampler;
@group(1) @binding(6) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(7) var occlusion_sampler: sampler;
@group(1) @binding(8) var emissive_texture: texture_2d<f32>;
@group(1) @binding(9) var emissive_sampler: sampler;
struct MaterialMaps {
    // Rows of each map's texture coordinate transform, in map order.
    transforms: array<vec4<f32>, 10>,
    // Linear emissive color.
    emissive: vec4<f32>,
    // Normal scale, occlusion strength and alpha cutoff.
    params: vec4<f32>,
    // Transmission (decision 038): its share, index of refraction and
    // thickness; w holds the material's variant bits (decision 049).
    volume: vec4<f32>,
    // The top layer (decision 093): its linear color and coverage, then
    // its roughness. The meters one repeat of world-mapped maps covers are
    // in `emissive.w`.
    top: vec4<f32>,
    layer: vec4<f32>,
}
@group(1) @binding(10) var<uniform> material_maps: MaterialMaps;

// The material's pipeline variant (decision 031). Without MATERIAL_MAPS only
// the base color texture is read, untransformed, and its alpha masks
// emission alone; the other flags need it except DOUBLE_SIDED.
override MATERIAL_MAPS: bool = false;
override NORMAL_MAP: bool = false;
override ALPHA_MASK: bool = false;
override DOUBLE_SIDED: bool = false;
// Maps the material has: absent ones read as the white texel they would
// sample. A packed occlusion map is the metallic-roughness texture, read in
// the same sample's red.
override METALLIC_ROUGHNESS_MAP: bool = true;
override OCCLUSION_MAP: bool = true;
override EMISSIVE_MAP: bool = true;
override PACKED_OCCLUSION: bool = false;
// Transmissive materials shade as glass in the transparent pass (decision
// 038).
override TRANSMISSION: bool = false;
// Maps laid from the world's axes rather than texture coordinates, and a
// layer settling on faces turned up (decision 093).
override WORLD_MAPPING: bool = false;
override TOP_LAYER: bool = false;
// Bases the ground stains (decision 096): slot vertex stages read `grounds`.
override GROUNDING: bool = false;
// Passes drawn rarely (decision 049), reflection probe captures and impostor
// bakes, share one pipeline per pipeline state: they read the material's
// variant bits from its uniform instead of the map constants above.
override UBER_MATERIALS: bool = false;

// Whether the material has feature `bit` of its variant: `specialized` in
// pipelines specialized for it, the uniform's bit in uber ones.
fn material_flag(bit: u32, specialized: bool) -> bool {
    if UBER_MATERIALS {
        return (bitcast<u32>(material_maps.volume.w) & bit) != 0u;
    }
    return specialized;
}

// What glass refracts (decision 038): the frame's opaque color while
// transparent surfaces draw with refraction, a white texel otherwise.
@group(0) @binding(8) var refraction_source: texture_2d<f32>;
@group(0) @binding(9) var refraction_sampler: sampler;
// Environment maps (decision 043): cubes of the light around, blurred down
// their levels by roughness; the first holds the sky.
@group(0) @binding(10) var environment_maps: texture_cube_array<f32>;
@group(0) @binding(11) var environment_sampler: sampler;
// Reflection probes (decision 043) whose environment maps surfaces reflect,
// found through a grid of cells over the ground plane listing the probes
// whose boxes reach each cell. In vec4 words: [0] the grid's first corner
// (world x and z) and cell size (bits of f32) and the probe count; [1] the
// cells across x and z and where the cells and their lists start; then a
// word per cell (the first entry and the length of its list); then the
// lists, smallest boxes first, two words an entry: the probe's center and
// blend distance (bits of f32), and its extent (bits of f32) and cube in the
// environment maps.
@group(0) @binding(12) var<storage, read> probe_words: array<vec4<u32>>;

// Stable engine-owned slots; three aligned rows give an exact 48-byte affine pose.
struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
// A slot's linear color and opacity as four 16-bit fractions (red and green,
// then blue and opacity; decision 026), its packed surface (decision 019)
// and flags: one more than the offset of a skinned slot's joint palette
// (decision 036), 0 for rigid slots.
struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
// Temporally antialiased frames: each slot's pose in the previous frame.
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
// Joint palettes (decision 036): each joint's displayed world matrix times
// its inverse bind matrix; temporally antialiased frames also bind the
// previous frame's.
@group(2) @binding(3) var<storage, read> palette: array<AffinePose>;
@group(2) @binding(4) var<storage, read> previous_palette: array<AffinePose>;
// What lies under placed objects (decision 096; `gpu_data::GroundData`): x
// and z of the ground's normal as two signed fractions, the meters each
// one's base sinks, the meters up its stain reaches (0 for none, as for
// every retained slot) and the stain's color as four bytes.
@group(2) @binding(12) var<storage, read> grounds: array<vec4<u32>>;
// Temporal visible IDs mark moving slots; others kept their previous pose.
const MOVED_SLOT: u32 = 0x80000000u;

// Virtual geometry (decision 044): the arena `virtual_geometry.wgsl` selects
// clusters from. Each mesh's bases (the last where the pool's page slots
// start) and quantization; three words per cluster (the second ends with
// its mesh, the third holds its vertex and triangle words in its page and
// ends with its global page); the page pool: each page's slot, then the
// slots of VIRTUAL_PAGE_WORDS words.
struct VirtualMeshRecord {
    bases: vec4<u32>,
    info: vec4<u32>,
    position_min: vec4<f32>,
    position_step: vec4<f32>,
    uv: vec4<f32>,
}
@group(2) @binding(5) var<storage, read> virtual_meshes: array<VirtualMeshRecord>;
@group(2) @binding(6) var<storage, read> virtual_clusters: array<vec4<u32>>;
@group(2) @binding(7) var<storage, read> virtual_pages: array<u32>;
const VIRTUAL_PAGE_WORDS: u32 = 32768u;

// The visibility buffer (decision 045): each pixel of virtual geometry
// holds its selected cluster's entry in `visible_clusters` plus one (0: no
// cluster) and its triangle | bin << 7; each material bin's resolve then
// shades the pixels of its 8x8 tiles. A draw's uniform: the bin's first
// entry this phase, the bin, where its tile list starts and the tiles
// across the viewport.
struct VisibilityDraw {
    visible_base: u32,
    bin: u32,
    tile_base: u32,
    tiles_across: u32,
}
@group(2) @binding(8) var<uniform> visibility_draw: VisibilityDraw;
// Visible ID and cluster of each selected cluster, by bin.
@group(2) @binding(9) var<storage, read> visible_clusters: array<vec2<u32>>;
@group(2) @binding(10) var visibility_ids: texture_2d<u32>;
@group(2) @binding(11) var<storage, read> resolve_tiles: array<u32>;

// Local lights (decision 023), binned into view clusters by lights.wgsl.
struct Clusters {
    view_projection: mat4x4<f32>,
    // Depth along the view of point p: dot(xyz, p) + w.
    depth: vec4<f32>,
    // x and y projection scales, clustered depth range start and end.
    projection: vec4<f32>,
    // Tile width and height in pixels; slice = log2(depth) * z + w.
    grid: vec4<f32>,
    // Tiles across and down, depth slices, and the per-cluster budget.
    counts: vec4<u32>,
    // Lights, an unused word, and the listed entries' capacity.
    lights: vec4<u32>,
}
struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    cos_outer: f32,
    direction: vec3<f32>,
    cos_inner: f32,
}
@group(3) @binding(0) var<uniform> clusters: Clusters;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
// Per cluster: the first entry of its list in `cluster_lights` and its
// length.
@group(3) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read> cluster_lights: array<u32>;

// Mesh vertices, which carry octahedral normals (`octahedral_normal`) in 24
// bytes, and the four-byte slot ID stream, the only one visibility and
// batching reorder.
struct AffineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) slot: u32,
    @location(7) uv: vec2<f32>,
}

// Four output slots: vertex outputs bound geometry throughput, so texture
// coordinates ride in the w lanes and the material stays packed and flat.
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // World normal and texture u.
    @location(0) normal_u: vec4<f32>,
    // World position and texture v.
    @location(1) world_v: vec4<f32>,
    // Linear color and opacity as four 16-bit fractions, and the packed
    // surface (decision 019).
    @location(2) @interpolate(flat) material: vec3<u32>,
}

// `VertexOutput` and the unjittered previous clip position: x, y and w.
struct TemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec3<u32>,
    @location(3) previous: vec3<f32>,
}

// Frames with screen effects (decision 021): graded linear color with the
// ambient share of its light in alpha, the pixel's glow weight and
// reflectivity (decision 025), and the surface reflections follow: its
// shading normal, octahedral in [0, 1], and roughness.
struct SceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
    @location(2) surface: vec4<f32>,
}

// The scene targets, and the offset from this pixel to the same surface
// point in the previous frame, in texture coordinates.
struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: vec2<f32>,
    @location(3) surface: vec4<f32>,
}
