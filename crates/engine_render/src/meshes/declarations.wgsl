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
    transforms: array<vec4<f32>, 10>,
    emissive: vec4<f32>,
    params: vec4<f32>,
    volume: vec4<f32>,
    top: vec4<f32>,
    layer: vec4<f32>,
}
@group(1) @binding(10) var<uniform> material_maps: MaterialMaps;

override MATERIAL_MAPS: bool = false;
override NORMAL_MAP: bool = false;
override ALPHA_MASK: bool = false;
override DOUBLE_SIDED: bool = false;
override METALLIC_ROUGHNESS_MAP: bool = true;
override OCCLUSION_MAP: bool = true;
override EMISSIVE_MAP: bool = true;
override PACKED_OCCLUSION: bool = false;
override TRANSMISSION: bool = false;
override WORLD_MAPPING: bool = false;
override TOP_LAYER: bool = false;
override GROUNDING: bool = false;
override FADE_LIST: bool = false;
override UBER_MATERIALS: bool = false;
override FOLIAGE: bool = false;
override PRETESTED: bool = false;

const VARIANT_MAPS: u32 = 1u;
const VARIANT_NORMAL_MAP: u32 = 2u;
const VARIANT_DOUBLE_SIDED: u32 = 8u;
const VARIANT_METALLIC_ROUGHNESS_MAP: u32 = 128u;
const VARIANT_OCCLUSION_MAP: u32 = 256u;
const VARIANT_EMISSIVE_MAP: u32 = 512u;
const VARIANT_PACKED_OCCLUSION: u32 = 1024u;
const VARIANT_WORLD_MAPPING: u32 = 4096u;
const VARIANT_TOP_LAYER: u32 = 8192u;
const VARIANT_FOLIAGE: u32 = 65536u;

fn material_flag(bit: u32, specialized: bool) -> bool {
    if UBER_MATERIALS {
        return (bitcast<u32>(material_maps.volume.w) & bit) != 0u;
    }
    return specialized;
}

@group(0) @binding(9) var refraction_source: texture_2d<f32>;
@group(0) @binding(10) var refraction_sampler: sampler;
const PROBE_ENTRY_WORDS: u32 = 3u;
@group(0) @binding(13) var<storage, read> probe_words: array<vec4<u32>>;

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}
struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}
@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
@group(2) @binding(2) var<storage, read> previous_poses: array<AffinePose>;
@group(2) @binding(3) var<storage, read> palette: array<AffinePose>;
@group(2) @binding(4) var<storage, read> previous_palette: array<AffinePose>;
@group(2) @binding(5) var<storage, read> grounds: array<vec4<u32>>;
const MOVED_SLOT: u32 = 0x80000000u;

const FADE_SLOT: u32 = 0x00ffffffu;
const FADE_WORD_ON: u32 = 0x100u;
const FADE_WORD_OUT: u32 = 0x80u;

fn slot_index(entry: u32) -> u32 {
    if FADE_LIST {
        return entry & FADE_SLOT;
    }
    return entry & ~MOVED_SLOT;
}

fn slot_moved(entry: u32) -> bool {
    return !FADE_LIST && (entry & MOVED_SLOT) != 0u;
}

fn slot_fade(entry: u32) -> u32 {
    return select(0u, (entry >> 24u) | FADE_WORD_ON, FADE_LIST);
}

fn fade_hidden(fade: u32, position: vec2<f32>) -> bool {
    if (fade & FADE_WORD_ON) == 0u {
        return false;
    }
    let shows = fade_noise(position) < f32(fade & 0x7fu) / 127.0;
    return shows == ((fade & FADE_WORD_OUT) != 0u);
}

struct VirtualMeshRecord {
    bases: vec4<u32>,
    info: vec4<u32>,
    position_min: vec4<f32>,
    position_step: vec4<f32>,
    uv: vec4<f32>,
}
@group(2) @binding(6) var<storage, read> virtual_meshes: array<VirtualMeshRecord>;
@group(2) @binding(7) var<storage, read> virtual_clusters: array<vec4<u32>>;
@group(2) @binding(8) var<storage, read> virtual_pages: array<u32>;
const VIRTUAL_PAGE_WORDS: u32 = 32768u;

struct VisibilityDraw {
    visible_base: u32,
    bin: u32,
    tile_base: u32,
    tiles_across: u32,
}
@group(2) @binding(9) var<uniform> visibility_draw: VisibilityDraw;
@group(2) @binding(10) var<storage, read> visible_clusters: array<vec2<u32>>;
@group(2) @binding(11) var visibility_ids: texture_2d<u32>;
@group(2) @binding(12) var<storage, read> resolve_tiles: array<u32>;

struct Clusters {
    view_projection: mat4x4<f32>,
    depth: vec4<f32>,
    projection: vec4<f32>,
    grid: vec4<f32>,
    counts: vec4<u32>,
    lights: vec4<u32>,
}
struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    cos_outer: f32,
    direction: vec3<f32>,
    cos_inner: f32,
    radius: f32,
    length: f32,
    region: u32,
    pad: u32,
}
const SHAPED_LIGHTS: u32 = 1u;
const KEPT_REGIONS: u32 = 2u;
const LIGHT_INTERIOR: u32 = 1u;
const LIGHT_EXTERIOR: u32 = 2u;
@group(3) @binding(0) var<uniform> clusters: Clusters;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
@group(3) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read> cluster_lights: array<u32>;

struct AffineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) slot: u32,
    @location(7) uv: vec2<f32>,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}

struct TemporalOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
}

struct SceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
    @location(2) surface: vec4<u32>,
}

struct TemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: vec2<f32>,
    @location(3) surface: vec4<u32>,
}
