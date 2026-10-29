struct Clusters {
    view_projection: mat4x4<f32>,
    depth: vec4<f32>,
    projection: vec4<f32>,
    grid: vec4<f32>,
    counts: vec4<u32>,
    lights: vec4<u32>,
    offsets: vec4<u32>,
}

const SHAPED_LIGHTS: u32 = 1u;
const KEPT_REGIONS: u32 = 2u;

override LOCAL_LIGHTS: bool = false;

@group(0) @binding(0) var<uniform> clusters: Clusters;
@group(0) @binding(1) var<storage, read> light_data: array<vec4<u32>>;
