// Cluster and placed-light records shared by binning, shading and depth filtering.
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
    // Lights, an occupancy phase tag (zero for full lists), list capacity,
    // and the probe-only flag (or phase number in occupancy uniforms).
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
