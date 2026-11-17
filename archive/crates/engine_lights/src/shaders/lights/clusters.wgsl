import package::generated::lights::{Clusters, LIT_VIEWS};

@group(0) @binding(0) var<uniform> clusters: array<Clusters, LIT_VIEWS>;
@group(0) @binding(0) var<storage, read> light_data: array<vec4<u32>>;
