struct KeptTexels {
    first: u32,
    count: u32,
    texels: vec4<f32>,
}

fn air_shadow(world: vec3<f32>) -> f32 {
    return 1.0;
}

fn kept_texels(world: vec3<f32>) -> KeptTexels {
    var kept: KeptTexels;
    return kept;
}
