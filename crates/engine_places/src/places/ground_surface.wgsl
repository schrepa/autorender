struct GroundMaterial {
    first: u32,
    second: u32,
    share: u32,
    wet: u32,
}

struct SurfaceGround {
    x: f32,
    z: f32,
    height: f32,
    gradient: vec2<f32>,
    curvature: f32,
    spacing: f32,
}

struct SurfaceInputs {
    x: f32,
    z: f32,
    height: f32,
    gradient: vec2<f32>,
    curvature: f32,
    spacing: f32,
    flow: f32,
    wetness: f32,
    lake: f32,
    erosion: f32,
    river: f32,
    grain: f32,
    power: f32,
    flood: f32,
    path_distance: f32,
    path_half: f32,
}

fn material_at(terrain: u32, at: vec2<f32>, height: f32, slope: f32) -> GroundMaterial {
    var material: GroundMaterial;
    return material;
}

fn surface_level(terrain: u32, spacing: f32) -> u32 {
    return 0u;
}

fn surface_inputs(terrain: u32, ground: SurfaceGround, level: u32, river: vec4<f32>)
        -> SurfaceInputs {
    var inputs: SurfaceInputs;
    return inputs;
}

fn surface_control(terrain: u32, inputs: SurfaceInputs) -> u32 {
    return 0u;
}
