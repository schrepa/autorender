struct OpenSky {
    held: bool,
    seen: f32,
    beyond: f32,
}

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    return 1.0;
}

fn open_sky(world: vec3<f32>) -> OpenSky {
    var open: OpenSky;
    return open;
}
