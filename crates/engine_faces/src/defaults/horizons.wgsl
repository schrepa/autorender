struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

struct BounceAt {
    first: u32,
    texel: vec2<f32>,
    next: vec2<f32>,
    blend: f32,
    layer: f32,
}

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    return 1.0;
}

fn bounce_at(world: vec3<f32>, on_terrain: bool) -> BounceAt {
    var found: BounceAt;
    found.texel = vec2<f32>(-1.0);
    return found;
}

fn bounce_light(world: vec3<f32>, normal: vec3<f32>, on_terrain: bool) -> BounceLight {
    var found: BounceLight;
    found.sky = 1.0;
    return found;
}

fn bounce_seen(at: BounceAt) -> f32 {
    return 1.0;
}

struct OpenSky {
    held: bool,
    seen: f32,
    beyond: f32,
}

fn open_sky(world: vec3<f32>) -> OpenSky {
    var open: OpenSky;
    return open;
}
