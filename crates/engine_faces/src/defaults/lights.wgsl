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

struct LightCluster {
    range: vec2<u32>,
    depth: f32,
    end: f32,
}

const LIGHT_INTERIOR: u32 = 1u;
const LIGHT_EXTERIOR: u32 = 2u;

fn light_cluster(world: vec3<f32>, pixel: vec2<f32>) -> LightCluster {
    var cluster: LightCluster;
    return cluster;
}

fn cluster_light(cluster: LightCluster, n: u32) -> Light {
    var light: Light;
    return light;
}

fn shaped_lights() -> bool {
    return false;
}

fn kept_regions() -> bool {
    return false;
}

fn lights_on() -> bool {
    return false;
}
