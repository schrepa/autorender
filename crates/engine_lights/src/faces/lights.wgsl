import package::lights::clusters::{KEPT_REGIONS, LOCAL_LIGHTS, SHAPED_LIGHTS, clusters, light_data};
import package::place::lights::{Light, LightCluster};

const LIGHT_INTERIOR: u32 = 1u;
const LIGHT_EXTERIOR: u32 = 2u;

fn light_cluster(world: vec3<f32>, pixel: vec2<f32>) -> LightCluster {
    var found: LightCluster;
    found.depth = dot(clusters.depth.xyz, world) + clusters.depth.w;
    found.end = clusters.projection.w;
    if found.depth >= found.end {
        return found;
    }
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(found.depth, 1.0e-6)) * clusters.grid.z
        + clusters.grid.w), 0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let ranges = light_data[clusters.offsets.y + cluster / 2u];
    found.range = select(ranges.xy, ranges.zw, (cluster & 1u) != 0u);
    return found;
}

fn cluster_light(cluster: LightCluster, n: u32) -> Light {
    let listed = cluster.range.x + n;
    let row = clusters.offsets.x + light_data[clusters.offsets.z + listed / 4u][listed % 4u] * 4u;
    let first = light_data[row];
    let second = light_data[row + 1u];
    let third = light_data[row + 2u];
    let fourth = light_data[row + 3u];
    var light: Light;
    light.position = bitcast<vec3<f32>>(first.xyz);
    light.range = bitcast<f32>(first.w);
    light.color = bitcast<vec3<f32>>(second.xyz);
    light.cos_outer = bitcast<f32>(second.w);
    light.direction = bitcast<vec3<f32>>(third.xyz);
    light.cos_inner = bitcast<f32>(third.w);
    light.radius = bitcast<f32>(fourth.x);
    light.length = bitcast<f32>(fourth.y);
    light.region = fourth.z;
    light.pad = fourth.w;
    return light;
}

fn shaped_lights() -> bool {
    return (clusters.lights.y & SHAPED_LIGHTS) != 0u;
}

fn kept_regions() -> bool {
    return (clusters.lights.y & KEPT_REGIONS) != 0u;
}

fn lights_on() -> bool {
    return LOCAL_LIGHTS;
}
