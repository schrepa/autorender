import package::engine::view::camera;
import package::generated::lights::{KEPT_REGIONS, LIT_VIEWS, REGION_EXTERIOR, REGION_INTERIOR,
    SHAPED_LIGHTS};
import package::lights::clusters::{clusters, light_data};
import package::place::lights::{Light, LightCluster};

const LIGHT_INTERIOR: u32 = REGION_INTERIOR;
const LIGHT_EXTERIOR: u32 = REGION_EXTERIOR;

fn light_cluster(world: vec3<f32>, pixel: vec2<f32>) -> LightCluster {
    var found: LightCluster;
    let view = camera.view.x;
    if view >= LIT_VIEWS {
        return found;
    }
    found.depth = dot(clusters[view].depth.xyz, world) + clusters[view].depth.w;
    found.end = clusters[view].projection.w;
    if found.depth >= found.end {
        return found;
    }
    let counts = clusters[view].counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters[view].grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(found.depth, 1.0e-6)) * clusters[view].grid.z
        + clusters[view].grid.w), 0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let ranges = light_data[clusters[view].offsets.y + cluster / 2u];
    found.range = select(ranges.xy, ranges.zw, (cluster & 1u) != 0u);
    return found;
}

fn cluster_light(cluster: LightCluster, n: u32) -> Light {
    let view = camera.view.x;
    let listed = cluster.range.x + n;
    let row = clusters[view].offsets.x
        + light_data[clusters[view].offsets.z + listed / 4u][listed % 4u] * 4u;
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
    return (clusters[0u].lights.y & SHAPED_LIGHTS) != 0u;
}

fn kept_regions() -> bool {
    return (clusters[0u].lights.y & KEPT_REGIONS) != 0u;
}

fn lights_on() -> bool {
    return true;
}
