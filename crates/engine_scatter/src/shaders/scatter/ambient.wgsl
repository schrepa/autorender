import package::engine::view::{detail_distance, detail_level};
import package::generated::pieces::scatter::piece;
import package::place::ambient::{EnvironmentAt, EnvironmentLight};

@group(0) @binding(0) var occupancy_coverage: texture_2d_array<f32>;
@group(0) @binding(0) var occupancy_heights: texture_2d_array<f32>;
@group(0) @binding(0) var occupancy_sampler: sampler;

fn occupancy_cover(world: vec3<f32>, map: u32) -> f32 {
    let canopy = piece();
    let inverse = canopy.occupancy.x / f32(1u << map);
    let centers = canopy.occupancy_centers[map / 2u];
    let center = select(centers.xy, centers.zw, (map & 1u) == 1u);
    let local = (world.xz - center) * inverse + 0.5;
    if any(local <= vec2<f32>(0.0)) || any(local >= vec2<f32>(1.0)) {
        return -1.0;
    }
    let size = canopy.occupancy.y;
    let uv = fract(world.xz * inverse);
    let at = uv * size - vec2<f32>(0.5);
    let corner = floor(at);
    var coverage: f32;
    if all(corner >= vec2<f32>(0.0)) && all(corner < vec2<f32>(size - 1.0)) {
        coverage = textureSampleLevel(occupancy_coverage, occupancy_sampler, uv, map, 0.0).r;
    } else {
        let texels = vec2<i32>(i32(size));
        let low = (vec2<i32>(corner) + texels) % texels;
        let high = (vec2<i32>(corner) + vec2<i32>(1) + texels) % texels;
        let share = at - corner;
        let below = mix(textureLoad(occupancy_coverage, low, map, 0).r,
            textureLoad(occupancy_coverage, vec2<i32>(high.x, low.y), map, 0).r, share.x);
        let above = mix(textureLoad(occupancy_coverage, vec2<i32>(low.x, high.y), map, 0).r,
            textureLoad(occupancy_coverage, high, map, 0).r, share.x);
        coverage = mix(below, above, share.y);
    }
    coverage = min(coverage, 1.0);
    if coverage <= 0.0 {
        return 0.0;
    }
    let texel = min(vec2<u32>(uv * size), vec2<u32>(u32(size) - 1u));
    let heights = textureLoad(occupancy_heights, texel, map, 0).rg;
    let height = world.y - canopy.occupancy_maps.y;
    let top = heights.x;
    let bottom = -heights.y;
    if height >= top {
        return 0.0;
    }
    return coverage * saturate((top - height) / max(top - bottom, 0.5));
}

fn occupancy_visibility(world: vec3<f32>) -> f32 {
    let canopy = piece();
    let strength = canopy.occupancy.w;
    if strength <= 0.0 {
        return 1.0;
    }
    let maps = u32(canopy.occupancy_maps.x);
    let picked = detail_level(detail_distance(world), 0.5 * canopy.occupancy.z,
        maps + 1u);
    var map = picked.level;
    var blend = picked.blend;
    var hidden = -1.0;
    for (; map < maps; map++) {
        hidden = occupancy_cover(world, map);
        if hidden >= 0.0 {
            break;
        }
        blend = 0.0;
    }
    if hidden < 0.0 {
        return 1.0;
    }
    if blend > 0.0 {
        var next = 0.0;
        if map + 1u < maps {
            next = occupancy_cover(world, map + 1u);
        }
        if next >= 0.0 {
            hidden = mix(hidden, next, blend);
        }
    }
    return 1.0 - strength * hidden;
}

fn environment(light: EnvironmentLight, at: EnvironmentAt) -> EnvironmentLight {
    var result = light;
    result.open = light.open * occupancy_visibility(at.world);
    return result;
}
