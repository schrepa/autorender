import package::engine::shading::{SPECULAR_LIMIT, SPECULAR_VARIANCE};
import package::engine::view::material_detail;

const WATER_SMOOTHEST: f32 = 0.03;
const WATER_ROUGHEST: f32 = 0.6;

fn water_spread(slopes: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(slopes.x * slopes.x, slopes.y * slopes.y, slopes.x * slopes.y);
}

fn water_spread_toward(spread: vec3<f32>, way: vec2<f32>) -> f32 {
    return spread.x * way.x * way.x + spread.y * way.y * way.y + 2.0 * spread.z * way.x * way.y;
}

struct WaterFootprint {
    dx: vec2<f32>,
    dy: vec2<f32>,
    along: f32,
}

fn water_footprint(dx: vec2<f32>, dy: vec2<f32>) -> WaterFootprint {
    let detail = material_detail();
    var footprint: WaterFootprint;
    footprint.dx = dx * detail;
    footprint.dy = dy * detail;
    let squares = dot(footprint.dx, footprint.dx) + dot(footprint.dy, footprint.dy);
    let area = abs(footprint.dx.x * footprint.dy.y - footprint.dx.y * footprint.dy.x);
    let sum = sqrt(squares + 2.0 * area);
    let difference = sqrt(max(squares - 2.0 * area, 0.0));
    footprint.along = 0.5 * (sum + difference);
    return footprint;
}

fn water_turning(slopes: vec2<f32>) -> vec3<f32> {
    let turning = 2.0 * SPECULAR_VARIANCE
        * (water_spread(dpdx(slopes)) + water_spread(dpdy(slopes)));
    return turning * min(SPECULAR_LIMIT / max(turning.x + turning.y, 1.0e-12), 1.0);
}

fn water_roughness(hidden: vec3<f32>, heading: vec2<f32>) -> vec2<f32> {
    let variances = vec2<f32>(water_spread_toward(hidden, heading),
        water_spread_toward(hidden, vec2<f32>(-heading.y, heading.x)));
    return clamp(sqrt(sqrt(2.0 * max(variances, vec2<f32>(0.0)))), vec2<f32>(WATER_SMOOTHEST),
        vec2<f32>(WATER_ROUGHEST));
}
