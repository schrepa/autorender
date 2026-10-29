import package::frame::view::{PI, lighting};
import package::shadows::sample::shadow;

@group(0) @binding(14) var terrain_horizons: texture_2d_array<f32>;
const HORIZON_RECORDS: u32 = 384u;
@group(0) @binding(15) var<uniform> terrain_horizon_records: array<vec4<f32>, HORIZON_RECORDS>;
@group(0) @binding(16) var terrain_horizon_sampler: sampler;

override TERRAIN_HORIZONS: bool = true;

const HORIZON_DIRECTIONS: u32 = 16u;
const HORIZON_LAYERS: u32 = 9u;
const HORIZON_RECORD_VECTORS: u32 = 6u;

struct HorizonTurn {
    first: u32,
    second: u32,
    between: f32,
}

fn horizon_turn(heading: vec2<f32>) -> HorizonTurn {
    let turn = fract(atan2(heading.y, heading.x) / (2.0 * PI) + 1.0) * f32(HORIZON_DIRECTIONS);
    var found: HorizonTurn;
    found.first = min(u32(turn), HORIZON_DIRECTIONS - 1u);
    found.second = (found.first + 1u) % HORIZON_DIRECTIONS;
    found.between = turn - f32(found.first);
    return found;
}

struct HorizonCrest {
    slope: f32,
    distance: f32,
    elevation: f32,
    inside: bool,
}

fn horizon_crest(at: u32, turn: HorizonTurn, world: vec3<f32>, on_terrain: bool)
        -> HorizonCrest {
    var crest: HorizonCrest;
    let place = terrain_horizon_records[at];
    let shape = terrain_horizon_records[at + 1u];
    let uv = (world.xz - place.xy) * place.zw;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return crest;
    }
    crest.inside = true;
    let base = u32(shape.x);
    let near = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
        base + turn.first / 2u, 0.0);
    var far = near;
    if (turn.first & 1u) == 1u {
        far = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            base + turn.second / 2u, 0.0);
    }
    let odd = (turn.first & 1u) == 1u;
    let heights = vec2<f32>(select(near.r, near.g, odd), select(far.g, far.r, odd));
    let distances = vec2<f32>(select(near.b, near.a, odd), select(far.a, far.b, odd));
    let slopes = max(heights / max(distances, vec2<f32>(1.0e-3)), vec2<f32>(-1.0));
    crest.slope = mix(slopes.x, slopes.y, turn.between);
    crest.distance = mix(distances.x, distances.y, turn.between);
    if !on_terrain {
        let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            base + HORIZON_LAYERS - 1u, 0.0).r;
        crest.elevation = world.y - shape.w - ground;
    }
    return crest;
}

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    if !TERRAIN_HORIZONS {
        return 1.0;
    }
    let sun = lighting.sun.xyz;
    let level = length(sun.xz);
    if level < 1.0e-4 {
        return 1.0;
    }
    let rise = sun.y / level;
    let turn = horizon_turn(sun.xz);
    let count = min(u32(shadow.level_state.w), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    var visible = 1.0;
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let crest = horizon_crest(at, turn, world, on_terrain);
        if !crest.inside {
            continue;
        }
        let clearance = crest.distance * (crest.slope - rise);
        let shape = terrain_horizon_records[at + 1u];
        let edge = max(crest.distance * shape.y, 0.5);
        visible = min(visible, smoothstep(-edge, edge, crest.elevation + shape.z - clearance));
        if visible <= 0.0 {
            break;
        }
    }
    return visible;
}
