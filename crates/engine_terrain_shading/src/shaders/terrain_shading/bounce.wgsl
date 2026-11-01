import package::engine::view::{detail_distance, detail_level};
import package::generated::pieces::terrain_shading::piece;
import package::terrain_shading::horizons::{horizon_bounces, horizon_records, terrain_horizon_sampler, terrain_horizons};
import package::generated::terrain_shading::{BOUNCE_HIGH, BOUNCE_LOW, BOUNCE_WINDOW_ROW, HORIZON_LAYERS, HORIZON_RECORDS, HORIZON_RECORD_VECTORS};

@group(0) @binding(0) var terrain_bounce_probes: texture_2d_array<f32>;

struct BounceAt {
    first: u32,
    texel: vec2<f32>,
    next: vec2<f32>,
    blend: f32,
    layer: f32,
}

fn bounce_layer(texel: vec2<f32>, first: u32, normal: vec3<f32>) -> vec4<f32> {
    let mean = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, texel, first, 0.0);
    let x = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, texel, first + 1u,
        0.0);
    let y = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, texel, first + 2u,
        0.0);
    let z = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, texel, first + 3u,
        0.0);
    return mean + x * normal.x + y * normal.y + z * normal.z;
}

fn bounce_texel(at: u32, level: u32, uv: vec2<f32>) -> vec2<f32> {
    let header = piece().records[at + 2u];
    let arranged = piece().records[at + 3u];
    let columns = f32(u32(header.y) >> level);
    let window = min(header.w, columns);
    let packed = u32(piece().records[at + 4u + level / 4u][level % 4u]);
    let first = vec2<f32>(f32(packed % BOUNCE_WINDOW_ROW), f32(packed / BOUNCE_WINDOW_ROW));
    let column = clamp(uv * columns - 0.5, vec2<f32>(0.0), vec2<f32>(columns - 1.0)) - first;
    if any(column < vec2<f32>(0.0)) || any(column > vec2<f32>(window - 1.0)) {
        return vec2<f32>(-1.0);
    }
    let row = u32(arranged.z);
    let region = vec2<f32>(f32(level % row), f32(level / row)) * header.w;
    return (region + column + 0.5) * arranged.x;
}

fn bounce_held(texel: vec2<f32>, first: u32, normal: vec3<f32>, above: f32) -> vec4<f32> {
    let held = bounce_layer(texel, first, normal);
    if above <= 0.0 {
        return held;
    }
    return mix(held, bounce_layer(texel, first + 4u, normal), above);
}

fn bounce_at(world: vec3<f32>, on_terrain: bool) -> BounceAt {
    var found: BounceAt;
    found.texel = vec2<f32>(-1.0);
    if !horizon_bounces() {
        return found;
    }
    let count = min(horizon_records(), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let header = piece().records[at + 2u];
        if header.x < 0.0 {
            continue;
        }
        let place = piece().records[at];
        let uv = (world.xz - place.xy) * place.zw;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            continue;
        }
        let arranged = piece().records[at + 3u];
        let levels = u32(header.z);
        let picked = detail_level(detail_distance(world), 0.5 * arranged.y, levels);
        var level = max(picked.level, u32(arranged.w));
        var blend = select(0.0, picked.blend, level == picked.level);
        var texel = bounce_texel(at, level, uv);
        for (; texel.x < 0.0 && level + 1u < levels; level++) {
            texel = bounce_texel(at, level + 1u, uv);
            blend = 0.0;
        }
        if texel.x < 0.0 {
            continue;
        }
        if !on_terrain {
            let shape = piece().records[at + 1u];
            let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
                u32(shape.x) + HORIZON_LAYERS - 1u, 0.0).r;
            found.layer = (world.y - shape.w - ground - BOUNCE_LOW)
                / (BOUNCE_HIGH - BOUNCE_LOW);
        }
        found.first = u32(header.x);
        found.texel = texel;
        if blend > 0.0 && level + 1u < levels {
            found.next = bounce_texel(at, level + 1u, uv);
            found.blend = select(0.0, blend, found.next.x >= 0.0);
        }
        return found;
    }
    return found;
}

fn bounce_seen(at: BounceAt) -> f32 {
    var lower = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, at.texel,
        at.first, 0.0).a;
    var upper = textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler, at.texel,
        at.first + 4u, 0.0).a;
    if at.blend > 0.0 {
        lower = mix(lower, textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler,
            at.next, at.first, 0.0).a, at.blend);
        upper = mix(upper, textureSampleLevel(terrain_bounce_probes, terrain_horizon_sampler,
            at.next, at.first + 4u, 0.0).a, at.blend);
    }
    return mix(lower, upper, saturate(at.layer));
}
