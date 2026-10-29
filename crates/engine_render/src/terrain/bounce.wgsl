import package::frame::view::{detail_distance, detail_level};
import package::shadows::sample::shadow;
import package::terrain::horizons::{HORIZON_LAYERS, HORIZON_RECORDS, HORIZON_RECORD_VECTORS,
    TERRAIN_HORIZONS, terrain_horizon_records, terrain_horizon_sampler, terrain_horizons};

override BOUNCE: bool = true;

const BOUNCE_LOW: f32 = 1.5;
const BOUNCE_HIGH: f32 = 24.0;
const BOUNCE_WINDOW_ROW: u32 = 4096u;

struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

fn bounce_layer(texel: vec2<f32>, first: u32, normal: vec3<f32>) -> vec4<f32> {
    let mean = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first, 0.0);
    let x = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 1u,
        0.0);
    let y = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 2u,
        0.0);
    let z = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 3u,
        0.0);
    return mean + x * normal.x + y * normal.y + z * normal.z;
}

fn bounce_texel(at: u32, level: u32, uv: vec2<f32>) -> vec2<f32> {
    let header = terrain_horizon_records[at + 2u];
    let arranged = terrain_horizon_records[at + 3u];
    let columns = f32(u32(header.y) >> level);
    let window = min(header.w, columns);
    let packed = u32(terrain_horizon_records[at + 4u + level / 4u][level % 4u]);
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

struct BounceAt {
    first: u32,
    texel: vec2<f32>,
    next: vec2<f32>,
    blend: f32,
    layer: f32,
}

fn bounce_at(world: vec3<f32>, on_terrain: bool) -> BounceAt {
    var found: BounceAt;
    found.texel = vec2<f32>(-1.0);
    if !BOUNCE || !TERRAIN_HORIZONS {
        return found;
    }
    let count = min(u32(shadow.level_state.w), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let header = terrain_horizon_records[at + 2u];
        if header.x < 0.0 {
            continue;
        }
        let place = terrain_horizon_records[at];
        let uv = (world.xz - place.xy) * place.zw;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            continue;
        }
        let arranged = terrain_horizon_records[at + 3u];
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
            let shape = terrain_horizon_records[at + 1u];
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

fn bounce_light(world: vec3<f32>, normal: vec3<f32>, on_terrain: bool) -> BounceLight {
    var found: BounceLight;
    found.sky = 1.0;
    let at = bounce_at(world, on_terrain);
    if at.texel.x < 0.0 {
        return found;
    }
    let above = saturate(at.layer);
    var held = bounce_held(at.texel, at.first, normal, above);
    if at.blend > 0.0 {
        held = mix(held, bounce_held(at.next, at.first, normal, above), at.blend);
    }
    found.light = max(held.rgb, vec3<f32>(0.0));
    found.sky = saturate(held.a);
    found.weight = 1.0;
    return found;
}

fn bounce_seen(at: BounceAt) -> f32 {
    var lower = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, at.texel,
        at.first, 0.0).a;
    var upper = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, at.texel,
        at.first + 4u, 0.0).a;
    if at.blend > 0.0 {
        lower = mix(lower, textureSampleLevel(terrain_horizons, terrain_horizon_sampler,
            at.next, at.first, 0.0).a, at.blend);
        upper = mix(upper, textureSampleLevel(terrain_horizons, terrain_horizon_sampler,
            at.next, at.first + 4u, 0.0).a, at.blend);
    }
    return mix(lower, upper, saturate(at.layer));
}
