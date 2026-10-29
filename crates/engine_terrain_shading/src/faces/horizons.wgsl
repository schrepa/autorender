import package::face::sun::sun_direction;
import package::frame::view::{detail_distance, detail_level};
import package::terrain::bounce::{BOUNCE, BOUNCE_HIGH, BOUNCE_LOW, bounce_held, bounce_texel};
import package::terrain::horizons::{HORIZON_LAYERS, HORIZON_RECORDS, HORIZON_RECORD_VECTORS,
    TERRAIN_HORIZONS, horizon_crest, horizon_records, horizon_turn, terrain_horizon_records,
    terrain_horizon_sampler, terrain_horizons};

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
    if !TERRAIN_HORIZONS {
        return 1.0;
    }
    let sun = sun_direction();
    let level = length(sun.xz);
    if level < 1.0e-4 {
        return 1.0;
    }
    let rise = sun.y / level;
    let turn = horizon_turn(sun.xz);
    let count = min(horizon_records(), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
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

fn bounce_at(world: vec3<f32>, on_terrain: bool) -> BounceAt {
    var found: BounceAt;
    found.texel = vec2<f32>(-1.0);
    if !BOUNCE || !TERRAIN_HORIZONS {
        return found;
    }
    let count = min(horizon_records(), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
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
