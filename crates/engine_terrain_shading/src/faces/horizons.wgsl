import package::face::sun::sun_direction;
import package::generated::pieces::pieces;
import package::terrain::bounce::{bounce_at, bounce_held, bounce_seen};
import package::terrain::horizons::{HORIZON_RECORDS, HORIZON_RECORD_VECTORS, TERRAIN_HORIZONS,
    horizon_crest, horizon_records, horizon_turn};

struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

struct OpenSky {
    held: bool,
    seen: f32,
    beyond: f32,
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
        let shape = pieces.horizons.records[at + 1u];
        let edge = max(crest.distance * shape.y, 0.5);
        visible = min(visible, smoothstep(-edge, edge, crest.elevation + shape.z - clearance));
        if visible <= 0.0 {
            break;
        }
    }
    return visible;
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

fn open_sky(world: vec3<f32>) -> OpenSky {
    var open: OpenSky;
    let at = bounce_at(world, false);
    if at.texel.x < 0.0 {
        return open;
    }
    open.held = true;
    open.seen = bounce_seen(at);
    open.beyond = saturate(at.layer - 1.0);
    return open;
}
