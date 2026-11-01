import package::face::sun::sun_direction;
import package::generated::pieces::terrain_shading::piece;
import package::place::horizons::OpenSky;
import package::terrain_shading::bounce::{bounce_at, bounce_seen};
import package::terrain_shading::horizons::{horizon_crest, horizon_records, horizon_turn};
import package::generated::terrain_shading::{HORIZON_RECORDS, HORIZON_RECORD_VECTORS};

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
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
        let shape = piece().records[at + 1u];
        let edge = max(crest.distance * shape.y, 0.5);
        visible = min(visible, smoothstep(-edge, edge, crest.elevation + shape.z - clearance));
        if visible <= 0.0 {
            break;
        }
    }
    return visible;
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
