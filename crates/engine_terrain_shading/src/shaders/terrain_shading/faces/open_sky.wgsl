import package::place::open_sky::OpenSky;
import package::terrain_shading::bounce::{bounce_at, bounce_seen};

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
