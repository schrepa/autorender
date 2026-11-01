import package::place::bounce::BounceLight;
import package::terrain_shading::bounce::{bounce_at, bounce_held};

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
