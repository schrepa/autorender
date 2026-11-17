import package::engine::view::{camera, stable_normalize};
import package::face::shadows::kept_texels;
import package::face::sky_visibility::sky_visibility;
import package::face::sun::sun_direction;
import package::generated::pieces::scatter::piece;
import package::generated::scatter::CANOPY_TOP_OVER_MEAN;
import package::place::far_cover::{FarGround, GroundAt};
import package::scatter::canopy::{CanopyHere, canopy_beside, canopy_color, canopy_gap,
    canopy_handed, canopy_here, canopy_lod, canopy_stand, canopy_surface};

const NORTH: vec2<f32> = vec2<f32>(0.0, 1.0);
const WEST: vec2<f32> = vec2<f32>(-0.8660254, -0.5);
const EAST: vec2<f32> = vec2<f32>(0.8660254, -0.5);

fn sky_seen(world: vec3<f32>) -> f32 {
    if camera.view.x != 0u {
        return 1.0;
    }
    let clip = camera.view_projection * vec4<f32>(world, 1.0);
    if !(clip.w > 0.0) {
        return 1.0;
    }
    let pixel = vec2<f32>(clip.x / clip.w + 1.0, 1.0 - clip.y / clip.w) / camera.jitter.zw;
    if any(pixel < vec2<f32>(0.0)) || any(pixel >= camera.region.xy) {
        return 1.0;
    }
    return sky_visibility(pixel, world + (world - camera.position.xyz)); // optics: the ground is the surface the picture was made for, so it asks from behind itself
}

fn crown_rise(here: CanopyHere, lod: f32) -> vec2<f32> {
    let reach = 0.5 * here.cell * exp2(lod);
    let apart = reach * here.scale;
    let rise = NORTH * canopy_surface(here, here.uv + NORTH * apart, lod)
        + WEST * canopy_surface(here, here.uv + WEST * apart, lod)
        + EAST * canopy_surface(here, here.uv + EAST * apart, lod);
    return rise * (2.0 / (3.0 * reach));
}

fn sun_through(here: CanopyHere, at: GroundAt) -> f32 {
    if here.span <= 0.0 {
        return 1.0;
    }
    let look = piece();
    let kept = kept_texels(at.world);
    let held = kept.count > 0u;
    let least = look.handover.y * kept.texels.x;
    if held && here.span >= 2.0 * least {
        return 1.0;
    }
    let sun = sun_direction();
    let low = max(length(sun.xz), 1.0e-6);
    let heading = sun.xz / low;
    let climb = sun.y / low + dot(at.normal.xz, heading) / max(at.normal.y, 1.0e-3);
    if climb <= 0.0 {
        return 1.0;
    }
    let reach = 0.5 * here.tallest / climb;
    let toward = canopy_stand(here, here.uv + heading * reach * here.scale,
        canopy_lod(here, max(max(reach, here.span), at.across)));
    if toward.cover <= 0.0 {
        return 1.0;
    }
    var missed = 1.0;
    if held {
        missed = 1.0 - saturate(0.5 + toward.size - log2(least));
    }
    if missed <= 0.0 {
        return 1.0;
    }
    let above = 1.0 - canopy_handed(toward.size, at.across) * toward.cover / CANOPY_TOP_OVER_MEAN;
    var crowns = toward;
    crowns.cover = toward.cover * (1.0 - look.light.y);
    let passing = pow(1.0 - crowns.cover, above) * canopy_beside(crowns, 1.0 / climb, above);
    return mix(1.0, passing, missed);
}

fn far_cover(ground: FarGround, at: GroundAt) -> FarGround {
    let here = canopy_here(at.world);
    if !here.found {
        return ground;
    }
    var result = ground;
    result.sun = ground.sun * sun_through(here, at);
    let lod = canopy_lod(here, at.footprint);
    let wide = max(lod, canopy_lod(here, here.span));
    let around = canopy_stand(here, here.uv, wide);
    if around.cover <= 0.0 {
        return result;
    }
    let look = piece();
    let open = mix(look.light.x, 1.0, canopy_gap(around, 1.0));
    result.occlusion = ground.occlusion * min(open / max(sky_seen(at.world), open), 1.0);
    let handed = canopy_handed(around.size, at.across);
    if handed <= 0.0 {
        return result;
    }
    let own = canopy_stand(here, here.uv, lod);
    let cosine = max(at.toward.y, 1.0e-3);
    let hidden = 1.0 - canopy_beside(around, sqrt(max(1.0 - cosine * cosine, 0.0)) / cosine, 1.0);
    let beside = (1.0 - own.cover) * hidden;
    let covered = own.cover + beside;
    let shown = handed * covered;
    if shown <= 0.0 {
        return result;
    }
    let far = beside / covered;
    let rise = crown_rise(here, lod) * look.handover.w;
    let level = at.normal.xz / max(at.normal.y, 1.0e-3);
    let crown = stable_normalize(vec3<f32>(level.x - rise.x, 1.0, level.y - rise.y));
    result.color = mix(ground.color, canopy_color(here, mix(lod, wide, far), ground.color), shown);
    result.roughness = mix(ground.roughness, look.light.w, shown);
    result.normal = stable_normalize(mix(ground.normal, crown, shown));
    result.occlusion = mix(result.occlusion, look.light.z, shown);
    result.lift = shown * mix(own.height, around.height, far);
    result.shown = shown;
    return result;
}
