import package::cover::faces::cover::{cover_header, cover_light, cover_merged, cover_shape};
import package::cover::laws::{FOLD_STEPS, FOLD_WIDEST, FORM_FOLD, LIT_LEAST, blade_above,
    blade_taper};
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::face::foliage::foliage;
import package::face::lighting::SURFACE_THIN;

const FOLD_SLOPE: f32 = FOLD_WIDEST / FOLD_STEPS;
const PLANE_LEAST: f32 = 1.0e-36;
const WIDTH_LEAST: f32 = 1.0e-4;

struct BladeFold {
    side: vec3<f32>,
    crease: f32,
}

fn blade_along(input: SurfaceInput) -> f32 {
    return saturate(input.uv.x);
}

fn fold_form(input: SurfaceInput) -> u32 {
    return u32(round(input.surface.w * 255.0));
}

fn fold_slope(form: u32) -> f32 {
    return f32(form & FORM_FOLD) * FOLD_SLOPE;
}

fn blade_fold(input: SurfaceInput) -> BladeFold {
    let turn = vec2<f32>(dpdx(input.uv.y), dpdy(input.uv.y));
    let plane = cross(input.world_dx, input.world_dy);
    let facet = plane * inverseSqrt(max(dot(plane, plane), PLANE_LEAST));
    let spread = cross(input.world_dy, facet) * turn.x + cross(facet, input.world_dx) * turn.y;
    let reach = dot(spread, spread);
    var fold: BladeFold;
    if reach > 0.0 {
        fold.side = spread * inverseSqrt(reach);
        fold.crease = clamp(input.uv.y / (abs(turn.x) + abs(turn.y)), -1.0, 1.0);
    }
    return fold;
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return input.color.rgb;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return 1.0;
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let fold = blade_fold(input);
    let toward = normalize(normalize(input.normal)
        + fold.side * (fold_slope(fold_form(input)) * fold.crease));
    let face = select(-toward, toward, input.front);
    let header = cover_header();
    let near = 1.0 - cover_merged(cover_light(), distance(input.world, input.eye));
    let bent = normalize(mix(face, vec3<f32>(0.0, 1.0, 0.0),
        mix(header.shading.y, header.layers.z, near)));
    let height = input.color.a;
    if foliage().w <= 0.0 || height < 0.0 {
        return bent;
    }
    let tilt = input.uv1;
    let ground = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
    let above = blade_along(input) * height;
    return normalize(mix(bent, ground, 1.0 - smoothstep(0.0, foliage().w, above)));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let near = 1.0 - cover_merged(cover_light(), distance(input.world, input.eye));
    let shading = cover_header().shading;
    return shading.w * mix(1.0, shading.z, near * blade_along(input));
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let shown = saturate(1.0 - abs(dpdx(input.uv.y)) - abs(dpdy(input.uv.y)));
    let shape = cover_shape();
    let rooted = pow(max(input.surface.x, LIT_LEAST), blade_above(shape.arc.w, blade_along(input)));
    let form = fold_form(input);
    let slope = fold_slope(form);
    if input.front || shown <= 0.0 || slope <= 0.0 {
        return rooted;
    }
    let breadth = blade_taper(shape.arc.w, shape.tips.xy, form, blade_along(input));
    let outward = saturate(abs(input.uv.y) / max(breadth, WIDTH_LEAST));
    let steep = slope * slope;
    let closing = (steep - 1.0) / (steep + 1.0);
    let hidden = 0.5 - 0.5 * (outward - closing)
        / sqrt(outward * outward + 1.0 - 2.0 * outward * closing);
    return rooted * (1.0 - hidden * shown);
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let glow = cover_light().glow;
    return min(foliage().x * mix(glow.x, glow.y, blade_along(input)), 1.0);
}

fn mirrored(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return 0.0;
}

fn kind(input: SurfaceInput, point: SurfacePoint) -> u32 {
    return SURFACE_THIN;
}

fn tangled(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return cover_merged(cover_light(), distance(input.world, input.eye));
}
