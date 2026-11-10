import package::cover::faces::cover::{COVER_FAR_SIDE, COVER_ROUGHNESS};
import package::cover::laws::COVER_ROOT_OCCLUSION;
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::face::foliage::foliage;
import package::face::lighting::SURFACE_THIN;
import package::face::sun::sun_direction;

struct CoverSide {
    face: vec3<f32>,
    through: bool,
}

fn cover_side(input: SurfaceInput) -> CoverSide {
    let toward = normalize(input.normal);
    let face = select(-toward, toward, input.front);
    return CoverSide(face, dot(face, sun_direction()) < 0.0);
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return input.color.rgb * select(1.0, COVER_FAR_SIDE, cover_side(input).through);
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return 1.0;
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let side = cover_side(input);
    let lit = select(side.face, -side.face, side.through);
    let bent = normalize(mix(lit, vec3<f32>(0.0, 1.0, 0.0), 0.6));
    let height = input.color.a;
    if foliage().w <= 0.0 || height < 0.0 {
        return bent;
    }
    let tilt = input.uv1;
    let ground = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
    let above = input.uv.x * height;
    return normalize(mix(bent, ground, 1.0 - smoothstep(0.0, foliage().w, above)));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return COVER_ROUGHNESS;
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return mix(COVER_ROOT_OCCLUSION, 1.0, input.uv.x);
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return foliage().x;
}

fn kind(input: SurfaceInput, point: SurfacePoint) -> u32 {
    return SURFACE_THIN;
}
