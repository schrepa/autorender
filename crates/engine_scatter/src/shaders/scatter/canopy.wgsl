import package::engine::detail::seen_share;
import package::face::ground_tiles::ground_layer;
import package::generated::pieces::scatter::piece;
import package::generated::scatter::{CANOPY_SIZE_LOW, CANOPY_SIZE_OCTAVES, CANOPY_TOP_OVER_MEAN,
    CanopyMap};

@group(0) @binding(0) var<storage, read> canopy_maps: array<CanopyMap>;

const CANOPY_SIDE: f32 = 1.2732395;

struct CanopyHere {
    found: bool,
    colors: u32,
    shapes: u32,
    uv: vec2<f32>,
    scale: vec2<f32>,
    cell: f32,
    tallest: f32,
    span: f32,
}

struct CanopyStand {
    cover: f32,
    surface: f32,
    height: f32,
    size: f32,
    aspect: f32,
}

fn canopy_here(world: vec3<f32>) -> CanopyHere {
    var here: CanopyHere;
    let count = u32(piece().handover.z);
    for (var index = 0u; index < count; index++) {
        let map = canopy_maps[index];
        let uv = (world.xz - map.rect.xy) * map.rect.zw;
        if map.layers.y > 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            here.found = true;
            here.colors = map.layers.x;
            here.shapes = map.layers.y;
            here.uv = uv;
            here.scale = map.rect.zw;
            here.cell = map.shape.x;
            here.tallest = map.shape.y;
            here.span = map.shape.z;
            return here;
        }
    }
    return here;
}

fn canopy_lod(here: CanopyHere, meters: f32) -> f32 {
    return log2(max(meters / here.cell, 1.0));
}

fn canopy_surface(here: CanopyHere, uv: vec2<f32>, lod: f32) -> f32 {
    return ground_layer(here.shapes, uv, lod).r * here.tallest;
}

fn canopy_stand(here: CanopyHere, uv: vec2<f32>, lod: f32) -> CanopyStand {
    let texel = ground_layer(here.shapes, uv, lod);
    var stand: CanopyStand;
    if texel.g <= 0.0 {
        return stand;
    }
    stand.cover = texel.g;
    stand.surface = texel.r * here.tallest;
    stand.height = stand.surface / texel.g;
    stand.size = CANOPY_SIZE_LOW + CANOPY_SIZE_OCTAVES * texel.b / texel.g;
    let width = exp2(CANOPY_SIZE_LOW + CANOPY_SIZE_OCTAVES * texel.a / texel.g);
    stand.aspect = CANOPY_TOP_OVER_MEAN * stand.height / width;
    return stand;
}

fn canopy_color(here: CanopyHere, lod: f32, ground: vec3<f32>) -> vec3<f32> {
    let texel = ground_layer(here.colors, here.uv, lod);
    if texel.a <= 0.0 {
        return ground;
    }
    let root = texel.rgb / texel.a;
    return root * root;
}

fn canopy_handed(size: f32, across: f32) -> f32 {
    return 1.0 - seen_share(exp2(size) / max(across * piece().handover.x, 1.0e-6));
}

fn canopy_beside(stand: CanopyStand, tangent: f32, above: f32) -> f32 {
    return pow(max(1.0 - stand.cover, 1.0e-6), above * CANOPY_SIDE * stand.aspect * tangent);
}

fn canopy_gap(stand: CanopyStand, tangent: f32) -> f32 {
    return (1.0 - min(stand.cover, 1.0)) * canopy_beside(stand, tangent, 1.0);
}
