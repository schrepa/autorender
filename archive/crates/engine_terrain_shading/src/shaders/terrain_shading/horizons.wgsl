import package::engine::math::PI;
import package::generated::pieces::terrain_shading::piece;
import package::generated::terrain_shading::{
    HORIZON_CELL_PEAKS, HORIZON_DIRECTIONS, HORIZON_LAYERS, HORIZON_NEAR_CELLS, HORIZON_RECORDS,
    HORIZON_SECTOR_MAPS,
};

@group(0) @binding(14) var terrain_horizons: texture_2d_array<f32>;
@group(0) @binding(16) var terrain_horizon_sampler: sampler;

const HORIZON_NEAR: f32 = f32(HORIZON_NEAR_CELLS);
const HORIZON_PAST: f32 = f32(HORIZON_NEAR_CELLS + 1u);
const HORIZON_PEAKS: f32 = f32(HORIZON_CELL_PEAKS);

fn horizon_records() -> u32 {
    return u32(piece().records[HORIZON_RECORDS].x);
}

fn horizon_bounces() -> bool {
    return piece().records[HORIZON_RECORDS].y > 0.0;
}

struct HorizonClear {
    known: bool,
    low: vec2<f32>,
    high: vec2<f32>,
    pad: f32,
    floor: f32,
    span: f32,
    slopes: vec2<f32>,
}

fn horizon_clear(at: u32, world: vec2<f32>, heading: vec2<f32>) -> HorizonClear {
    var clear: HorizonClear;
    let place = piece().records[at];
    let shape = piece().records[at + 1u];
    let uv = (world - place.xy) * place.zw;
    let cells = textureDimensions(terrain_horizons).x / HORIZON_SECTOR_MAPS;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) || cells == 0u {
        return clear;
    }
    let cell = min(vec2<u32>(uv * f32(cells)), vec2<u32>(cells - 1u));
    let turn = fract(atan2(heading.y, heading.x) / (2.0 * PI) + 1.0) * f32(HORIZON_DIRECTIONS);
    let sector = u32(turn + 0.5) % HORIZON_DIRECTIONS;
    let map = vec2<u32>(sector % HORIZON_SECTOR_MAPS, sector / HORIZON_SECTOR_MAPS) * cells;
    let layer = u32(shape.x) + HORIZON_LAYERS - 1u;
    let floor = textureLoad(terrain_horizons, cell, layer, 0).a;
    let top = textureLoad(terrain_horizons, cell + vec2<u32>(cells, 0u), layer, 0).a;
    let size = 1.0 / (place.zw * f32(cells));
    clear.known = true;
    clear.low = place.xy + (vec2<f32>(cell) - HORIZON_NEAR) * size;
    clear.high = place.xy + (vec2<f32>(cell) + HORIZON_PAST) * size;
    clear.pad = min(size.x, size.y) / HORIZON_PEAKS;
    clear.floor = shape.w + floor;
    clear.span = top - floor;
    clear.slopes = textureLoad(terrain_horizons, map + cell, layer, 0).gb;
    return clear;
}
