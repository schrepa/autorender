import package::engine::view::{DetailLevel, detail_across, detail_level};
import package::generated::pieces::shadows::piece;
import package::generated::shadows::{LEVEL_MARGIN, SOFT_GATHERS, ShadowLevel};
import package::place::sun_blockers::SunAt;

@group(0) @binding(0) var shadow_levels: texture_depth_2d_array;
@group(0) @binding(0) var<storage, read> shadow_level_data: array<ShadowLevel>;
@group(0) @binding(0) var shadow_sampler: sampler_comparison;
@group(0) @binding(0) var shadow_texels: sampler;

const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

struct Texels {
    across: vec4<f32>,
    down: vec4<f32>,
}

fn sun_shadowed() -> bool {
    return piece().state.x != 0.0;
}

fn level_count() -> u32 {
    return u32(piece().level_state.x);
}

fn level_texel(level: u32) -> f32 {
    return shadow_level_data[level].band.x;
}

fn crown_softness(softness: f32, per_meter: f32) -> f32 {
    return softness * abs(per_meter);
}

fn texel_slope(dx: vec3<f32>, dy: vec3<f32>, limit: f32) -> vec2<f32> {
    let det = dx.x * dy.y - dx.y * dy.x;
    var slope = vec2<f32>(0.0);
    if abs(det) > 1.0e-12 {
        slope = vec2<f32>(dx.z * dy.y - dy.z * dx.y, dy.z * dx.x - dx.z * dy.x) / det;
    }
    return clamp(slope, vec2<f32>(-limit), vec2<f32>(limit));
}

fn wrapped(texel: vec2<f32>) -> vec2<f32> {
    let side = piece().level_state.z;
    return texel - floor(texel / side) * side;
}

fn gather_corner(texel: vec2<f32>) -> vec2<f32> {
    return floor(texel - vec2<f32>(0.5)) + vec2<f32>(1.0);
}

fn still_depths(level: u32, corner: vec2<f32>) -> vec4<f32> {
    let uv = wrapped(corner) / piece().level_state.z;
    return textureGather(shadow_levels, shadow_texels, uv, level);
}

fn moving_depths(level: u32, corner: vec2<f32>) -> vec4<f32> {
    let uv = corner / piece().level_state.z;
    return textureGather(shadow_levels, shadow_texels, uv, level_count() + level);
}

fn soft_texels() -> f32 {
    return piece().soft.x;
}

fn soft_gathers() -> u32 {
    return u32(piece().soft.y);
}

fn gather_texels(corner: vec2<f32>, point: vec2<f32>) -> Texels {
    let apart = corner - point;
    return Texels(apart.x + vec4<f32>(-0.5, 0.5, 0.5, -0.5),
        apart.y + vec4<f32>(0.5, 0.5, -0.5, -0.5));
}

fn receiver_gaps(level: u32, moving: vec4<f32>, corner: vec2<f32>, texels: Texels, depth: f32,
        slope: vec2<f32>) -> vec4<f32> {
    var depths = still_depths(level, corner);
    if moving.z != 0.0 {
        depths = min(depths, moving_depths(level, corner - moving.xy));
    }
    let plane = depth + slope.x * texels.across + slope.y * texels.down - RECEIVER_MARGIN;
    return max(plane - depths, vec4<f32>(0.0));
}

fn box_shares(texels: Texels, width: vec2<f32>) -> vec4<f32> {
    let across = saturate(vec4<f32>(0.5 * width.x + 0.5) - abs(texels.across));
    let down = saturate(vec4<f32>(0.5 * width.y + 0.5) - abs(texels.down));
    return across * down / (width.x * width.y);
}

fn rows_turn(rows: array<vec4<f32>, 3>, way: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(rows[0].xyz, way), dot(rows[1].xyz, way), dot(rows[2].xyz, way));
}

fn rows_place(rows: array<vec4<f32>, 3>, world: vec3<f32>) -> vec3<f32> {
    return rows_turn(rows, world) + vec3<f32>(rows[0].w, rows[1].w, rows[2].w);
}

fn shadow_level(world: vec3<f32>) -> DetailLevel {
    return detail_level(detail_across(world) * piece().state.y, piece().level_state.y,
        level_count());
}

fn level_inside(level: ShadowLevel, light: vec3<f32>, margin: f32) -> bool {
    return all(light.xy >= level.window.xy + vec2<f32>(margin))
        && all(light.xy < level.window.zw - vec2<f32>(margin)) && light.z > 0.0 && light.z < 1.0;
}

fn level_lift(level: u32, at: SunAt) -> f32 {
    if level < at.lift_first || level - at.lift_first >= 4u {
        return 0.0;
    }
    let lift = level - at.lift_first;
    return select(select(select(at.lifts.w, at.lifts.z, lift == 2u), at.lifts.y, lift == 1u),
        at.lifts.x, lift == 0u);
}

fn sun_spread() -> f32 {
    return piece().state.w;
}

fn lifted_space(level: ShadowLevel, index: u32, sun: SunAt, blend: f32) -> vec3<f32> {
    let up = vec3<f32>(level.rows[0].y, level.rows[1].y, level.rows[2].y);
    return rows_place(level.rows, sun.world + sun.normal * (level.band.z * (1.0 + blend)))
        + up * level_lift(index, sun);
}

fn level_visibility(index: u32, sun: SunAt, blend: f32) -> f32 {
    let level = shadow_level_data[index];
    let light = lifted_space(level, index, sun, blend);
    let gathers = soft_gathers();
    if !level_inside(level, light, f32(gathers)) {
        return -1.0;
    }
    let slope = texel_slope(rows_turn(level.rows, sun.world_dx),
        rows_turn(level.rows, sun.world_dy), level.band.w);
    let first = gather_corner(light.xy) - vec2<f32>(f32(gathers) - 1.0);
    let most = vec2<f32>(soft_texels());
    var blocking = vec2<f32>(0.0);
    for (var row = 0u; row < SOFT_GATHERS; row++) {
        if row >= gathers {
            break;
        }
        for (var column = 0u; column < SOFT_GATHERS; column++) {
            if column >= gathers {
                break;
            }
            let corner = first + 2.0 * vec2<f32>(f32(column), f32(row));
            let texels = gather_texels(corner, light.xy);
            let gap = receiver_gaps(index, level.moving, corner, texels, light.z, slope);
            let share = box_shares(texels, most);
            blocking += vec2<f32>(dot(share, gap), dot(share, sign(gap)));
        }
    }
    let penumbra = 2.0 * sun_spread() * blocking.x
        / (max(blocking.y, 1.0e-6) * level.band.y * level.band.x);
    let lean = slope / (level.band.x * level.band.y);
    let stretch = sqrt(vec2<f32>(1.0) + lean * lean);
    let width = clamp(penumbra * max(stretch.yx / stretch.xy, vec2<f32>(1.0)),
        vec2<f32>(1.0 + blend), most);
    let soft = crown_softness(sun.softness, level.band.y);
    var visible = 0.0;
    for (var row = 0u; row < SOFT_GATHERS; row++) {
        if row >= gathers {
            break;
        }
        for (var column = 0u; column < SOFT_GATHERS; column++) {
            if column >= gathers {
                break;
            }
            let corner = first + 2.0 * vec2<f32>(f32(column), f32(row));
            let texels = gather_texels(corner, light.xy);
            let gap = receiver_gaps(index, level.moving, corner, texels, light.z, slope);
            var lit = vec4<f32>(1.0) - sign(gap);
            if soft > 0.0 {
                lit = exp2(-gap / soft);
            }
            visible += dot(box_shares(texels, width), lit);
        }
    }
    return visible;
}

fn map_lookup(sun: SunAt) -> vec2<f32> {
    let pick = shadow_level(sun.world);
    for (var level = pick.level; level < level_count(); level++) {
        let visible = level_visibility(level, sun, select(0.0, pick.blend, level == pick.level));
        if visible >= 0.0 {
            return vec2<f32>(visible, f32(level));
        }
    }
    return vec2<f32>(1.0, -1.0);
}

fn map_visibility(sun: SunAt) -> f32 {
    return map_lookup(sun).x;
}

fn air_visibility(world: vec3<f32>) -> f32 {
    let pick = shadow_level(world);
    let side = piece().level_state.z;
    for (var index = pick.level; index < level_count(); index++) {
        let level = shadow_level_data[index];
        let light = rows_place(level.rows, world);
        if !level_inside(level, light, LEVEL_MARGIN) {
            continue;
        }
        var visible = textureSampleCompareLevel(shadow_levels, shadow_sampler,
            wrapped(light.xy) / side, index, light.z);
        if level.moving.z != 0.0 {
            visible = min(visible, textureSampleCompareLevel(shadow_levels, shadow_sampler,
                (light.xy - level.moving.xy) / side, level_count() + index, light.z));
        }
        return visible;
    }
    return 1.0;
}
