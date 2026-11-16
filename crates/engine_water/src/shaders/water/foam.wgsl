import package::engine::math::{pcg, unit};
import package::engine::view::material_detail;
import package::generated::water::{FROUDE_GUARD_DEPTH, GRAVITY, WATER_GRADIENT_RANGE};
import package::water::flow::{WaterStream, water_broken, water_carry_moved};
import package::water::footprint::WaterFootprint;
import package::water::uniforms::water;

@group(2) @binding(24) var water_foam_texture: texture_2d<f32>;
@group(2) @binding(25) var water_foam_sampler: sampler;

const WATER_DENSEST: f32 = 0.99212598;

const WATER_FOAM_SALT: u32 = 0x7f4a7c15u;

fn water_climbing(under: f32, change: f32, speed: f32, facing: f32) -> f32 {
    let tearing = smoothstep(water.foam.tearing.x, water.foam.tearing.y, speed);
    if tearing <= 0.0 {
        return 0.0;
    }
    let climbs = speed * speed / (2.0 * GRAVITY) * facing;
    let band = max(max(climbs, change * material_detail()), 1.0e-9);
    return tearing * saturate(1.0 - under / band) * (climbs / band); // filter: a band narrower than a pixel spreading its share over it
}

fn water_contact(standing: f32, change: f32, footprint: WaterFootprint, speed: f32,
        facing: f32) -> f32 {
    let rise = change * material_detail() / max(footprint.along, 1.0e-6);
    let solid = 1.0 - smoothstep(WATER_GRADIENT_RANGE, 2.0 * WATER_GRADIENT_RANGE, rise);
    return solid * water_climbing(standing, change, speed, facing);
}

fn water_blocking(standing: f32, depth: f32) -> f32 {
    let blocked = 1.0 - standing / max(depth, FROUDE_GUARD_DEPTH);
    return smoothstep(water.foam.blocking.x, water.foam.blocking.y, blocked);
}

fn water_whitening(stream: WaterStream, breaks: f32, dropping: f32, contact: f32) -> f32 {
    let broken = water_broken(stream.speed, stream.froude, breaks, dropping);
    return 1.0 - (1.0 - broken) * (1.0 - saturate(contact));
}

fn water_sheer(standing: f32, dropping: f32) -> f32 {
    let falling = smoothstep(water.foam.drop.x, water.foam.drop.y, dropping);
    return mix(1.0, saturate(standing / water.falls.rock_roughness), falling);
}

fn water_filming(stream: WaterStream, standing: f32) -> vec2<f32> {
    let lace = water.shore.lace_amount
        * (1.0 - smoothstep(water.shore.lace_depth.x, water.shore.lace_depth.y, standing))
        * smoothstep(water.shore.lace_speed.x, water.shore.lace_speed.y, stream.speed);
    let lines = smoothstep(water.flow.moving.x, water.flow.moving.y, stream.speed)
        * mix(water.foam.drift_amount.x, water.foam.drift_amount.y, saturate(stream.turbulence));
    return vec2<f32>(lace, lines);
}

fn water_foam_texel(uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    return textureSampleGrad(water_foam_texture, water_foam_sampler, uv, dx, dy);
}

struct WaterWhite {
    layers: vec2<f32>,
    slope: vec2<f32>,
    pores: f32,
}

struct WaterFoamSeen {
    cover: f32,
    film: f32,
    body: f32,
    shade: f32,
    slope: vec2<f32>,
}

fn water_foam_reach(amount: f32) -> f32 {
    let soft = water.foam.soft;
    if amount < 0.5 * soft {
        return sqrt(2.0 * soft * amount);
    }
    if amount > 1.0 - 0.5 * soft {
        return 1.0 + soft - sqrt(2.0 * soft * (1.0 - amount));
    }
    return amount + 0.5 * soft;
}

fn water_foam_seen(stream: WaterStream, amount: f32, dx: vec2<f32>, dy: vec2<f32>)
        -> WaterFoamSeen {
    var seen: WaterFoamSeen;
    let fade = smoothstep(water.foam.fades.x, water.foam.fades.y, stream.across);
    let frames = stream.frames;
    let lateral = vec2<f32>(-frames.along.y, frames.along.x);
    let tile = water.foam.tile;
    let moved = water_carry_moved(stream.carry, tile, stream.time);
    let du = vec2<f32>(dot(dx, frames.along), dot(dx, lateral)) / tile;
    let dv = vec2<f32>(dot(dy, frames.along), dot(dy, lateral)) / tile;
    let rungs = vec2<f32>(unit(pcg(bitcast<u32>(stream.carry.rung) ^ WATER_FOAM_SALT)),
        unit(pcg(bitcast<u32>(stream.carry.rung + 1) ^ WATER_FOAM_SALT)));
    var order = 0.0;
    var squared = 0.0;
    var rest = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let corner = vec2<u32>(k & 1u, k >> 1u);
        let side = vec2<f32>(corner);
        let shares = mix(vec2<f32>(1.0) - frames.weights, frames.weights, side);
        let weight = shares.x * shares.y;
        if weight <= 0.0 {
            continue;
        }
        let away = (frames.within - side) * water.flow.frame;
        let q = vec2<f32>(dot(away, frames.along), dot(away, lateral)) / tile;
        let cell = bitcast<vec2<u32>>(frames.cell + vec2<i32>(corner));
        let draw = pcg(cell.x ^ pcg(cell.y ^ WATER_FOAM_SALT));
        let scatter = vec2<f32>(unit(draw), unit(pcg(draw)));
        let band = water.flow.rung_band;
        let chosen = mix(band, 1.0 - band, unit(pcg(draw ^ 0x9e3779b9u)));
        let higher = smoothstep(chosen - band, chosen + band, stream.carry.toward);
        let kept = weight * inverseSqrt(higher * higher + (1.0 - higher) * (1.0 - higher));
        if higher < 1.0 {
            let texel = water_foam_texel(q + scatter + rungs.x - vec2<f32>(moved.x, 0.0), du, dv);
            order += kept * (1.0 - higher) * (texel.x - 0.5);
            rest += weight * (1.0 - higher) * texel.yzw;
        }
        if higher > 0.0 {
            let texel = water_foam_texel(q + scatter + rungs.y - vec2<f32>(moved.y, 0.0), du, dv);
            order += kept * higher * (texel.x - 0.5);
            rest += weight * higher * texel.yzw;
        }
        squared += weight * weight;
    }
    let rank = saturate(0.5 + order * inverseSqrt(max(squared, 1.0e-6)));
    let over = (rank - 1.0 + water_foam_reach(amount)) / water.foam.soft;
    seen.cover = mix(saturate(over), amount, fade); // filter: structure finer than a pixel draws averages out to its cover
    seen.film = mix(saturate(over / water.foam.halo), amount, fade);
    seen.body = mix(saturate(over / water.foam.body), amount, fade);
    seen.shade = rest.x;
    let tilt = seen.cover * (2.0 * rest.yz - 1.0);
    seen.slope = frames.along * tilt.x + lateral * tilt.y;
    return seen;
}

fn water_white(stream: WaterStream, bumps: vec2<f32>, amount: f32, films: vec2<f32>,
        shear: f32, sheer: f32, dx: vec2<f32>, dy: vec2<f32>) -> WaterWhite {
    var white: WaterWhite;
    white.pores = 0.5;
    if amount <= 0.0 && max(films.x, films.y) <= 0.0 {
        return white;
    }
    let ribbons = films.y
        * smoothstep(water.foam.drift_gathers.x, water.foam.drift_gathers.y, shear);
    let crested = saturate(amount * (1.0 + dot(water.foam.crested, bumps)));
    let asked = saturate(1.0 - (1.0 - crested) * (1.0 - films.x) * (1.0 - ribbons));
    let seen = water_foam_seen(stream, asked, dx, dy);
    let gaps = saturate(water.foam.open * (1.0 - seen.shade)) * (1.0 - seen.body);
    let heaped = seen.cover * (1.0 - gaps);
    let thick = water.foam.covers.y * heaped * mix(water.foam.filmy, 1.0, seen.body) * sheer;
    let thin = water.foam.covers.x * seen.film * (1.0 - heaped);
    white.layers = vec2<f32>(water_foam_heaped(thick), water_foam_heaped(thin));
    white.pores = 1.0 - seen.shade;
    white.slope = -water.foam.steep * seen.slope;
    return white;
}

struct WaterFoam {
    cover: f32,
    thin: f32,
    shade: f32,
}

fn water_foam(white: f32, film: f32, pores: f32) -> WaterFoam {
    var result: WaterFoam;
    let layers = max(white, 0.0) + max(film, 0.0);
    result.cover = 1.0 - 2.0 / (water.foam.optical * layers + 2.0);
    result.thin = max(film, 0.0) / max(layers, 1.0e-6);
    result.shade = water.foam.hollow * (1.0 - result.thin) * pores;
    return result;
}

fn water_foam_heaped(share: f32) -> f32 {
    let covered = clamp(share, 0.0, WATER_DENSEST);
    return 2.0 * covered / (water.foam.optical * (1.0 - covered));
}
