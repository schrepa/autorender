import package::frame::view::{detail_across, detail_level, material_detail};
import package::water::footprint::{WaterFootprint, water_spread};
import package::water::particles::PARTICLE_SMALLEST;
import package::water::surface::{WATER_BREAK_THINNEST, WATER_GRADIENT_RANGE, WATER_GRAVITY, water};

@group(2) @binding(24) var water_particles: texture_2d_array<f32>;
@group(2) @binding(25) var water_particles_sampler: sampler;
@group(2) @binding(49) var water_foam_drawn: texture_2d<f32>;

const WATER_FOAM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);
const WATER_FOAM_HOLLOW: f32 = 0.35;
const WATER_FOAM_OPTICAL: f32 = 4.0;
const WATER_DENSEST: f32 = 0.99212598;
const WATER_FOAM_BLOCKING: vec2<f32> = vec2<f32>(0.5, 0.9);

fn water_climbing(under: f32, change: f32, speed: f32, facing: f32) -> f32 {
    let tearing = smoothstep(WATER_FOAM_TEARING.x, WATER_FOAM_TEARING.y, speed);
    if tearing <= 0.0 {
        return 0.0;
    }
    let climbs = speed * speed / (2.0 * WATER_GRAVITY) * facing;
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
    let blocked = 1.0 - standing / max(depth, WATER_BREAK_THINNEST);
    return smoothstep(WATER_FOAM_BLOCKING.x, WATER_FOAM_BLOCKING.y, blocked);
}

struct WaterFoam {
    cover: f32,
    thin: f32,
    shade: f32,
}

fn water_foam(white: f32, foam: f32) -> WaterFoam {
    var result: WaterFoam;
    let layers = max(white, 0.0) + max(foam, 0.0);
    let depth = WATER_FOAM_OPTICAL * layers;
    result.cover = depth / (depth + 2.0);
    result.thin = max(foam, 0.0) / max(layers, 1.0e-6);
    result.shade = WATER_FOAM_HOLLOW * (1.0 - result.thin);
    return result;
}

fn water_layers(share: f32) -> f32 {
    return -log(1.0 - clamp(share, 0.0, WATER_DENSEST));
}

fn water_foam_layers(pixel: vec2<f32>) -> vec2<f32> {
    return textureLoad(water_foam_drawn, vec2<i32>(pixel), 0).xy;
}

fn water_eddy_size(depth: f32, texel: f32, largest: f32) -> f32 {
    let smallest = PARTICLE_SMALLEST * texel;
    return clamp(depth, smallest, max(largest, smallest));
}

struct WaterSplatted {
    fall: vec2<f32>,
    hidden: vec3<f32>,
    shown: f32,
}

fn water_splatted(world: vec3<f32>, footprint: WaterFootprint) -> WaterSplatted {
    var held: WaterSplatted;
    let texel = water.foam.x;
    if texel <= 0.0 {
        return held;
    }
    let picked = detail_level(2.0 * water.foam.z * detail_across(world), texel, 2u);
    if picked.level > 0u {
        return held;
    }
    let side = water.foam.y;
    let at = world.xz / side;
    let dx = footprint.dx / side;
    let dy = footprint.dy / side;
    let splatted = textureSampleGrad(water_particles, water_particles_sampler, at, 0u, dx, dy);
    let product = textureSampleGrad(water_particles, water_particles_sampler, at, 1u, dx,
        dy).x;
    let hidden = vec3<f32>(splatted.zw, product) - water_spread(splatted.xy);
    held.shown = 1.0 - picked.blend;
    held.fall = splatted.xy * held.shown;
    held.hidden = vec3<f32>(max(hidden.xy, vec2<f32>(0.0)), hidden.z);
    return held;
}
