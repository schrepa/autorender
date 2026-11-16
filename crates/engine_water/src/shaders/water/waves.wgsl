import package::engine::math::PI;
import package::generated::water::{WATER_GUST_FINE_SALT, WATER_GUST_SALT, WATER_NOISE_REACH,
    WATER_NOISE_WRAP, WATER_PUSH_FINE_SALT, WATER_PUSH_SALT, WATER_STEEPEST, WATER_WAVES_MOST, WaterWaveRow};
import package::water::flow::{water_noise, water_noise_gradient};
import package::water::footprint::water_spread;
import package::water::uniforms::water;

@group(1) @binding(58) var<storage, read> water_waves: array<WaterWaveRow>;

struct WaterSwell {
    slope: vec2<f32>,
    hidden: vec3<f32>,
    height: f32,
    rate: f32,
}

struct WaterNoiseCurve {
    slope: vec2<f32>,
    curve: vec3<f32>,
}

struct WaterPush {
    at: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
    carry: mat2x2<f32>,
    drift: vec2<f32>,
    blur: f32,
}

fn water_calm(speed: f32) -> f32 {
    return 1.0 - smoothstep(water.waves.still.x, water.waves.still.y, speed);
}

fn water_gusts(spot: vec2<f32>) -> vec2<f32> {
    let downwind = water.waves.downwind;
    let blown = vec2<f32>(dot(spot, downwind), dot(spot, vec2<f32>(-downwind.y, downwind.x)));
    let broad = water.waves.gusts[0];
    let fine = water.waves.gusts[1];
    return vec2<f32>(water_noise(blown * broad.z + broad.xy, WATER_GUST_SALT).x,
        water_noise(blown * fine.z + fine.xy, WATER_GUST_FINE_SALT).x) / WATER_NOISE_REACH;
}

fn water_noise_curve(p: vec2<f32>, seed: u32) -> WaterNoiseCurve {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ddu = 60.0 * f * (f * (2.0 * f - 3.0) + 1.0);
    let low = vec2<u32>(vec2<i32>(cell) & vec2<i32>(WATER_NOISE_WRAP - 1));
    let high = vec2<u32>((vec2<i32>(cell) + 1) & vec2<i32>(WATER_NOISE_WRAP - 1));
    let ga = water_noise_gradient(low, seed);
    let gb = water_noise_gradient(vec2<u32>(high.x, low.y), seed);
    let gc = water_noise_gradient(vec2<u32>(low.x, high.y), seed);
    let gd = water_noise_gradient(high, seed);
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
    let mixed = va - vb - vc + vd;
    let bent = ga - gb - gc + gd;
    let east = gb - ga + u.y * bent;
    let north = gc - ga + u.x * bent;
    let rise = vec2<f32>(vb, vc) - va + u.yx * mixed;
    var bowed: WaterNoiseCurve;
    bowed.slope = ga + u.x * east + u.y * (gc - ga) + du * rise;
    bowed.curve = vec3<f32>(2.0 * du.x * east.x + ddu.x * rise.x,
        2.0 * du.y * north.y + ddu.y * rise.y,
        du.y * north.x + du.x * east.y + du.x * du.y * mixed);
    return bowed;
}

fn water_push(spot: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, coarsest: f32) -> WaterPush {
    let shown = water.waves.shown;
    let mean = 0.5 * (dot(dx, dx) + dot(dy, dy));
    let area = dx.x * dy.y - dx.y * dy.x;
    let wide = max(sqrt(mean + sqrt(max(mean * mean - area * area, 0.0))), coarsest);
    var push: WaterPush;
    push.at = spot;
    push.carry = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    for (var octave = 0u; octave < 2u; octave++) {
        let field = water.waves.pushes[octave];
        let ride = water.waves.rides[octave];
        let bowed = water_noise_curve(spot * field.z + field.xy,
            select(WATER_PUSH_SALT, WATER_PUSH_FINE_SALT, octave == 1u));
        push.at += field.w * vec2<f32>(-bowed.slope.y, bowed.slope.x);
        let kept = smoothstep(shown.x, shown.y, 1.0 / max(field.z * wide, 1.0e-12)); // filter: a push octave finer than the pixel or the mesh piece scrambles the waves it would bend
        let bend = (kept * field.w * field.z)
            * mat2x2<f32>(-bowed.curve.z, bowed.curve.x, -bowed.curve.y, bowed.curve.z);
        push.carry += bend;
        push.drift += bend * ride.xy;
        push.blur += (1.0 - kept * kept) * ride.z;
    }
    push.dx = push.carry * dx;
    push.dy = push.carry * dy;
    return push;
}

fn water_rest(index: u32, lift: f32, gusts: vec2<f32>) -> vec3<f32> {
    let rests = water_waves[index].rests;
    return rests[0].xyz + lift * (rests[1].xyz + lift * rests[2].xyz)
        + gusts.x * (rests[3].xyz + gusts.x * rests[5].xyz + lift * rests[8].xyz)
        + gusts.y * (rests[4].xyz + gusts.y * rests[6].xyz + gusts.x * rests[7].xyz
            + lift * rests[9].xyz);
}

fn water_piece(allowed: f32) -> f32 {
    let look = water.waves;
    var coarser = vec2<f32>(0.0);
    for (var index = 0u; index < WATER_WAVES_MOST; index++) {
        if index >= look.count {
            break;
        }
        let wave = water_waves[index].wave;
        let finer = vec2<f32>(wave.wavelength / look.shown.x, wave.left);
        if finer.y <= allowed {
            if index == 0u {
                return finer.x;
            }
            return mix(finer.x, coarser.x,
                (allowed - finer.y) / max(coarser.y - finer.y, 1.0e-12));
        }
        coarser = finer;
    }
    return coarser.x;
}

fn water_finest(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let downwind = water.waves.downwind;
    let side = vec2<f32>(-downwind.y, downwind.x);
    let along = vec2<f32>(dot(dx, downwind), dot(dy, downwind));
    let across = vec2<f32>(dot(dx, side), dot(dy, side));
    let mean = 0.5 * (dot(along, along) + dot(across, across));
    let half = 0.5 * (dot(along, along) - dot(across, across));
    let mixed = dot(along, across);
    let swing = sqrt(half * half + mixed * mixed);
    let area = along.x * across.y - along.y * across.x;
    let narrowest = area * area / max(mean + swing, 1.0e-30);
    let wide = select(vec2<f32>(mixed, swing - half), vec2<f32>(half + swing, mixed), half >= 0.0);
    let narrow = vec2<f32>(-wide.y, wide.x) * inverseSqrt(max(dot(wide, wide), 1.0e-30));
    let off = max(abs(narrow.y) * water.waves.fan.x - abs(narrow.x) * water.waves.fan.y, 0.0);
    return sqrt(narrowest + 2.0 * swing * off * off);
}

fn water_swell(spot: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, shortest: f32, depth: f32,
        calm: f32) -> WaterSwell {
    var swell: WaterSwell;
    let look = water.waves;
    let gusts = water_gusts(spot);
    let lift = max(1.0 + look.gusts[0].w * gusts.x + look.gusts[1].w * gusts.y, look.calmest)
        - 1.0;
    let coarsest = shortest / look.shown.x;
    let push = water_push(spot, dx, dy, coarsest);
    let narrow = max(water_finest(push.dx, push.dy), coarsest);
    let least = look.shown.x * sqrt(narrow * narrow + push.blur);
    let tight = 1.0 / (PI * max(look.breaking * depth, 1.0e-4));
    let band = 1.0 / max(look.shown.y - look.shown.x, 1.0e-6);
    let coarsest_squared = coarsest * coarsest;
    var at = push.at;
    for (var index = 0u; index < WATER_WAVES_MOST; index++) {
        if index >= look.count {
            break;
        }
        let wave = water_waves[index].wave;
        let open = calm
            * max(1.0 + lift * wave.gusted + dot(gusts, wave.group), look.calmest) * wave.steep;
        let tall = open * max(wave.wavelength * tight, 1.0 / WATER_STEEPEST);
        let kept = inverseSqrt(sqrt(1.0 + tall * tall * tall * tall));
        if wave.wavelength <= least {
            swell.hidden += (calm * calm * kept * kept) * water_rest(index, lift, gusts);
            break;
        }
        let steep = open * kept;
        let across = vec2<f32>(dot(push.dx, wave.heading), dot(push.dy, wave.heading));
        let spans = wave.wavelength * inverseSqrt(
            max(max(dot(across, across), coarsest_squared) + push.blur, 1.0e-12));
        let part = saturate((spans - look.shown.x) * band); // filter: a wave narrower than its pixels along its travel averages out to roughness
        let shown = part * part * (3.0 - 2.0 * part);
        swell.hidden += (0.5 * steep * steep * (1.0 - shown * shown)) * water_spread(wave.heading);
        if shown <= 0.0 {
            continue;
        }
        let phase = wave.number * dot(wave.heading, at) + wave.phase;
        let squeeze = look.chop * steep;
        let first = phase - squeeze * cos(phase);
        let back = phase - squeeze * cos(first);
        let risen = sin(back);
        let level = cos(back);
        let reach = steep / wave.number;
        let tilted = shown * steep * level / (1.0 - squeeze * risen);
        swell.slope -= wave.heading * tilted;
        swell.rate -= wave.speed * tilted;
        swell.height += shown * reach * risen;
        at -= wave.heading * (look.chop * reach * level);
    }
    swell.rate += dot(swell.slope, push.drift);
    swell.slope = swell.slope * push.carry;
    let carried = transpose(push.carry) * mat2x2<f32>(swell.hidden.xz, swell.hidden.zy)
        * push.carry;
    swell.hidden = vec3<f32>(carried[0].x, carried[1].y, carried[0].y);
    return swell;
}
