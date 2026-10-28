const GUST_SLOW_SHARE: f32 = 0.6;
const GUST_SLOW_RATE: f32 = 0.9;
const GUST_SLOW_TRAVEL: f32 = 0.05;
const GUST_QUICK_SHARE: f32 = 0.4;
const GUST_QUICK_RATE: f32 = 2.3;
const GUST_QUICK_TRAVEL: f32 = 0.17;
const GUST_QUICK_ACROSS: f32 = 0.31;

const WIND_FULL_SPEED: f32 = 12.0;
const WIND_SPAN: f32 = 10.0;
const WIND_TRUNK_FLUTTER: f32 = 0.12;
const WIND_TRUNK_RATE: f32 = 4.7;
const WIND_BRANCH: f32 = 4.0;
const WIND_BRANCH_RATE: f32 = 3.1;
const WIND_BRANCH_RISE: f32 = 2.1;
const WIND_FLUTTER: f32 = 0.2;
const WIND_FLUTTER_RATE: f32 = 13.0;
const WIND_SHAKE_WHOLE: f32 = 1.0;
const WIND_SHAKE_STILL: f32 = 0.5;

fn wind_shown(swing: f32, pixels: f32) -> f32 {
    return smoothstep(WIND_SHAKE_STILL, WIND_SHAKE_WHOLE, swing * pixels);
}

fn wind_sway(wind: vec4<f32>, pixels: f32, root: vec3<f32>, height: f32,
        local: vec3<f32>, sway: f32) -> vec3<f32> {
    let speed = length(wind.xy);
    if speed < 1.0e-4 {
        return vec3<f32>(0.0);
    }
    let toward = wind.xy / speed;
    let along = vec3<f32>(toward.x, 0.0, toward.y);
    let time = wind.w;
    let mean = sway * min(speed / WIND_FULL_SPEED, 1.0);
    let rise = max(height, 0.0) / WIND_SPAN;
    let bend = mean * rise * rise;
    var offset = along * bend;
    let travel = dot(root.xz, toward);
    let gusts = wind.z
        * (GUST_SLOW_SHARE * sin(time * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
            + GUST_QUICK_SHARE * sin(time * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                + root.x * GUST_QUICK_ACROSS));
    let strongest = 1.0 + abs(wind.z);
    let own = dot(root.xz, vec2<f32>(0.83, 0.57));
    let trunk = wind_shown(bend * (strongest * (1.0 + WIND_TRUNK_FLUTTER) - 1.0), pixels);
    if trunk > 0.0 {
        let flutter = WIND_TRUNK_FLUTTER * sin(time * WIND_TRUNK_RATE + own);
        offset += along * (bend * ((1.0 + gusts) * (1.0 + flutter) - 1.0) * trunk);
    }
    let reach = length(local.xz);
    if reach <= 0.0 {
        return offset;
    }
    let lever = reach / WIND_SPAN;
    let slant = sqrt(2.0) * mean * strongest * lever;
    let bobbing = wind_shown(WIND_BRANCH * slant * lever, pixels);
    let shivering = wind_shown(WIND_FLUTTER * slant, pixels);
    if bobbing <= 0.0 && shivering <= 0.0 {
        return offset;
    }
    let weight = mean * (1.0 + gusts) * lever;
    let around = local.xz / reach;
    let turn = vec2<f32>(around.x * (4.0 * around.x * around.x - 3.0),
        around.y * (3.0 - 4.0 * around.y * around.y));
    let beat = time * WIND_BRANCH_RATE + local.y * WIND_BRANCH_RISE + own;
    let bob = WIND_BRANCH * weight * lever * (sin(beat) * turn.x + cos(beat) * turn.y);
    let shiver = WIND_FLUTTER * weight
        * sin(time * WIND_FLUTTER_RATE + dot(local, vec3<f32>(7.1, 9.7, 8.3)) + own);
    return offset + vec3<f32>(toward.x, 1.0, toward.y) * (bob * bobbing)
        + vec3<f32>(-toward.y, 1.0, toward.x) * (shiver * shivering);
}

fn slot_sway(flags: u32) -> f32 {
    let sway = bitcast<f32>(flags);
    return select(0.0, sway, sway > 1.0e-6 && sway <= 10.0);
}
