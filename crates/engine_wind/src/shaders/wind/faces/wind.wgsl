import package::generated::pieces::wind::piece;
import package::generated::wind::{GUST_QUICK_ACROSS, GUST_QUICK_RATE, GUST_QUICK_SHARE,
    GUST_QUICK_TRAVEL, GUST_SLOW_RATE, GUST_SLOW_SHARE, GUST_SLOW_TRAVEL, WIND_BRANCH,
    WIND_BRANCH_RATE, WIND_BRANCH_RISE, WIND_FLUTTER, WIND_FLUTTER_RATE, WIND_FULL_SPEED,
    WIND_SHAKE_STILL, WIND_SHAKE_WHOLE, WIND_SPAN, WIND_TRUNK_FLUTTER, WIND_TRUNK_RATE};

fn wind_shown(swing: f32, pixels: f32) -> f32 {
    return smoothstep(WIND_SHAKE_STILL, WIND_SHAKE_WHOLE, swing * pixels);
}

fn wind_sway(wind: vec4<f32>, pixels: f32, root: vec3<f32>, height: f32,
        local: vec3<f32>, sway: f32) -> vec3<f32> {
    let blowing = piece().wind;
    let speed = length(blowing.xy);
    if speed < 1.0e-4 {
        return vec3<f32>(0.0);
    }
    let toward = blowing.xy / speed;
    let along = vec3<f32>(toward.x, 0.0, toward.y);
    let time = wind.w;
    let mean = sway * min(speed / WIND_FULL_SPEED, 1.0);
    let rise = max(height, 0.0) / WIND_SPAN;
    let bend = mean * rise * rise;
    var offset = along * bend;
    let travel = dot(root.xz, toward);
    let gusts = blowing.z
        * (GUST_SLOW_SHARE * sin(time * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
            + GUST_QUICK_SHARE * sin(time * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                + root.x * GUST_QUICK_ACROSS));
    let strongest = 1.0 + abs(blowing.z);
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
