// The wind's push on placed objects, as the mesh module draws them and
// the shadow module casts them, in three layers from where each vertex sits
// on its object: the trunk bends the whole object along the wind, the more
// the higher; its branches bob on a slower wave of their own, the more the
// farther they reach from the object's vertical axis; and its needles and
// leaves flutter quickly there besides. The trunk's mean bend holds at
// every distance; what shakes about it (the gusts' swing, the branches' bob
// and the needles' flutter) fades out as its swing shrinks under about a
// pixel on screen, seen from the detail eye.

// Gusts: two waves running with the wind, each's share of the gust
// strength, rate and variation along it, the quicker one's across x too
// (`engine_core::GUST_*`, which bodies feel).
const GUST_SLOW_SHARE: f32 = 0.6;
const GUST_SLOW_RATE: f32 = 0.9;
const GUST_SLOW_TRAVEL: f32 = 0.05;
const GUST_QUICK_SHARE: f32 = 0.4;
const GUST_QUICK_RATE: f32 = 2.3;
const GUST_QUICK_TRAVEL: f32 = 0.17;
const GUST_QUICK_ACROSS: f32 = 0.31;

// The layers (`gpu_layouts::constants::WIND_*`): the wind speed objects
// sway fully in (meters a second), and the meters up a trunk and out from
// its axis at which they sway their sway; the trunk's own flutter about its
// gusts, its share of the push and its rate (radians a second); the
// branches' share of it, their wave's rate (radians a second) and how far
// its phase turns a meter up (radians); the needles' share and rate; and
// the pixels on screen a shaking layer's swing spans where it shows whole
// and where it has stilled: as other detail, it hands over once it moves
// the object less than a pixel.
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

// How much of a layer shaking `swing` meters either way shows where a meter
// spans `pixels` pixels on screen: all of it while its swing spans
// WIND_SHAKE_WHOLE pixels or more, none by WIND_SHAKE_STILL.
fn wind_shown(swing: f32, pixels: f32) -> f32 {
    return smoothstep(WIND_SHAKE_STILL, WIND_SHAKE_WHOLE, swing * pixels);
}

// How far the wind pushes the vertex at object-space point `local`,
// `height` meters above the root as placed, of an object standing at
// `root` that sways `sway` meters at ten meters up in full wind: the wind
// blowing at `wind.xy` (x and z, meters per second) with gusts of strength
// `wind.z`, at time `wind.w`, where a meter at the root spans `pixels`
// pixels on screen seen from the detail eye (0 where no eye sees it).
//
// The trunk bends the whole object along the wind, growing with the square
// of the height: by its mean bend in the mean wind, and about that by its
// push rising and falling as gusts travel with the wind, with a quicker
// flutter of the object's own. Branches bob as far up as along the wind,
// growing with the square of the vertex's distance from the object's
// vertical axis through its root (`local.xz`, whatever the object's scale
// and turn), on a slower wave whose phase turns three times around the axis
// and on with the height, so neighboring branches move apart. Needles and
// leaves flutter up and across the wind, growing with that distance,
// quicker and smaller, their phase from the vertex's place, so sprays
// shimmer. Branches and needles push with the gusts too. Each shaking layer
// shows by the pixels its widest swing spans (`wind_shown`), so far objects
// keep their mean bend and stand still about it, and an eye seeing none
// sees the mean bend alone.
fn wind_sway(wind: vec4<f32>, pixels: f32, root: vec3<f32>, height: f32,
        local: vec3<f32>, sway: f32) -> vec3<f32> {
    let speed = length(wind.xy);
    if speed < 1.0e-4 {
        return vec3<f32>(0.0);
    }
    let toward = wind.xy / speed;
    let along = vec3<f32>(toward.x, 0.0, toward.y);
    let time = wind.w;
    // The meters the mean wind pushes the object at the span, which every
    // layer scales, and the trunk's mean bend.
    let mean = sway * min(speed / WIND_FULL_SPEED, 1.0);
    let rise = max(height, 0.0) / WIND_SPAN;
    let bend = mean * rise * rise;
    var offset = along * bend;
    // The gusts' swing about the mean push, at most their strength either
    // way, as the waves' shares add to one; and the object's own phase.
    let travel = dot(root.xz, toward);
    let gusts = wind.z
        * (GUST_SLOW_SHARE * sin(time * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
            + GUST_QUICK_SHARE * sin(time * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                + root.x * GUST_QUICK_ACROSS));
    let strongest = 1.0 + abs(wind.z);
    let own = dot(root.xz, vec2<f32>(0.83, 0.57));
    // The trunk shakes about its mean bend by its gusts and its flutter
    // together, at most this share of the bend either way.
    let trunk = wind_shown(bend * (strongest * (1.0 + WIND_TRUNK_FLUTTER) - 1.0), pixels);
    if trunk > 0.0 {
        let flutter = WIND_TRUNK_FLUTTER * sin(time * WIND_TRUNK_RATE + own);
        offset += along * (bend * ((1.0 + gusts) * (1.0 + flutter) - 1.0) * trunk);
    }
    let reach = length(local.xz);
    if reach <= 0.0 {
        return offset;
    }
    // Branches and needles push along slants the square root of two long:
    // how far each of their shares swings the vertex at most, in the
    // strongest gust.
    let lever = reach / WIND_SPAN;
    let slant = sqrt(2.0) * mean * strongest * lever;
    let bobbing = wind_shown(WIND_BRANCH * slant * lever, pixels);
    let shivering = wind_shown(WIND_FLUTTER * slant, pixels);
    if bobbing <= 0.0 && shivering <= 0.0 {
        return offset;
    }
    let weight = mean * (1.0 + gusts) * lever;
    // The cosine and sine of three times the vertex's angle around the
    // axis, which the branches' wave turns by.
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

// Placed objects' sway rides in their slots' flags as a
// float; skinned slots' joint palette offsets read as none.
fn slot_sway(flags: u32) -> f32 {
    let sway = bitcast<f32>(flags);
    return select(0.0, sway, sway > 1.0e-6 && sway <= 10.0);
}
