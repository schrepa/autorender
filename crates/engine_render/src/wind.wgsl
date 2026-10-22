// The wind's push on placed objects, as the mesh module draws them and
// the shadow module casts them, in three layers from where each vertex sits
// on its object: the trunk bends the whole object along the wind, the more
// the higher; its branches bob on a slower wave of their own, the more the
// farther they reach from the object's vertical axis; and its needles and
// leaves flutter quickly there besides. Each layer fades out with the eye's
// distance, branches and needles first.

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
// its axis at which they sway their sway; the branches' share of it, their
// wave's rate (radians a second) and how far its phase turns a meter up
// (radians); the needles' share and rate; and the eye's distances over
// which branches and needles fade out, then the trunk.
const WIND_FULL_SPEED: f32 = 12.0;
const WIND_SPAN: f32 = 10.0;
const WIND_BRANCH: f32 = 4.0;
const WIND_BRANCH_RATE: f32 = 3.1;
const WIND_BRANCH_RISE: f32 = 2.1;
const WIND_FLUTTER: f32 = 0.2;
const WIND_FLUTTER_RATE: f32 = 13.0;
const WIND_DETAIL_NEAR: f32 = 30.0;
const WIND_DETAIL_FAR: f32 = 70.0;
const WIND_TRUNK_NEAR: f32 = 250.0;
const WIND_TRUNK_FAR: f32 = 500.0;

// How far the wind pushes the vertex at object-space point `local`,
// `height` meters above the root as placed, of an object standing at
// `root` that sways `sway` meters at ten meters up in full wind: the wind
// blowing at `wind.xy` (x and z, meters per second) with gusts of strength
// `wind.z`, at time `wind.w`, seen from `eye`.
//
// The trunk bends the whole object along the wind, growing with the square
// of the height, its push rising and falling as gusts travel with the wind,
// with a quicker flutter of the object's own. Branches bob as far up as
// along the wind, growing with the square of the vertex's distance from
// the object's vertical axis through its root (`local.xz`, whatever the
// object's scale and turn), on a slower wave whose phase turns three times
// around the axis and on with the height, so neighboring branches move
// apart. Needles and leaves flutter up and across the wind, growing with
// that distance, quicker and smaller, their phase from the vertex's place,
// so sprays shimmer. Branches and needles push with the gusts too, and
// fade out with the eye's distance before the trunk does; an eye of w 0
// sees the trunk alone, unfaded.
fn wind_sway(wind: vec4<f32>, eye: vec4<f32>, root: vec3<f32>, height: f32,
        local: vec3<f32>, sway: f32) -> vec3<f32> {
    // How much of the trunk's push, then of the branches' and needles',
    // reaches this far from the eye: objects past the trunk's fade cost
    // nothing more.
    var fades = vec2<f32>(1.0, 0.0);
    if eye.w != 0.0 {
        let away = vec2<f32>(length(root - eye.xyz));
        fades = 1.0 - smoothstep(vec2<f32>(WIND_TRUNK_NEAR, WIND_DETAIL_NEAR),
            vec2<f32>(WIND_TRUNK_FAR, WIND_DETAIL_FAR), away);
    }
    let speed = length(wind.xy);
    if speed < 1.0e-4 || fades.x <= 0.0 {
        return vec3<f32>(0.0);
    }
    let toward = wind.xy / speed;
    let time = wind.w;
    let travel = dot(root.xz, toward);
    let gust = 1.0 + wind.z
        * (GUST_SLOW_SHARE * sin(time * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
            + GUST_QUICK_SHARE * sin(time * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                + root.x * GUST_QUICK_ACROSS));
    // The object's own phase, and the meters it is pushed at the span,
    // which every layer scales.
    let own = dot(root.xz, vec2<f32>(0.83, 0.57));
    let push = sway * min(speed / WIND_FULL_SPEED, 1.0) * gust;
    let rise = max(height, 0.0) / WIND_SPAN;
    let flutter = 1.0 + 0.12 * sin(time * 4.7 + own);
    var offset = vec3<f32>(toward.x, 0.0, toward.y) * (push * rise * rise * flutter * fades.x);
    let reach = length(local.xz);
    if fades.y > 0.0 && reach > 0.0 {
        let lever = reach / WIND_SPAN;
        let weight = push * lever * fades.y;
        // The cosine and sine of three times the vertex's angle around the
        // axis, which the branches' wave turns by.
        let around = local.xz / reach;
        let turn = vec2<f32>(around.x * (4.0 * around.x * around.x - 3.0),
            around.y * (3.0 - 4.0 * around.y * around.y));
        let beat = time * WIND_BRANCH_RATE + local.y * WIND_BRANCH_RISE + own;
        let bob = WIND_BRANCH * weight * lever * (sin(beat) * turn.x + cos(beat) * turn.y);
        let shiver = WIND_FLUTTER * weight
            * sin(time * WIND_FLUTTER_RATE + dot(local, vec3<f32>(7.1, 9.7, 8.3)) + own);
        offset += vec3<f32>(toward.x, 1.0, toward.y) * bob
            + vec3<f32>(-toward.y, 1.0, toward.x) * shiver;
    }
    return offset;
}

// Placed objects' sway rides in their slots' flags as a
// float; skinned slots' joint palette offsets read as none.
fn slot_sway(flags: u32) -> f32 {
    let sway = bitcast<f32>(flags);
    return select(0.0, sway, sway > 1.0e-6 && sway <= 10.0);
}
