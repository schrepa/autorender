// The wind's push on placed objects, as the mesh module draws them and
// the shadow module casts them.

// How far the wind pushes a point `height` meters up an
// object standing at `root` that sways `sway` meters at ten meters up in
// full wind, at time `time`, the wind blowing at `velocity` (x and z,
// meters per second) with gusts of strength `gusts`: along the wind,
// growing with the square of the height, its push rising and falling as
// gusts travel with the wind, with a quicker flutter of the object's own.
fn wind_sway(velocity: vec2<f32>, gusts: f32, root: vec3<f32>, height: f32, sway: f32,
        time: f32) -> vec3<f32> {
    let speed = length(velocity);
    if speed < 1.0e-4 {
        return vec3<f32>(0.0);
    }
    let toward = velocity / speed;
    let travel = dot(root.xz, toward);
    let gust = 1.0 + gusts * (0.6 * sin(time * 0.9 - travel * 0.05)
        + 0.4 * sin(time * 2.3 - travel * 0.17 + root.x * 0.31));
    let rise = max(height, 0.0) * 0.1;
    let flutter = 1.0 + 0.12 * sin(time * 4.7 + dot(root.xz, vec2<f32>(0.83, 0.57)));
    let bend = sway * min(speed / 12.0, 1.0) * gust * rise * rise * flutter;
    return vec3<f32>(toward.x, 0.0, toward.y) * bend;
}

// Placed objects' sway rides in their slots' flags as a
// float; skinned slots' joint palette offsets read as none.
fn slot_sway(flags: u32) -> f32 {
    let sway = bitcast<f32>(flags);
    return select(0.0, sway, sway > 1.0e-6 && sway <= 10.0);
}
