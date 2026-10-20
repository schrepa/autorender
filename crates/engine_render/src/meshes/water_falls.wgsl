// Waterfalls' sheets (`water_falls.rs`): each fall's water from a little up
// its river over its lip, through the air on each jet and down the chutes
// between, into the water it lands in at last. Each vertex holds where
// across the sheet it lies, how long the water passing it has been on its
// way and how fast it left the lip, so everything is read where the water
// passing a point was across the sheet and when it left the lip: its
// strands, streaks and clumps travel with it and stretch out as it speeds
// up. Water leaves the lip clear and smooth and turns white on its way, in
// patches first, its foam's strands brightest; off a ledge it gathers into
// tongues; the air breaks it up as it flies, its edges first, into strands
// with gaps through which its far side (the same water, the sheet's
// thickness behind) and the rock behind show; it boils where it lands, and
// runs whole down the chutes, its edges ragged, into lace where it runs
// thin, its white breaking into patches where it lets its air go. White
// water scatters the sun's and sky's light, and thin white water glows
// with the sun behind it. It thins out where it meets the rock.
struct FallInput {
    @location(0) position: vec3<f32>,
    // The sheet's face, turned out from the rock it falls from.
    @location(1) normal: vec3<f32>,
    // The way its water runs, and how fast (w, meters a second).
    @location(2) tangent: vec4<f32>,
    // Where across the sheet it lies (-1 to 1), the seconds since the water
    // passing it left the lip (negative up the river), the meters a second
    // it left at and the sheet's half width at the lip (meters).
    @location(3) jet: vec4<f32>,
    // The seconds since the water passing it last took off where it flies,
    // less those since it last landed down a chute, the sheet's thickness
    // (meters), its fall's number, and 1 where its water flies.
    @location(4) fall: vec4<f32>,
    // How white its water has turned, how far the air has broken it up and
    // how much of it shows (0 to 1 each), and 1 where it is a boil.
    @location(5) state: vec4<f32>,
    // How much of the sky it sees (0 to 1), the ground around hiding the
    // rest.
    @location(6) sky: f32,
}

struct FallOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) jet: vec4<f32>,
    @location(4) fall: vec4<f32>,
    @location(5) state: vec4<f32>,
    @location(6) sky: f32,
}

struct FallTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) jet: vec4<f32>,
    @location(4) fall: vec4<f32>,
    @location(5) state: vec4<f32>,
    @location(6) sky: f32,
    @location(7) previous: vec3<f32>,
}

// How far white water heaves out of its sheet at a point: surges and roll
// waves running down with the water (read where the water passing it was
// across the sheet and when it left the lip, `p`, as its pattern is),
// rising and falling a share of the sheet's thickness, most where it has
// turned white and clings to the rock.
fn fall_heave(p: vec2<f32>, shift: vec2<f32>, thickness: f32, white: f32, aloft: f32) -> f32 {
    let waves = water_gradient_noise(p / FALL_HEAVES + shift.yx) * 2.0 - 1.0;
    return waves * min(FALL_HEAVE * thickness, FALL_HEAVIEST) * white * mix(1.0, 0.35, aloft);
}

// The meters a fall's pattern shifts for fall `number`, each fall's its own.
fn fall_shift(number: u32) -> vec2<f32> {
    let key = number * 0x9e3779b9u;
    return vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u)) * 97.0;
}

fn fall_output(input: FallInput) -> FallOutput {
    var output: FallOutput;
    let time = camera.position.w;
    let p = vec2<f32>(input.jet.x * input.jet.w, (time - input.jet.y) * input.jet.z);
    let heave = fall_heave(p, fall_shift(u32(input.fall.z + 0.5)), input.fall.y, input.state.x,
        input.fall.w);
    let position = input.position + input.normal * heave;
    output.clip_position = camera.view_projection * vec4<f32>(position, 1.0);
    output.relative = position - camera.position.xyz;
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.jet = input.jet;
    output.fall = input.fall;
    output.state = input.state;
    output.sky = input.sky;
    return output;
}

@vertex
fn vs_fall(input: FallInput) -> FallOutput {
    return fall_output(input);
}

// The sheet stands still: its motion is the camera's.
@vertex
fn vs_fall_temporal(input: FallInput) -> FallTemporalOutput {
    let output = fall_output(input);
    var temporal: FallTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.normal = output.normal;
    temporal.tangent = output.tangent;
    temporal.jet = output.jet;
    temporal.fall = output.fall;
    temporal.state = output.state;
    temporal.sky = output.sky;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(input.position, 1.0)).xyw;
    return temporal;
}

fn fall_temporal_output(input: FallTemporalOutput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.jet = input.jet;
    output.fall = input.fall;
    output.state = input.state;
    output.sky = input.sky;
    return output;
}

// Falling water: how many times longer along the water than across it its
// map stands, and the meters across and along (at the speed it left its
// lip) it spans; how much its strands and their fibers make of how dense
// it stands, its clumps the rest; the meters across the clumps the air it
// takes in gathers it into,
// and how fast they churn (a second); how much light white water scatters
// back, and how white the foam between its strands stands; the meters its
// strands stand out of the sheet; how much the sheet's far side shows in
// its gaps, and how much darker it stands for the near side's shade; how
// strongly white water scatters the sun forward (Henyey and Greenstein's
// g) and how much of what it scatters a thin sheet lets through; the
// meters over which it thins out where it meets the rock; and the seconds
// water boils for after it lands, churning faster and gathering bigger
// clumps the harder it landed.
const FALL_BOIL: f32 = 0.8;
const FALL_MAP_STRETCH: f32 = 4.0;
const FALL_MAP: vec2<f32> = vec2<f32>(4.0, 4.0 * FALL_MAP_STRETCH);
const FALL_STRANDS: f32 = 0.45;
const FALL_FIBERS: f32 = 0.35;
const FALL_CLUMPS: f32 = 0.55;
const FALL_CHURN: f32 = 1.2;
const FALL_ALBEDO: f32 = 0.8;
const FALL_BETWEEN: f32 = 0.55;
const FALL_RELIEF: f32 = 0.15;
const FALL_FAR: f32 = 0.9;
const FALL_FAR_SHADE: f32 = 0.6;
const FALL_PHASE_G: f32 = 0.5;
const FALL_THROUGH: f32 = 0.15;
// The meters across and along (at the lip's speed) of the waves white
// water heaves in, the share of its thickness it heaves by and the most
// (meters); and the meters across and along of the surges passing down it.
const FALL_HEAVES: vec2<f32> = vec2<f32>(3.0, 6.0);
const FALL_HEAVE: f32 = 0.5;
const FALL_HEAVIEST: f32 = 0.3;
const FALL_SURGES: vec2<f32> = vec2<f32>(9.0, 14.0);
// How much longer down a chute the water's pattern draws out, and the
// meters across and along of the rolls passing down it.
const FALL_STREAMING: f32 = 1.5;
const FALL_ROLLS: vec2<f32> = vec2<f32>(30.0, 5.0);
const FALL_SOFT: f32 = 0.6;
// How thin water running down rock (meters, from the second to the first)
// breaks into lace over it, and how much of it breaks away there; the
// seconds over which a jet gathers into tongues as it leaves its lip, the
// cubic meters a second each meter of its width carries below which it
// gathers (fully below the first), and the meters across its tongues; and
// the meters a second (from the first to the second) over which clear
// water lying in a pool comes to show as running water.
const FALL_THIN: vec2<f32> = vec2<f32>(0.06, 0.25);
const FALL_LACE: f32 = 0.5;
const FALL_GATHERING: f32 = 0.2;
const FALL_GATHERED: vec2<f32> = vec2<f32>(0.4, 2.5);
const FALL_TONGUES: f32 = 2.5;
// The meters across and along (at the lip's speed) of the rock thin water
// leaves bare as it runs down a chute.
const FALL_THREADS: vec2<f32> = vec2<f32>(1.2, 2.5);
// The meters across and along (at the lip's speed) of the lanes thick
// water runs down a chute in, and how much of its air a glassy lane keeps.
const FALL_LANES: vec2<f32> = vec2<f32>(4.0, 18.0);
const FALL_LANE_AIR: f32 = 0.55;
const FALL_POOLED: vec2<f32> = vec2<f32>(1.0, 4.0);
// The meters between the rings running out over a boil, and how high they
// stand where it begins.
const FALL_RINGS: f32 = 0.9;
const FALL_RING: f32 = 0.04;

// What stands of falling water at `p` (meters across, and along at the
// speed it left its lip, of where the water passing it was as it left;
// changing by `dx` and `dy` a pixel), `flown` seconds on its way, its
// pattern `shift` meters along: how dense its strands, their fibers and its
// churning clumps stand there (about a half, x), its clumps alone (y), and
// its foam's bubbles' walls (z).
fn fall_pattern(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flown: f32, shift: vec2<f32>)
        -> vec3<f32> {
    let map = textureSampleGrad(water_fall_map, water_wave_sampler, (p + shift) / FALL_MAP,
        dx / FALL_MAP, dy / FALL_MAP);
    let clumps = water_gradient_noise3(vec3<f32>(p / FALL_CLUMPS + shift * 0.6,
        flown * FALL_CHURN));
    let dense = FALL_STRANDS * map.r + FALL_FIBERS * map.g
        + (1.0 - FALL_STRANDS - FALL_FIBERS) * clumps;
    return vec3<f32>(dense, clumps, map.b);
}

// The share of a pixel a sheet whose pattern reads `density` covers, where
// it breaks up below `threshold`, as sharply as the pattern's change
// across the pixel (`blur`) allows.
fn fall_cover(density: f32, threshold: f32, blur: f32) -> f32 {
    let soft = 0.04 + blur;
    return smoothstep(threshold - soft, threshold + soft, density);
}

// A fall's light and the share of what lies behind it it hides.
struct FallShade {
    shaded: Shaded,
    coverage: f32,
}

fn shade_fall(input: FallOutput) -> FallShade {
    let world = input.relative + camera.position.xyz;
    // Where water is simulated, its water runs over the rock itself: of
    // the sheets, only what falls high over the simulated water below
    // draws, as a fall's curtain does.
    var kept = 1.0;
    if water_simulated(world.xz) {
        kept = smoothstep(SIM_SHEETS_ABOVE.x, SIM_SHEETS_ABOVE.y,
            world.y - sim_water_lowest_at(world.xz));
        if kept < 0.01 {
            discard;
        }
    }
    let time = camera.position.w;
    let side = input.jet.x;
    let flown = input.jet.y;
    let lip_speed = input.jet.z;
    let half = input.jet.w;
    let thickness = input.fall.y;
    let number = u32(input.fall.z + 0.5);
    let aloft = input.fall.w;
    // It boils where it lands, for FALL_BOIL after.
    let boil = (1.0 - smoothstep(0.0, FALL_BOIL, -input.fall.x)) * (1.0 - aloft);
    let speed = max(input.tangent.w, 0.3);
    let falling = max(flown, 0.0);
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let normal = normalize(input.normal);
    let tangent = normalize(input.tangent.xyz);
    let across = cross(tangent, normal);
    // The face the eye sees, and how squarely.
    let facing = dot(normal, view);
    let seen = select(-normal, normal, facing >= 0.0);
    let squarely = max(abs(facing), 0.05);
    // Where the water passing here was across the sheet and when it left
    // the lip (meters at the lip's speed), each fall's pattern its own; and
    // where the view leaves the sheet through its far side, the sheet's
    // thickness behind, in the same terms: as far across and along it as
    // the view runs within it, the water there having left the lip as much
    // earlier as it has run farther.
    let shift = fall_shift(number);
    let p = vec2<f32>(side * half, (time - flown) * lip_speed);
    let crossing = (seen * dot(view, seen) - view) * (thickness / squarely);
    let far_p = p + vec2<f32>(dot(crossing, across), -dot(crossing, tangent) * lip_speed / speed);
    // Down a chute its pattern draws out into streaks along the water, and
    // rolls pass down it in bands across it.
    let streaming = (1.0 - aloft) * (1.0 - boil);
    let drawn = vec2<f32>(p.x, p.y / (1.0 + FALL_STREAMING * streaming));
    let near_p = drawn * (1.0 - 0.35 * boil);
    let near = fall_pattern(near_p, dpdx(near_p), dpdy(near_p), falling * (1.0 + 2.0 * boil),
        shift);
    let far = fall_pattern(far_p, dpdx(far_p), dpdy(far_p), falling,
        shift + vec2<f32>(31.7, 17.3));
    // The air breaks the water up as it flies, its edges first; down a
    // chute it runs whole, its edges ragged where its strands thin out, and
    // thin water running down rock breaks into lace over it.
    let broken = input.state.y * aloft;
    let edge = smoothstep(0.5, 1.0, abs(side));
    let thin = (1.0 - aloft) * (1.0 - boil)
        * (1.0 - smoothstep(FALL_THIN.x, FALL_THIN.y, thickness));
    let threshold = mix(-0.1, 0.45, broken) + FALL_LACE * thin + edge * mix(0.6, 0.7, broken);
    let blur = fwidth(near.x);
    // Off a ledge the water gathers into tongues where its lip dips, the
    // rock showing between them, the more the less each meter of its width
    // carries: each jet its own, opening as it leaves its lip.
    let gathered = aloft * smoothstep(0.0, FALL_GATHERING, input.fall.x)
        * (1.0 - smoothstep(FALL_GATHERED.x, FALL_GATHERED.y, thickness * speed));
    var tongues = 1.0;
    if gathered > 0.0 {
        let took_off = flown - input.fall.x;
        let tongue = water_gradient_noise(vec2<f32>(side * half / FALL_TONGUES,
            took_off * 5.3) + shift);
        tongues = mix(1.0, smoothstep(0.4, 0.56, tongue), gathered);
    }
    // Thin water running down a chute gathers into threads where its rock
    // lies low, leaving the rock's high spots bare: fixed where the rock is
    // (read where the water passes, not when it left its lip), the more
    // the less each meter of its width carries, its edges first.
    let chute = (1.0 - aloft) * (1.0 - boil) * (1.0 - input.state.w)
        * smoothstep(0.3, 0.6, -tangent.y);
    let full = smoothstep(FALL_GATHERED.x, FALL_GATHERED.y, thickness * speed);
    let sparse = chute * (1.0 - full);
    var threads = 1.0;
    if sparse > 0.0 {
        let rock = water_gradient_noise(vec2<f32>(side * half, flown * lip_speed) / FALL_THREADS
            + shift * 0.37);
        threads = mix(1.0, smoothstep(0.35, 0.5, rock + 0.2 * (1.0 - abs(side)) - 0.1), sparse);
    }
    // Thick water runs down a chute in lanes: deeper, faster tongues stay
    // glassier between the white bands it breaks into over the rock, fixed
    // where the rock is.
    var aerated = input.state.x;
    if chute * full > 0.0 {
        let lane = water_gradient_noise(vec2<f32>(side * half, flown * lip_speed) / FALL_LANES
            + shift * 0.53);
        aerated *= mix(1.0, mix(FALL_LANE_AIR, 1.0, smoothstep(0.3, 0.55, lane)), chute * full);
    }
    let near_cover = fall_cover(near.x, threshold, blur) * tongues * threads;
    let far_cover = fall_cover(far.x, threshold, blur) * FALL_FAR * aloft * tongues;
    let cover = near_cover + far_cover * (1.0 - near_cover);
    // Surges: denser white passing down it in broad bands, as the flow
    // pulses.
    let surge = water_gradient_noise(p / mix(FALL_SURGES, FALL_ROLLS, streaming) + shift * 0.3);
    // White where it has taken in air: all of it once the air has reached
    // through it, its densest strands and clumps first while it takes air
    // in or lets it go, lying water laced by its foam's bubbles' walls;
    // glassy between; its strands whitest.
    let density = near.x + 0.3 * (near.y - 0.5);
    let lacing = near.z * smoothstep(0.0, 0.3, aerated) * (1.0 - aloft);
    let patches = max(smoothstep(0.75 - aerated, 0.95 - aerated, density), lacing);
    let white = patches * mix(FALL_BETWEEN, 1.0, max(smoothstep(0.2, 0.8, density), boil))
        * mix(0.85, 1.1, surge);
    // Its strands stand out of the sheet and it heaves as its geometry
    // does: the normal they bend (surface gradients, Mikkelsen).
    // Rings run out over a boil from where its jet lands, as the water
    // carries them, flattening as they spread.
    let out = flown * lip_speed / max(half, 1.0e-3);
    let rings = input.state.w * FALL_RING * (1.0 - smoothstep(0.3, 1.0, out))
        * sin(6.2831853 * p.y / FALL_RINGS);
    let height = near.x * FALL_RELIEF * (1.0 + 0.5 * boil + streaming)
        + fall_heave(p, shift, thickness, aerated, aloft) + rings;
    let r1 = cross(world_dy, seen);
    let r2 = cross(seen, world_dx);
    let det = dot(world_dx, r1);
    let gradient = sign(det) * (dpdx(height) * r1 + dpdy(height) * r2);
    let bent = normalize(abs(det) * seen - gradient);
    // Clear water seen edge on crosses more of itself; slow, lying in a
    // pool, it shows as the pool's water does instead. It fades in over the
    // river up from its lip, and thins out where it meets the rock.
    let clear = (1.0 - pow(0.45, 1.0 / squarely))
        * mix(smoothstep(FALL_POOLED.x, FALL_POOLED.y, speed), 1.0, aloft);
    let behind = water_ground(input.clip_position.xy);
    let meeting = smoothstep(0.0, FALL_SOFT, length(behind) - length(input.relative));
    // A boil's foam breaks into patches toward its rim.
    let rim = input.state.w * smoothstep(0.25, 0.7, out);
    let coverage = cover * mix(clear, 1.0, patches) * smoothstep(-0.3, 0.0, flown) * meeting
        * input.state.z * mix(1.0, smoothstep(0.3, 0.6, near.x), rim) * kept;
    // White water scatters the sun forward through itself: thin, it glows
    // with the sun behind it.
    let sun = lighting.sun.xyz;
    var through = vec3<f32>(0.0);
    if sun.y > 0.0 && dot(sun, seen) < 0.0 {
        let g = FALL_PHASE_G;
        let denominator = 1.0 + g * g + 2.0 * g * dot(sun, view);
        let phase = (1.0 - g * g) / (denominator * sqrt(denominator));
        through = lighting.sun_color.rgb * (FALL_ALBEDO * white * FALL_THROUGH * phase
            * (1.0 - 0.5 * near_cover)) * sun_visibility(world, -seen, world_dx, world_dy,
            length(input.relative), false);
    }
    // Clear water shows the light its body scatters back. The ground
    // around hides as much of the sky from it as it hides from the water.
    let body = water.color.rgb * sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w
        * (0.5 * (1.0 - white) * input.sky);
    var point: SurfacePoint;
    point.albedo = vec3<f32>(FALL_ALBEDO);
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = mix(1.0, 0.8, broken) * input.sky;
    point.bounce = vec3<f32>(0.0);
    point.roughness = mix(0.08, 0.8, white);
    point.alpha = point.roughness * point.roughness;
    point.f0 = vec3<f32>(0.02);
    point.diffuse_color = vec3<f32>(FALL_ALBEDO * white);
    point.geometric = seen;
    point.normal = bent;
    point.distance = length(input.relative);
    point.view = view;
    point.n_v = max(dot(bent, view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 0.0;
    var shade: FallShade;
    shade.shaded = finish(point, body + through);
    // The far side's water shows darker in the near side's shade.
    let shown = near_cover + far_cover * (1.0 - near_cover) * FALL_FAR_SHADE;
    shade.shaded.color *= shown / max(cover, 1.0e-4);
    shade.coverage = coverage;
    if coverage < 0.01 {
        discard;
    }
    return shade;
}

// Falls blend premultiplied over what they fall past, as water does.
@fragment
fn fs_fall(input: FallOutput) -> @location(0) vec4<f32> {
    let shade = shade_fall(input);
    return vec4<f32>(encoded(shade.shaded.color) * shade.coverage, shade.coverage);
}

@fragment
fn fs_fall_post(input: FallOutput) -> SceneTargets {
    let shade = shade_fall(input);
    var targets = scene_targets(shade.shaded);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}

@fragment
fn fs_fall_temporal(input: FallTemporalOutput) -> TemporalTargets {
    let shade = shade_fall(fall_temporal_output(input));
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}
