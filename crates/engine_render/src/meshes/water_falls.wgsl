// Waterfalls' jets (`water_falls.rs`): each jet's water through the air,
// from where it leaves its lip to where it meets the water or rock below.
// Each vertex holds where across the sheet it lies, how long the water
// passing it has been on its way and how fast it left the lip, so
// everything is read where the water passing a point was across the sheet
// and when it left the lip: its strands, streaks and clumps travel with it
// and stretch out as it speeds up. Water turns white on its way, in patches
// first, its foam's strands brightest; off a ledge it gathers into tongues;
// the air breaks it up as it flies, its edges first, into strands with gaps
// through which its far side (the same water, the sheet's thickness behind)
// and the rock behind show. Each threshold it takes on its pattern covers
// the same share of it however far off it is (`water_past`). Its clear
// water lets through what lies behind it, as the water's own clarity takes
// it, and mirrors the sky and the sun over it; white water scatters the
// sun's and sky's light, and thin white water glows with the sun behind it.
// It shows only where it stands clear of the water under it, so it joins
// the river at its lip and the pool it lands in, and it thins out where it
// meets the rock.
struct FallInput {
    @location(0) position: vec3<f32>,
    // The sheet's face, turned out from the rock it falls from.
    @location(1) normal: vec3<f32>,
    // The way its water runs, and how fast (w, meters a second).
    @location(2) tangent: vec4<f32>,
    // Where across the sheet it lies (-1 to 1), the seconds since the water
    // passing it left the fall's lip, the meters a second it left at and
    // the sheet's half width there (meters).
    @location(3) jet: vec4<f32>,
    // The seconds since the water passing it took off, the sheet's
    // thickness (meters) and its fall's number.
    @location(4) fall: vec3<f32>,
    // How white its water has turned and how far the air has broken it up
    // (0 to 1 each).
    @location(5) state: vec2<f32>,
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
    @location(4) fall: vec3<f32>,
    @location(5) state: vec2<f32>,
    @location(6) sky: f32,
}

struct FallTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) jet: vec4<f32>,
    @location(4) fall: vec3<f32>,
    @location(5) state: vec2<f32>,
    @location(6) sky: f32,
    @location(7) previous: vec3<f32>,
}

// How far white water heaves out of its sheet at a point: surges running
// down with the water (read where the water passing it was across the
// sheet and when it left the lip, `p`, as its pattern is), rising and
// falling a share of the sheet's thickness where it has turned white.
fn fall_heave(p: vec2<f32>, shift: vec2<f32>, thickness: f32, white: f32) -> f32 {
    let waves = water_gradient_noise(p / FALL_HEAVES + shift.yx) * 2.0 - 1.0;
    return waves * min(FALL_HEAVE * thickness, FALL_HEAVIEST) * white;
}

// The meters a fall's pattern shifts for fall `number`, each fall's its own.
fn fall_shift(number: u32) -> vec2<f32> {
    let key = number * 0x9e3779b9u;
    return vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u)) * 97.0;
}

// Its white water heaves while its waves span two pixels seen from the
// detail eye (`water_shown`), not at all below.
fn fall_output(input: FallInput) -> FallOutput {
    var output: FallOutput;
    let time = camera.position.w;
    let p = vec2<f32>(input.jet.x * input.jet.w, (time - input.jet.y) * input.jet.z);
    let heave = fall_heave(p, fall_shift(u32(input.fall.z + 0.5)), input.fall.y, input.state.x)
        * water_shown(FALL_HEAVES.x, detail_across(input.position));
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
// it stands, its clumps the rest, and how much more its clumps make of
// where it turns white first; the meters across the clumps the air it
// takes in gathers it into, and how fast they churn (a second); how much
// light white water scatters back, and how white the foam between its
// strands stands; the meters its strands stand out of the sheet; how much
// the sheet's far side shows in its gaps, and how much darker it stands for
// the near side's shade; how strongly white water scatters the sun forward
// (Henyey and Greenstein's g) and how much of what it scatters a thin sheet
// lets through; and the meters over which it thins out where it meets the
// rock.
const FALL_MAP_STRETCH: f32 = 4.0;
const FALL_MAP: vec2<f32> = vec2<f32>(4.0, 4.0 * FALL_MAP_STRETCH);
const FALL_STRANDS: f32 = 0.45;
const FALL_FIBERS: f32 = 0.35;
const FALL_CLUMPED: f32 = 0.3;
const FALL_CLUMPS: f32 = 0.55;
const FALL_CHURN: f32 = 1.2;
const FALL_ALBEDO: f32 = 0.8;
const FALL_BETWEEN: f32 = 0.55;
const FALL_RELIEF: f32 = 0.15;
const FALL_FAR: f32 = 0.9;
const FALL_FAR_SHADE: f32 = 0.6;
const FALL_PHASE_G: f32 = 0.5;
const FALL_THROUGH: f32 = 0.15;
const FALL_SOFT: f32 = 0.6;
// Falling water's map's strands and fibers (`water_fall_map`'s r and g,
// `engine_tables::falls`): the texels across and along it their grain
// spans, the strands' coarser octave's cells, and the texels a fiber's
// noise is smoothed across and integrated along its line, by which a read
// resolves each; and how far each spreads about its middle of a half.
const FALL_MAP_STRAND_ACROSS: f32 = 32.0;
const FALL_MAP_STRAND_ALONG: f32 = 64.0;
const FALL_MAP_FIBER_ACROSS: f32 = 3.0;
const FALL_MAP_FIBER_ALONG: f32 = 33.0;
const FALL_MAP_SPREAD: f32 = 0.17;
// How far gradient noise spreads about its middle of a half
// (`water_gradient_noise`, `water_gradient_noise3`): its standard
// deviation, 0.16 in two dimensions and 0.15 in three, as their quintic
// weights spread lattice gradients uniform over a square and a cube.
const FALL_NOISE_SPREAD: f32 = 0.155;
// The meters across and along (at the lip's speed) of the waves white
// water heaves in, the share of its thickness it heaves by and the most
// (meters); and the meters across and along of the surges passing down it.
const FALL_HEAVES: vec2<f32> = vec2<f32>(3.0, 6.0);
const FALL_HEAVE: f32 = 0.175;
const FALL_HEAVIEST: f32 = 0.105;
const FALL_SURGES: vec2<f32> = vec2<f32>(9.0, 14.0);
// The share of a jet its strands cover: all of it as it leaves its lip,
// FALL_BROKEN of it once the air has broken it up, and toward its edges,
// where it breaks up first, FALL_EDGE of it while whole and none once
// broken. The seconds over which a jet gathers into tongues as it leaves
// its lip, the cubic meters a second each meter of its width carries below
// which it gathers (fully below the first), the meters across its tongues
// and the share of its width they take.
const FALL_BROKEN: f32 = 0.7;
const FALL_EDGE: f32 = 0.5;
const FALL_GATHERING: f32 = 0.2;
const FALL_GATHERED: vec2<f32> = vec2<f32>(0.4, 2.5);
const FALL_TONGUES: f32 = 2.5;
const FALL_TONGUED: f32 = 0.56;

// What stands of falling water at `p` (meters across, and along at the
// speed it left its lip, of where the water passing it was as it left;
// changing by `dx` and `dy` a pixel), `flown` seconds on its way, its
// pattern `shift` meters along, its clumps drawn `clumping` of the way
// (`water_shown`), their mean the rest: how dense its strands, their fibers
// and its churning clumps stand there (x, about a half), and its clumps
// alone (y). Its map samples as finely as the display shows it: the mip
// bias.
fn fall_pattern(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flown: f32, shift: vec2<f32>,
        clumping: f32) -> vec2<f32> {
    let detail = material_detail();
    let map = textureSampleGrad(water_fall_map, water_fall_sampler, (p + shift) / FALL_MAP,
        dx * detail / FALL_MAP, dy * detail / FALL_MAP);
    var clumps = 0.5;
    if clumping > 0.0 {
        clumps += (water_gradient_noise3(vec3<f32>(p / FALL_CLUMPS + shift * 0.6,
            flown * FALL_CHURN)) - 0.5) * clumping;
    }
    let dense = FALL_STRANDS * map.r + FALL_FIBERS * map.g
        + (1.0 - FALL_STRANDS - FALL_FIBERS) * clumps;
    return vec2<f32>(dense, clumps);
}

// How far a sum of falling water's pattern's parts spreads, its strands,
// their fibers and its clumps each taken `parts` times (x), and the share
// of its variance a pixel keeping `kept` of each part's resolves (y).
fn fall_spread(parts: vec3<f32>, kept: vec3<f32>) -> vec2<f32> {
    let spreads = parts * vec3<f32>(FALL_MAP_SPREAD, FALL_MAP_SPREAD, FALL_NOISE_SPREAD);
    let variances = spreads * spreads;
    let variance = variances.x + variances.y + variances.z;
    return vec2<f32>(sqrt(variance), dot(variances, kept) / variance);
}

// The simulated water under `point` (x and z, world meters), as its mesh
// between its cells' middles holds it (`sim_water_output`'s, each square
// of four middles split from its first corner to its last): its surface
// (world meters), and how much of it its shading shows there, none where
// it is dry or its mesh stands sheer, down a cliff (SIM_SHEER).
fn fall_sim_water(point: vec2<f32>) -> vec2<f32> {
    let cells = (point - water.sim.xy) / water.sim.z - 0.5;
    let corner = vec2<i32>(floor(cells));
    let within = cells - floor(cells);
    let first = sim_water_level(corner.x, corner.y);
    let last = sim_water_level(corner.x + 1, corner.y + 1);
    // The triangle's third corner, its weights of its first, third and last
    // corners, and how far its surface rises along x and z across a cell.
    var third: vec3<f32>;
    var weights: vec3<f32>;
    var rise: vec2<f32>;
    if within.x >= within.y {
        third = sim_water_level(corner.x + 1, corner.y);
        weights = vec3<f32>(1.0 - within.x, within.x - within.y, within.y);
        rise = vec2<f32>(third.x - first.x, last.x - third.x);
    } else {
        third = sim_water_level(corner.x, corner.y + 1);
        weights = vec3<f32>(1.0 - within.y, within.y - within.x, within.x);
        rise = vec2<f32>(last.x - third.x, third.x - first.x);
    }
    let level = first * weights.x + third * weights.y + last * weights.z;
    let slope = rise / water.sim.z;
    let upright = inverseSqrt(1.0 + dot(slope, slope));
    let shown = smoothstep(SIM_SHEER.x, SIM_SHEER.y, upright) * select(0.0, 1.0, level.y > 0.0);
    return vec2<f32>(level.x + water.sim_datum.x, shown);
}

// How much of a jet `thickness` meters thick shows at `world` (world
// meters) over the water the pass shows under it: none within its
// thickness of that water's surface, all of it from twice its thickness
// over it, and all of it where no water lies under it. That water is the
// simulated water within a live window (`water_simulated`,
// `fall_sim_water`), and elsewhere tiles' water, as a pixel there resolves
// level water seen from the detail eye (`water_pixel`), none lying past
// where the samples holding its surface end (WATER_HELD).
fn fall_clear(world: vec3<f32>, thickness: f32) -> f32 {
    var surface = 0.0;
    var under = 0.0;
    if water_simulated(world.xz) {
        let simulated = fall_sim_water(world.xz);
        surface = simulated.x;
        under = simulated.y;
    } else {
        let at = water_pixel(world.xz - terrain.origin.xz,
            detail_resolved(detail_span(world, vec3<f32>(0.0, 1.0, 0.0)).xy));
        surface = terrain.origin.y + at.surface;
        under = select(0.0, 1.0, at.depth > 0.0 && at.held >= WATER_HELD);
    }
    return mix(1.0, smoothstep(thickness, 2.0 * thickness, world.y - surface), under);
}

// A fall's light, how much of the pixel it shows (its far side's darker in
// the near side's shade, as far as the near side lets it through) and the
// share of what lies behind it it hides.
struct FallShade {
    shaded: Shaded,
    shown: f32,
    coverage: f32,
}

fn shade_fall(input: FallOutput) -> FallShade {
    let world = eye_world(input.relative);
    let thickness = input.fall.y;
    // Within its thickness of the water under it, that water shows instead.
    let clear = fall_clear(world, thickness);
    if clear < 0.01 {
        discard;
    }
    let time = camera.position.w;
    let side = input.jet.x;
    let flown = input.jet.y;
    let lip_speed = input.jet.z;
    let half = input.jet.w;
    let number = u32(input.fall.z + 0.5);
    let speed = max(input.tangent.w, 0.3);
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let normal = normalize(input.normal);
    let tangent = normalize(input.tangent.xyz);
    let across = cross(tangent, normal);
    // The face the eye sees, and how squarely.
    let facing = dot(normal, view); // optics: the face of the sheet the eye sees
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
    let crossing = (seen * dot(view, seen) - view) * (thickness / squarely); // optics: the view's way through the sheet
    let far_p = p + vec2<f32>(dot(crossing, across), -dot(crossing, tangent) * lip_speed / speed);
    // The pixel's footprint on its pattern (`water_footprint`), by which its
    // computed noises draw while their cells span two pixels, their means
    // below (`water_shown`); and the share of each part's variance it
    // keeps: the map's strands and fibers as its levels and anisotropy
    // average them, read in their grain's own sizes
    // (`water_resolved_share`), and the clumps as far as they draw.
    let p_dx = dpdx(p);
    let p_dy = dpdy(p);
    let footprint = water_footprint(p_dx, p_dy);
    let clumping = water_shown(FALL_CLUMPS, footprint.resolved);
    let texel = FALL_MAP / vec2<f32>(textureDimensions(water_fall_map));
    let strand = texel * vec2<f32>(FALL_MAP_STRAND_ACROSS, FALL_MAP_STRAND_ALONG);
    let fiber = texel * vec2<f32>(FALL_MAP_FIBER_ACROSS, FALL_MAP_FIBER_ALONG);
    let kept = vec3<f32>(water_resolved_share(water_footprint(p_dx / strand, p_dy / strand), 1.0),
        water_resolved_share(water_footprint(p_dx / fiber, p_dy / fiber), 1.0),
        clumping * clumping);
    let near = fall_pattern(p, p_dx, p_dy, flown, shift, clumping);
    let far = fall_pattern(far_p, dpdx(far_p), dpdy(far_p), flown,
        shift + vec2<f32>(31.7, 17.3), clumping);
    // How far its density spreads and how much of that the pixel resolves;
    // and the same of where it turns white first, its clumps counting more.
    let parts = vec3<f32>(FALL_STRANDS, FALL_FIBERS, 1.0 - FALL_STRANDS - FALL_FIBERS);
    let dense = fall_spread(parts, kept);
    let whitening = fall_spread(parts + vec3<f32>(0.0, 0.0, FALL_CLUMPED), kept);
    // The air breaks the water up as it flies, its edges first: on each of
    // its sides its strands cover the share of it the air leaves whole, past
    // the threshold that share sets on how dense it stands (`water_past`).
    let broken = input.state.y;
    let edge = smoothstep(0.5, 1.0, abs(side));
    let covered = mix(mix(1.0, FALL_BROKEN, broken), mix(FALL_EDGE, 0.0, broken), edge);
    let strands = water_past((vec2<f32>(near.x, far.x) - 0.5) / dense.x, vec2<f32>(dense.y),
        vec2<f32>(covered));
    // White where it has taken in air, its densest strands and clumps
    // first: of what its strands cover on each side, as much as its water
    // has turned white, past the threshold that share of the pixel sets; and
    // the share of each side's water white.
    let aerated = input.state.x;
    let densities = vec2<f32>(near.x, far.x) + FALL_CLUMPED * (vec2<f32>(near.y, far.y) - 0.5);
    let whites = water_past((densities - 0.5) / whitening.x, vec2<f32>(whitening.y),
        vec2<f32>(aerated * covered));
    let whiteness = min(whites.covers, strands.covers) / max(strands.covers, vec2<f32>(1.0e-4));
    // Off a ledge the water gathers into tongues where its lip dips, the
    // rock showing between them through both its sides, the more the less
    // each meter of its width carries: each jet its own, opening as it
    // leaves its lip, drawn while they span two pixels across it,
    // FALL_TONGUED of its width.
    let gathered = smoothstep(0.0, FALL_GATHERING, input.fall.x)
        * (1.0 - smoothstep(FALL_GATHERED.x, FALL_GATHERED.y, thickness * speed));
    var tongue = 0.0;
    var tongue_kept = 0.0;
    if gathered > 0.0 {
        let took_off = flown - input.fall.x;
        let tongued = water_shown(FALL_TONGUES,
            water_footprint_toward(footprint, vec2<f32>(1.0, 0.0)));
        tongue = (water_gradient_noise(vec2<f32>(side * half / FALL_TONGUES, took_off * 5.3)
            + shift) - 0.5) * tongued / FALL_NOISE_SPREAD;
        tongue_kept = tongued * tongued;
    }
    let tongues = mix(1.0, water_past(vec2<f32>(tongue, 0.0), vec2<f32>(tongue_kept, 0.0),
        vec2<f32>(FALL_TONGUED, 0.0)).covers.x, gathered);
    let near_cover = strands.covers.x;
    let far_cover = strands.covers.y * FALL_FAR;
    // Its strands stand out of the sheet and it heaves as its geometry
    // does: the normal they bend (surface gradients, Mikkelsen).
    let height = near.x * FALL_RELIEF + fall_heave(p, shift, thickness, aerated)
        * water_shown(FALL_HEAVES.x, footprint.resolved);
    let r1 = cross(world_dy, seen);
    let r2 = cross(seen, world_dx);
    let det = dot(world_dx, r1);
    let gradient = sign(det) * (dpdx(height) * r1 + dpdy(height) * r2);
    let bent = normalize(abs(det) * seen - gradient);
    // Each of the sheet's faces takes half the view's way through it: its
    // clear water lets through what lies behind it as the water's own
    // clarity takes it over that way (`water_transmitted`, by the river's
    // water's settings), and its white water hides it. What lies behind the
    // near face shows through its gaps and its clear water, the far side
    // among it, darker in its shade. It thins out where it meets the rock.
    let transmit = water_transmitted(0.5 * thickness / squarely, 0.0);
    let opacities = mix(vec2<f32>(1.0 - luminance(transmit)), vec2<f32>(1.0), whiteness);
    let behind = water_ground(input.clip_position.xy);
    let meeting = smoothstep(0.0, FALL_SOFT, length(behind) - length(input.relative)); // optics: what the view meets behind the sheet
    let passed = 1.0 - near_cover * opacities.x;
    let faces = vec2<f32>(near_cover, FALL_FAR_SHADE * far_cover * passed);
    let shown = (faces.x + faces.y) * tongues * clear * meeting;
    let coverage = (1.0 - passed * (1.0 - far_cover * opacities.y)) * tongues * clear * meeting;
    // The share of the water it shows white; surges, denser white passing
    // down it in broad bands as the flow pulses, drawn while they span two
    // pixels; glassy between its strands, its strands whitest.
    let shown_white = dot(faces, whiteness) / max(faces.x + faces.y, 1.0e-4);
    let surge = 0.5 + (water_gradient_noise(p / FALL_SURGES + shift * 0.3) - 0.5)
        * water_shown(FALL_SURGES.x, footprint.resolved);
    let white = shown_white * mix(FALL_BETWEEN, 1.0, smoothstep(0.2, 0.8, densities.x))
        * mix(0.85, 1.1, surge);
    // The sun's light on the sheet as its shadows leave it, through the
    // face it stands before: in front of the face the eye sees, the clear
    // water's body scatters it back; behind it, white water scatters it
    // forward through itself, and thin, it glows with the sun behind it.
    let sun = lighting.sun.xyz;
    let backlit = dot(sun, seen) < 0.0;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb * sun_visibility(world, select(seen, -seen, backlit),
            world_dx, world_dy, length(input.relative), false);
    }
    var through = vec3<f32>(0.0);
    if backlit {
        let g = FALL_PHASE_G;
        let denominator = 1.0 + g * g + 2.0 * g * dot(sun, view);
        let phase = (1.0 - g * g) / (denominator * sqrt(denominator));
        through = sunlight * (FALL_ALBEDO * white * FALL_THROUGH * phase
            * (1.0 - 0.5 * near_cover));
    }
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
    // Clear water's body scatters back what it takes of the light the
    // surface lets into it, as the river's does (`water_body`,
    // `water_scattered`); its Fresnel reflection lies over it. The ground
    // around hides as much of the sky from it as it hides from the water.
    let reflectance = luminance(environment_brdf(point.f0, point.roughness, point.n_v));
    let body = water_body(0.0) * (vec3<f32>(1.0) - transmit)
        * water_scattered(view, sunlight, seen)
        * ((1.0 - reflectance) * (1.0 - shown_white) * input.sky);
    var shade: FallShade;
    shade.shaded = finish(point, body + through);
    shade.shown = shown;
    shade.coverage = coverage;
    if shown < 0.01 {
        discard;
    }
    return shade;
}

// Fall shade `shade` at `pixel` with the shares screen-space reflections
// take of what the pixel shows (`finish`'s of its own light): what it
// mirrors makes up the less of all the pixel shows the more of the opaque
// scene behind it (`water_scene`, which frames drawing the effects' targets
// copy) it lets through, and the scene mirrored in its place goes as far as
// it shows.
fn fall_mirroring(shade: FallShade, pixel: vec2<f32>) -> FallShade {
    var result = shade;
    let own = luminance(shade.shaded.color) * shade.shown;
    let behind = luminance(textureLoad(water_scene, vec2<i32>(pixel), 0).rgb)
        * (1.0 - shade.coverage);
    result.shaded.mirrored *= own / max(own + behind, 1.0e-6);
    result.shaded.reflect *= shade.shown;
    return result;
}

// Falls blend premultiplied over what they fall past, as water does: their
// light as far as they show, what lies behind them as far as they let it
// through.
@fragment
fn fs_fall(input: FallOutput) -> @location(0) vec4<f32> {
    let shade = shade_fall(input);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.shown,
        shade.coverage);
}

@fragment
fn fs_fall_post(input: FallOutput) -> SceneTargets {
    let pixel = input.clip_position.xy;
    let shade = fall_mirroring(shade_fall(input), pixel);
    var targets = scene_targets(shade.shaded, pixel);
    targets.color = vec4<f32>(stored(shade.shaded.color, pixel) * shade.shown, shade.coverage);
    return targets;
}

@fragment
fn fs_fall_temporal(input: FallTemporalOutput) -> TemporalTargets {
    let pixel = input.clip_position.xy;
    let shade = fall_mirroring(shade_fall(fall_temporal_output(input)), pixel);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = vec4<f32>(stored(shade.shaded.color, pixel) * shade.shown, shade.coverage);
    return targets;
}
