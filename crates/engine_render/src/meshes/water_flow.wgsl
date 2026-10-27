// Terrains' water's waves (`shade_water`), from its state and the bed it
// flows over: the bed's stones read where they lie, as the ground draws them
// (`water_bed`), raise the surface over them and leave wedges of waves
// behind them (`water_bed_waves`), and what the cook finds standing in it
// does as a solid up through its surface (`water_obstacle_waves`), each by
// the water's own speed and depth there. Its eddies draw no waves of their
// own: the slopes they raise roughen the surface unseen (`water_stirred`).
// Where none rise, the water lies a calm mirror. Its waves are drawn through
// the pixel's own footprint (`water_footprint`), each computed part while it
// spans two pixels along the way it varies (`water_shown`), so waves running
// across the view stay drawn far along it, while those running along it
// average out where the footprint spans them. What a pixel averages of their
// slopes roughens the surface instead (LEAN mapping), along the ways they
// vary, never faded.

// Patterns the flow carries (Vlachos's flow maps), each drawn in two sets
// half a life apart (`water_set`), each set carried from half a life before
// its middle to half a life after, where it starts over unseen and anew, so
// it shows clearest where the flow has not yet stretched it (Neyret's
// latency), a life lasting no longer than the flow's shear takes to strain
// a pattern by WATER_STRAIN.
const WATER_STRAIN: f32 = 0.8;
// The meters across the patches of water whose patterns start over
// together.
const WATER_FLOW_PHASE: f32 = 40.0;
// The waves running water's eddies raise (Froude scaling): as high in root
// mean square as the eddies' speed squared over gravity, their dynamic
// pressure lifting the surface; as long as the eddies are large, one to two
// depths, spread evenly over that octave as the saturation range spreads
// them (their heights' spectrum falling as k⁻³), every way; and drawn out
// by the current twice as long along it as across it. Their slopes'
// variance over their heights', times the depth squared: 2 ln 2 over
// (1/π)² − (1/2π)² for the octave, of which drawing it out keeps 5/8 (half
// the variance along the flow, over two squared, and half across it),
// 5π² ln 2 / 3.
const WATER_EDDY_SLOPES: f32 = 11.401814;
// The share of those slopes' variance that lies along the flow: of the 5/8
// drawing them out keeps, the 1/8 along it, against the 1/2 across it.
const WATER_EDDY_ALONG: f32 = 0.2;
// The steepest slope a wave stands at before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of π/7 at most, 0.32 in
// root mean square).
const WATER_STEEPEST: f32 = 0.44879895;
// How many root mean square heights a sine's crests stand over its mean:
// waves stand at the tops of their crests as high as this over the root of
// their heights' variance.
const WATER_CREST_HEIGHTS: f32 = 1.41421356;
// The meters a stone of the bed stands across for each meter it rises over
// the bed around it: about as round as it is wide, sunk in it to its middle.
const WATER_STONE_WIDTH: f32 = 2.0;
// The sine of the half angle of the wedge of waves standing behind what
// water flows past where it runs deep against those waves' length (Kelvin
// 1887: asin(1/3), 19.47°), and the Froude number from which that wedge
// widens toward the half angle the water's long waves run out at, all of 90°
// as the current reaches their speed, about as Havelock (1908) finds it
// widening in water of finite depth.
const WATER_KELVIN: f32 = 0.33333333;
const WATER_HAVELOCK: f32 = 0.85;
// Below a drop or a jump, where the water runs slower than Froude 1.7 as
// below an undular jump, the length and height of its waves, in depths, and
// the wavelengths over which their train falls off.
const WATER_UNDULAR: vec3<f32> = vec3<f32>(5.0, 0.2, 6.0);

// Water's waves at a point as a pixel shows them: every source's slopes it
// resolves, summed (meters a meter); the covariance of the slopes it hides
// (`water_spread`); the place's whole slope variance, resolved and hidden,
// a Gaussian surface's share steeper than a slope s being e^(−s²/spread);
// how near the tops of the resolved waves' crests the point stands (0 to
// 1), their summed height over WATER_CREST_HEIGHTS roots of their summed
// heights' variance; and the place's whole heights' variance, resolved and
// hidden (square meters), as `spread` sums their slopes'.
struct WaterWaves {
    slopes: vec2<f32>,
    hidden: vec3<f32>,
    spread: f32,
    crest: f32,
    heights: f32,
}

// Waves standing over the bed and about what stands in the water as a
// pixel shows them (`drawn`), and the variance of all their slopes and of
// all their heights, resolved and hidden.
struct WaterStanding {
    drawn: WaterRead,
    spread: f32,
    heights: f32,
}

// One of the two sets (`which`) of a pattern carried in cycles of `cycle`
// seconds, at `time`, the place's clock running `stagger` cycles ahead:
// the seconds it has been carried from its middle (negative before it),
// how much it shows, how far through its cycle it is (0 as it starts) and
// where it lies in the noise, anew each cycle and salted by `salt` (up to
// 64 of the noise's units along each axis).
struct WaterSet {
    seconds: f32,
    weight: f32,
    age: f32,
    offset: vec2<f32>,
}

fn water_set(time: f32, stagger: f32, cycle: f32, which: u32, salt: u32) -> WaterSet {
    let shifted = time / cycle + stagger + 0.5 * f32(which);
    let life = floor(shifted);
    var carried: WaterSet;
    carried.age = shifted - life;
    carried.seconds = (carried.age - 0.5) * cycle;
    carried.weight = 1.0 - abs(2.0 * carried.age - 1.0);
    let key = ((u32(life) * 2u + which) * 0x9e3779b9u) ^ salt;
    carried.offset = vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u)) * 64.0;
    return carried;
}

// What running water's eddies roughen its surface by where they turn at
// `eddy` meters a second in water `depth` meters deep (at least
// WATER_BREAK_THINNEST, as `water_eddies` takes it), drawn as no pattern:
// the variance of the slopes (x) and of the heights (y, square meters) of
// the waves they raise, as high in root mean square as u_t²/g, their
// slopes' variance WATER_EDDY_SLOPES times their heights' over the depth
// squared, and no steeper than waves stand before they break
// (WATER_STEEPEST, a sine's root mean square of it), their heights lowered
// alike. So shallow fast riffles run rough and deep slow pools stay glassy.
fn water_stirred(eddy: f32, depth: f32) -> vec2<f32> {
    let deep = max(depth, WATER_BREAK_THINNEST);
    let lifted = eddy * eddy / WATER_GRAVITY;
    let steepening = WATER_EDDY_SLOPES / (deep * deep);
    let slopes = min(lifted * lifted * steepening, 0.5 * WATER_STEEPEST * WATER_STEEPEST);
    return vec2<f32>(slopes, slopes / steepening);
}

// Standing waves `sum` with a part added whose slopes are `gradient`
// (meters a meter), of which a pixel draws the share `drawn`
// (`water_shown`), what it averages of them hiding in their covariance,
// along them; and whose height is `height` (meters), `amplitude` meters high
// at its tallest there, of which it draws the share `raised`, its heights'
// variance a sine's as tall.
fn water_resolved(sum: WaterStanding, gradient: vec2<f32>, drawn: f32, height: f32,
        amplitude: f32, raised: f32) -> WaterStanding {
    var added = sum;
    added.drawn.slopes += gradient * drawn;
    added.drawn.hidden += water_spread(gradient) * (1.0 - drawn * drawn);
    added.drawn.height += height * raised;
    added.drawn.heights += 0.5 * amplitude * amplitude * raised * raised;
    added.spread += dot(gradient, gradient);
    added.heights += 0.5 * amplitude * amplitude;
    return added;
}

// How fast a height `height` (meters) the pixel computes rises along x and
// z (meters a meter): its change across the pixel and down it, over how the
// point moves on the water, `dx` and `dy` (along x and z) a pixel. Its
// derivatives need it called in uniform control flow.
fn water_rate(height: f32, dx: vec2<f32>, dy: vec2<f32>) -> vec2<f32> {
    let change = vec2<f32>(dpdx(height), dpdy(height));
    let spanned = dx.x * dy.y - dx.y * dy.x;
    return vec2<f32>(change.x * dy.y - change.y * dx.y, change.y * dx.x - change.x * dy.x)
        * (sign(spanned) / max(abs(spanned), 1.0e-12));
}

// The sine of the half angle of the wedge of waves standing behind what
// water at Froude number `froude` flows past: Kelvin's in water running
// deep against them (WATER_KELVIN), widening from WATER_HAVELOCK to all of
// 90° as the current reaches the speed of its long waves, past which the
// fronts those waves pile into trail at asin(1/Fr) (30° at Froude 2).
fn water_wake_sine(froude: f32) -> f32 {
    let deep = mix(WATER_KELVIN, 1.0, smoothstep(WATER_HAVELOCK, 1.0, froude));
    return select(deep, 1.0 / max(froude, 1.0), froude >= 1.0);
}

// The bed water flows over as a pixel reads it (`water_bed`), its stones
// the ground's materials' heights over their mean: how far they rise at
// most (`reach`, meters) and the meters across a stone so tall stands
// (`size`, WATER_STONE_WIDTH of its rise); how far the bed rises over its
// mean at the pixel's point, as finely as the pixel resolves it (`fine`) and
// as broadly as half a stone's width (`broad`); the stone the bed's mean
// along each arm of the wedge of waves standing behind its stones stands for
// (`arms`: its height, were it alone there, meters), each arm read over
// DETAIL_ANISOTROPY of their half widths upstream of the point, and along
// the flow upstream of it (`upstream`); and how much of the bed's broad rise
// (x) and of each arm (yz) a pixel draws (`water_shown`). None of it where
// no stone can come near the surface.
struct WaterBed {
    reach: f32,
    size: f32,
    fine: f32,
    broad: f32,
    arms: vec2<f32>,
    upstream: f32,
    shown: vec3<f32>,
}

// How far a bed's stones `layers` rise over their mean at `point` (meters,
// under it where negative), read through a footprint of sides `dx` and `dy`
// (meters, x, y and z): each of the ground's materials there by its share,
// its heights over their mean as the ground's pixel band takes them
// (`terrain_relief_rise`), filtered over that footprint as every read of the
// ground filters.
fn water_stones(layers: TerrainReliefLayers, point: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    return terrain_relief_rise(layers.a, point, dx, dy)
        + terrain_relief_rise(layers.b, point, dx, dy)
        + terrain_relief_rise(layers.c, point, dx, dy)
        + terrain_relief_rise(layers.d, point, dx, dy);
}

// The mean of how far a bed's stones `layers` rise over their mean
// (`water_stones`) along the line `reaching` meters long running from
// `point` up against `way` (a unit vector along x and z), `wide` meters
// across it: one read of a footprint as long as the line and as wide,
// which anisotropic filtering averages along within DETAIL_ANISOTROPY of
// its width.
fn water_line(layers: TerrainReliefLayers, point: vec3<f32>, way: vec2<f32>, reaching: f32,
        wide: f32) -> f32 {
    let along = vec3<f32>(way.x, 0.0, way.y);
    let across = vec3<f32>(-way.y, 0.0, way.x);
    return water_stones(layers, point - along * (0.5 * reaching), along * reaching,
        across * wide);
}

// A footprint's side `change` (meters, x, y and z), stretched along itself
// to span `meters` at least.
fn water_spanning(change: vec3<f32>, meters: f32) -> vec3<f32> {
    return change * max(1.0, meters / max(length(change), 1.0e-9));
}

// The bed under the point `world` of water `depth` meters deep flowing
// along `flow` (meters a second) at Froude number `froude`, as the ground
// draws it (`WaterBed`): the terrain's ground the water's depth stands over,
// raised by each of its materials' heights over their mean, both bands of
// them (`terrain_relief_layer`), each material of the ground's blend there
// (`terrain_blend`, at the levels a pixel seen from the detail eye shades
// the ground at, `terrain_pixel_shading`) by its share, on ground of the
// water's plane's normal `normal`. Every read lies at a point of the world,
// through the pixel's own footprint (its point's change a pixel `world_dx`
// and `world_dy`, per pixel of the display as the ground's maps take it),
// stretched to span what each read asks at least: the point's own, as
// finely as the pixel resolves it; the broad rise, half a stone across; and
// each line upstream (`water_line`), half a stone across and DETAIL_ANISOTROPY
// times that long, as far as the read can average along it at its width.
// The arms of the wedge leave at the half angle `water_wake_sine` gives. A
// stone's line's mean taken for one stone of it alone gives its height, the
// mean over the line's length over its width. Read nowhere water stands
// still, nor where even the deepest of the terrain's materials
// (`terrain.relief.x`), as tall as all its depth, could raise no stone so
// near the surface that the water climbs it or the share of its rise the
// flow carries up through the water (`water_bed_waves`) shows in an 8-bit
// target (WATER_FOAM_LEAST); then not where the materials here could not.
fn water_bed(world: vec3<f32>, depth: f32, normal: vec3<f32>, flow: vec2<f32>, froude: f32,
        footprint: WaterFootprint, world_dx: vec3<f32>, world_dy: vec3<f32>) -> WaterBed {
    var bed: WaterBed;
    let speed = length(flow);
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let deepest = terrain.relief.x;
    if speed <= 0.0 || deepest <= 0.0 {
        return bed;
    }
    let clear = max(depth - deepest, 0.0);
    if clear > head && exp(-PI * clear / (WATER_STONE_WIDTH * deepest)) < WATER_FOAM_LEAST {
        return bed;
    }
    let point = world - vec3<f32>(0.0, depth, 0.0);
    let blend = terrain_blend(point.xz - terrain.origin.xz, normal, terrain_pixel_shading(point));
    var site: TerrainSite;
    site.world = point;
    site.geometric = normal;
    var layers: TerrainReliefLayers;
    layers.a = terrain_relief_layer(blend.ids.x, blend.weights.x, site);
    layers.b = terrain_relief_layer(blend.ids.y, blend.weights.y, site);
    layers.c = terrain_relief_layer(blend.ids.z, blend.weights.z, site);
    layers.d = terrain_relief_layer(blend.ids.w, blend.weights.w, site);
    let reach = terrain_relief_reach(layers.a).x + terrain_relief_reach(layers.b).x
        + terrain_relief_reach(layers.c).x + terrain_relief_reach(layers.d).x;
    let top = max(depth - reach, 0.0);
    let size = WATER_STONE_WIDTH * reach;
    if reach <= 0.0 || (top > head && exp(-PI * top / size) < WATER_FOAM_LEAST) {
        return bed;
    }
    bed.reach = reach;
    bed.size = size;
    let detail = material_detail();
    let dx = world_dx * detail;
    let dy = world_dy * detail;
    let half = 0.5 * size;
    bed.fine = water_stones(layers, point, dx, dy);
    bed.broad = water_stones(layers, point, water_spanning(dx, half), water_spanning(dy, half));
    bed.shown.x = water_shown(PI * half, footprint.resolved);
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    let sine = water_wake_sine(froude);
    let cosine = sqrt(max(1.0 - sine * sine, 0.0));
    let reaching = DETAIL_ANISOTROPY * half;
    for (var arm = 0u; arm < 2u; arm++) {
        let way = heading * cosine + side * (sine * select(-1.0, 1.0, arm == 0u));
        let span = water_footprint_toward(footprint, vec2<f32>(-way.y, way.x));
        let wide = max(half, span);
        bed.arms[arm] = water_line(layers, point, way, reaching, wide) * (reaching / half);
        bed.shown[arm + 1u] = water_shown(PI * half, span);
    }
    let wide = max(half, water_footprint_toward(footprint, side));
    bed.upstream = water_line(layers, point, heading, reaching, wide) * (reaching / half);
    return bed;
}

// The waves the bed `bed` (`water_bed`) raises in water `depth` meters deep
// flowing along `flow` (meters a second), added to `sum`, for a pixel whose
// footprint is `footprint` (`water_footprint`), its point changing by
// `world_dx` and `world_dy` a pixel. Over its stones the surface rises in
// their own shape, the bed's broad rise, by the share the flow's linear
// response to a bed of their width (π/k, `WaterBed`'s `size`) carries up to
// the surface (Lamb 1932, a stream over an uneven bed): the response of a
// steady current to a bed of wave number k dies by e^(−k·s) up through the s
// meters of water over its top, here the depth over the bed's rise there;
// and it holds F²/(1 + F²)
// of the bed, F² = U²k/g the Froude number of the stone's own waves (twice
// the velocity head times k), near all of it where the current outruns waves
// as long as the stone and the surface follows the bed, and twice the
// velocity head times the stone's steepness where it does not; never higher
// nor lower than the velocity head U²/2g, the most the flow's energy lifts it
// (Bernoulli). Behind each stone a wedge of waves stands, its arms each as
// high along it as the hump the stone its line's mean stands for raises, the
// stones taken as near the surface as the tallest; where the current runs
// past the speed of the water's long waves, they are its fronts. So shallow
// stones near the surface in fast water raise clear humps and wedges, while
// deep stones, whose response dies through the water, and calm water, whose
// head is none, raise nothing. Each part is drawn while it spans two pixels
// (`water_shown`), its slopes the change of its height across the pixel
// (`water_rate`), the rest hiding in their variance (`water_resolved`); and
// the stones finer than the pixel resolves (`water_resolved_share`) roughen
// it as much as the three parts would, each a sine as tall as the tallest
// stones' hump and twice their width long. Its derivatives need it called in
// uniform control flow.
fn water_bed_waves(sum: WaterStanding, bed: WaterBed, flow: vec2<f32>, depth: f32,
        footprint: WaterFootprint, world_dx: vec3<f32>, world_dy: vec3<f32>) -> WaterStanding {
    var standing = sum;
    let speed = length(flow);
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let size = max(bed.size, 1.0e-6);
    let number = PI / size;
    let wave = 2.0 * head * number;
    let response = wave / (1.0 + wave);
    let hump = clamp(response * exp(-number * max(depth - bed.broad, 0.0)) * bed.broad, -head,
        head);
    let carried = response * exp(-number * max(depth - bed.reach, 0.0));
    let tallest = min(carried * bed.reach, head);
    let arms = clamp(carried * bed.arms, vec2<f32>(-head), vec2<f32>(head));
    let dx = world_dx.xz;
    let dy = world_dy.xz;
    standing = water_resolved(standing, water_rate(hump, dx, dy), bed.shown.x, hump, tallest,
        bed.shown.x);
    standing = water_resolved(standing, water_rate(arms.x, dx, dy), bed.shown.y, arms.x, tallest,
        bed.shown.y);
    standing = water_resolved(standing, water_rate(arms.y, dx, dy), bed.shown.z, arms.y, tallest,
        bed.shown.z);
    let finer = 1.0 - water_resolved_share(footprint, size);
    let hidden = 1.5 * (number * tallest) * (number * tallest) * finer;
    standing.drawn.hidden += vec3<f32>(0.5 * hidden, 0.5 * hidden, 0.0);
    standing.spread += hidden;
    return standing;
}

// The waves the obstacle nearest the water, `obstacle` (its offset from the
// point along x and z and its radius, meters, and how strongly a drop or a
// jump lies just upstream), raises in water `depth` meters deep flowing
// along `flow` (meters a second) at Froude number `froude`, the share
// `anchored` of them where a pixel's samples agree on that obstacle, added
// to `sum`, for a pixel whose footprint is `footprint` (`water_footprint`).
// What the cook finds standing in the water, rock breaking its surface, a
// solid standing in it or a boulder under it, whose tops its words do not
// hold, stands as a round solid of its radius up through the surface. The
// water piles up against it and falls along its sides as Bernoulli's
// surface over the potential flow past a circular cylinder lies (Lamb
// 1932): U²/2g (2 (R/r)² cos 2θ − (R/r)⁴) at r from its middle, θ from the
// flow, as high as the velocity head where the flow meets it and three times
// as low beside it, no lower than the bed; its lee, where the flow leaves
// it, keeps only its fall. A wedge of waves trails it as a stone's does
// (`water_bed_waves`), each arm a crest as wide as it, as high as the bed's
// response raises the hump of a stone as wide reaching the surface, from its
// middle out to DETAIL_ANISOTROPY of its radii. Below a drop or a jump,
// where the water runs slower than Froude 1.7 as below an undular jump, a
// train stands of waves about five depths long, varying along the flow. Each
// part is drawn while it spans two pixels along the way it varies
// (`water_shown`, `water_resolved`), a Gaussian w meters in half width as a
// wave πw long: fixed where the bed is, which the flow never smears.
fn water_obstacle_waves(sum: WaterStanding, obstacle: vec4<f32>, flow: vec2<f32>, depth: f32,
        froude: f32, footprint: WaterFootprint, anchored: f32) -> WaterStanding {
    var standing = sum;
    let radius = obstacle.z;
    let speed = length(flow);
    if radius <= 0.0 || speed <= 0.0 || anchored <= 0.0 {
        return standing;
    }
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    // Meters from the obstacle's middle, on its rim where the point lies
    // within it, downstream and across the flow.
    let apart = -obstacle.xy;
    let rim = apart * max(1.0, radius / max(length(apart), 1.0e-6));
    let along = dot(rim, heading);
    let across = dot(rim, side);
    let squared = radius * radius;
    let far = max(dot(rim, rim), squared);
    let rise = (2.0 * squared * (along * along - across * across) - squared * squared)
        / (far * far);
    let rates = vec2<f32>(along * (squared - along * along + 3.0 * across * across),
        across * (squared - 3.0 * along * along + across * across))
        * (4.0 * squared / (far * far * far));
    let kept = select(1.0, 0.0, along > 0.0 && rise > 0.0) * anchored;
    let floored = rise * head < -depth;
    let pillow = max(rise * head, -depth) * kept;
    let pillow_slope = select((heading * rates.x + side * rates.y) * head, vec2<f32>(0.0),
        floored) * kept;
    let about = water_shown(PI * radius, footprint.resolved);
    standing = water_resolved(standing, pillow_slope, about, pillow, head * anchored, about);
    let sine = water_wake_sine(froude);
    let cosine = sqrt(max(1.0 - sine * sine, 0.0));
    let number = PI / (2.0 * radius);
    let wave = 2.0 * head * number;
    let tallest = min(wave / (1.0 + wave) * depth, head) * anchored;
    let reaching = DETAIL_ANISOTROPY * radius;
    for (var arm = 0u; arm < 2u; arm++) {
        let way = heading * cosine + side * (sine * select(-1.0, 1.0, arm == 0u));
        let outward = vec2<f32>(-way.y, way.x);
        let run = dot(apart, way);
        let beside = dot(apart, outward);
        let trailing = smoothstep(0.0, radius, run)
            * (1.0 - smoothstep(reaching - radius, reaching, run));
        let crest = tallest * trailing * exp(-beside * beside / squared);
        let shown = water_shown(PI * radius, water_footprint_toward(footprint, outward));
        standing = water_resolved(standing, outward * (-2.0 * crest * beside / squared), shown,
            crest, tallest, shown);
    }
    let undular = obstacle.w * (1.0 - smoothstep(1.4, 1.7, froude));
    if undular > 0.0 {
        let wavelength = WATER_UNDULAR.x * max(depth, WATER_THINNEST);
        let height = WATER_UNDULAR.y * depth * undular
            * exp(-length(apart) / (WATER_UNDULAR.z * wavelength)) * anchored;
        let rate = 2.0 * PI / wavelength;
        let phase = dot(apart, heading);
        let shown = water_shown(wavelength, water_footprint_toward(footprint, heading));
        standing = water_resolved(standing, heading * (-height * rate * sin(rate * phase)),
            shown, height * cos(rate * phase), height, shown);
    }
    return standing;
}

// The waves of water as `at` holds it, for a pixel whose footprint is
// `footprint` (`water_footprint`), its point changing by `world_dx` and
// `world_dy` a pixel, at Froude number `froude`, its eddies turning at
// `eddy` meters a second (`water_eddies`), `flowing` of the way from still to
// running (0 to 1), over the bed `bed` (`water_bed`), the share `anchored` of
// its nearest obstacle's waves a pixel's samples agree on, as `WaterWaves`
// holds them. Waves stand over its bed's stones and behind them
// (`water_bed_waves`) and about what stands in it (`water_obstacle_waves`),
// and running water's eddies roughen it unseen (`water_stirred`), as far as
// the blend from still to running shows running water, keeping their
// spread, so nothing changes at once where a river meets a lake. Where none
// rise, it reads none: a calm mirror. Its derivatives need it called in
// uniform control flow.
fn water_flow_waves(footprint: WaterFootprint, at: WaterAt, eddy: f32, froude: f32,
        flowing: f32, anchored: f32, bed: WaterBed, world_dx: vec3<f32>,
        world_dy: vec3<f32>) -> WaterWaves {
    let flow = at.words.flow;
    let depth = at.depth;
    // The waves standing over the bed and about what stands in it, each part
    // drawn while it spans two pixels along the way it varies.
    var standing: WaterStanding;
    standing = water_bed_waves(standing, bed, flow, depth, footprint, world_dx, world_dy);
    standing = water_obstacle_waves(standing, at.obstacle, flow, depth, froude, footprint,
        anchored);
    var read = standing.drawn;
    // The share of running water's variances the blend from still to running
    // shows, each field weighed by its share over the root of the shares'
    // squares to keep their spread: the eddies', all hidden, WATER_EDDY_ALONG
    // of theirs along the flow and the rest across it.
    let running_share = flowing * flowing
        / ((1.0 - flowing) * (1.0 - flowing) + flowing * flowing);
    let stirred = water_stirred(eddy, depth) * running_share;
    let speed = length(flow);
    let heading = select(vec2<f32>(1.0, 0.0), flow / speed, speed > 0.0);
    read.hidden += stirred.x * (WATER_EDDY_ALONG * water_spread(heading)
        + (1.0 - WATER_EDDY_ALONG) * water_spread(vec2<f32>(-heading.y, heading.x)));
    var waves: WaterWaves;
    waves.slopes = read.slopes;
    waves.hidden = read.hidden;
    // The whole variance of their slopes, and of their heights, alike.
    waves.spread = standing.spread + stirred.x;
    waves.heights = standing.heights + stirred.y;
    if read.heights > 1.0e-12 {
        waves.crest = saturate(read.height / (WATER_CREST_HEIGHTS * sqrt(read.heights)));
    }
    return waves;
}
