// Terrains' water's waves (`shade_water`), from its state alone: the
// wind's (`water_spectrum.wgsl`), standing where they evolve on still water
// and carried by the current on running water, in sets living as long as
// its eddies take to turn them over; the waves its bed holds still, and the
// boils its eddies bring up, carried by the current and calming the wind's
// waves on them (`water_stream.wgsl`); and waves standing about what it
// flows past. Its eddies draw no waves of their own: the slopes they raise
// roughen the surface unseen (`water_stirred`). Every source is drawn as far
// as its waves span the steady size (STEADY_PIXELS), and what a pixel
// averages of the finer ones' slopes roughens the surface instead (LEAN
// mapping), never faded; a source with no share, or whose waves the steady
// size hides whole, reads nothing, its slopes' variance known from what
// raised them.

// Patterns the flow carries (Vlachos's flow maps), each drawn in two sets
// half a life apart, each set carried from half a life before its middle to
// half a life after, where it starts over unseen and anew in its tiles, so
// it shows clearest where the flow has not yet stretched it (Neyret's
// latency). A life lasts as long as the place's eddies take to turn its
// waves over, no longer than the flow's shear takes to strain a pattern by
// WATER_STRAIN, in whole octaves of WATER_SHORTEST_LIFE: each place takes
// the two octaves nearest its own, crossfading over the middle share
// WATER_BAND_BLEND of each, so lives change smoothly across the water.
// Strain fades what the flow has drawn out as it runs (Chentanez and
// Müller: e^(−Ω τ |∇u|), Ω being WATER_STRAIN_FADE).
const WATER_STRAIN: f32 = 0.8;
const WATER_BAND_BLEND: vec2<f32> = vec2<f32>(0.25, 0.75);
const WATER_STRAIN_FADE: f32 = 1.0;
// The shortest life (seconds) a carried pattern lives: fifteen frames at
// 60 Hz, over fewer of which its sets' coming and going reads as flicker.
const WATER_SHORTEST_LIFE: f32 = 0.25;
// The least shear (a second) a place strains its patterns by: half a step
// of the shear the words pack, the least they tell apart from none.
const WATER_LEAST_SHEAR: f32 = WATER_SHEAR_RANGE / 510.0;
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
// The steepest slope a wave stands at before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of π/7 at most, 0.32 in
// root mean square).
const WATER_STEEPEST: f32 = 0.44879895;
// How many root mean square heights a sine's crests stand over its mean:
// waves stand at the tops of their crests as high as this over the root of
// their heights' variance.
const WATER_CREST_HEIGHTS: f32 = 1.41421356;
// Standing waves (Neyret and Praizelin; Yu, Neyret and Steed): the least
// radius (meters) of what stands in the water, and the least speed (meters
// a second) of the water flowing past it, that raise any; how far upstream
// of its obstacle's middle a pillow heaps up, how long it runs along the
// flow and how high it rises at most, in radii (the water it heaps up
// spills round the obstacle); a front's crest's half width, in depths and
// at least (meters), and the radii over which fronts fall off downstream;
// the share of the velocity head the waves standing past obstacles in
// slower water rise by, how much their wake widens a meter downstream, and
// the wavelengths over which trains fall off; and the length and height of
// an undular jump's waves, in depths.
const WATER_STANDING_LEAST: vec2<f32> = vec2<f32>(0.0625, 0.2);
const WATER_PILLOW: vec3<f32> = vec3<f32>(0.5, 0.7, 0.5);
const WATER_FRONT_CREST: vec2<f32> = vec2<f32>(2.0, 0.03);
const WATER_FRONT_REACH: f32 = 8.0;
const WATER_TRAIN: f32 = 0.3;
const WATER_WAKE: f32 = 0.25;
const WATER_TRAIN_REACH: f32 = 6.0;
const WATER_UNDULAR: vec2<f32> = vec2<f32>(5.0, 0.2);

// Water's waves at a point as a pixel shows them: every source's slopes it
// resolves, summed (meters a meter); the variance of the slopes it hides;
// the place's whole slope variance, resolved and hidden, a Gaussian
// surface's share steeper than a slope s being e^(−s²/spread); how near
// the tops of the resolved waves' crests the point stands (0 to 1), their
// summed height over WATER_CREST_HEIGHTS roots of their summed heights'
// variance; and the place's whole heights' variance, resolved and hidden
// (square meters), as `spread` sums their slopes'.
struct WaterWaves {
    slopes: vec2<f32>,
    hidden: f32,
    spread: f32,
    crest: f32,
    heights: f32,
}

// Waves standing about what stands in the water as a pixel shows them
// (`drawn`), and the variance of all their slopes and of all their heights,
// resolved and hidden.
struct WaterStanding {
    drawn: WaterRead,
    spread: f32,
    heights: f32,
}

// How long the waves running water carries live (seconds) where it runs
// `depth` meters deep, its eddies turning at `eddy` meters a second, and
// shears `shear` a second: as long as its eddies, as large as it is deep
// (at least WATER_BREAK_THINNEST, as `water_eddies` takes them), take to
// turn over, their size over their speed, turning over the waves riding
// them; no longer than the shear takes to strain those waves by
// WATER_STRAIN; and at least WATER_SHORTEST_LIFE.
fn water_wave_life(depth: f32, eddy: f32, shear: f32) -> f32 {
    let turning = max(depth, WATER_BREAK_THINNEST) / max(eddy, 1.0e-6);
    let straining = WATER_STRAIN / max(shear, WATER_LEAST_SHEAR);
    return max(min(turning, straining), WATER_SHORTEST_LIFE);
}

// The lives a carried pattern living `life` seconds takes: the octave of
// WATER_SHORTEST_LIFE at or below it (`first`, seconds, and `index`,
// octaves above the shortest), and how much the octave above it, twice as
// long, shows.
struct WaterBands {
    first: f32,
    index: u32,
    toward: f32,
}

fn water_bands(life: f32) -> WaterBands {
    let octave = log2(max(life, WATER_SHORTEST_LIFE) / WATER_SHORTEST_LIFE);
    let below = floor(octave);
    var bands: WaterBands;
    bands.first = WATER_SHORTEST_LIFE * exp2(below);
    bands.index = u32(below);
    bands.toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    return bands;
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

// Waves `first` and `second` of independent sources together, as
// `WaterRead` sums them: their slopes and heights add, and so do their
// variances.
fn water_waves_added(first: WaterRead, second: WaterRead) -> WaterRead {
    var sum: WaterRead;
    sum.slopes = first.slopes + second.slopes;
    sum.hidden = first.hidden + second.hidden;
    sum.height = first.height + second.height;
    sum.heights = first.heights + second.heights;
    return sum;
}

// Waves `first` blended `toward` waves `second` (`WaterRead`), independent
// fields keeping their spread: each weighed by its share over the root of
// the shares' squares.
fn water_waves_mixed(first: WaterRead, second: WaterRead, toward: f32) -> WaterRead {
    let shares = vec2<f32>(1.0 - toward, toward)
        * inverseSqrt((1.0 - toward) * (1.0 - toward) + toward * toward);
    var mixed: WaterRead;
    mixed.slopes = first.slopes * shares.x + second.slopes * shares.y;
    mixed.hidden = first.hidden * shares.x * shares.x + second.hidden * shares.y * shares.y;
    mixed.height = first.height * shares.x + second.height * shares.y;
    mixed.heights = first.heights * shares.x * shares.x + second.heights * shares.y * shares.y;
    return mixed;
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

// The wind's waves running water carries at `p` (meters along x and z),
// which changes by `dx` and `dy` a pixel, where it flows along `flow`
// (meters a second) shearing `shear` a second, its waves living `life`
// seconds (`water_wave_life`), at `time`, the place's clock running
// `stagger` cycles ahead, each of the wind's tiles as strong as `gains` says
// (`water_wind_waves`): they ride the current (Gonzalez-Ochoa, Uncharted 4's
// rapids), read where the flow carried them from in both sets of both
// octaves of its lives (`water_bands`, `water_set`), faded as the flow
// strains them and blended keeping their spread, Σwᵢxᵢ/√Σwᵢ² of waves whose
// mean is none (Yu, Neyret, Bruneton and Holzschuch). What strain fades of
// their slopes hides in the variance, roughening the water instead.
fn water_carried_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flow: vec2<f32>,
        shear: f32, life: f32, time: f32, stagger: f32, gains: vec4<f32>) -> WaterRead {
    let bands = water_bands(life);
    var sum: WaterRead;
    var weights = 0.0;
    for (var band = 0u; band < 2u; band++) {
        let shown = select(1.0 - bands.toward, bands.toward, band == 1u);
        if shown <= 0.0 {
            continue;
        }
        let cycle = bands.first * f32(1u + band);
        let salt = ((bands.index + band) * 0x632be5abu) ^ 0x3c6ef372u;
        for (var which = 0u; which < 2u; which++) {
            let carried = water_set(time, stagger, cycle, which, salt);
            let weight = shown * carried.weight;
            if weight <= 0.0 {
                continue;
            }
            let at = p - flow * carried.seconds + carried.offset;
            let read = water_wind_waves(at, dx, dy, gains);
            let faded = weight * exp(-WATER_STRAIN_FADE * abs(carried.seconds) * shear);
            sum.slopes += read.slopes * faded;
            sum.hidden += read.hidden * weight * weight
                + (weight * weight - faded * faded) * dot(read.slopes, read.slopes);
            sum.height += read.height * faded;
            sum.heights += read.heights * faded * faded;
            weights += weight * weight;
        }
    }
    let kept = max(weights, 1.0e-6);
    sum.slopes *= inverseSqrt(kept);
    sum.hidden /= kept;
    sum.height *= inverseSqrt(kept);
    sum.heights /= kept;
    return sum;
}

// Standing waves `sum` with a part added whose slopes are `gradient`
// (meters a meter) and height `height` (meters), `amplitude` meters high at
// its tallest there, its features `width` meters across, as a pixel
// spanning `pixel` meters shows it: whole where they span the steady size
// (STEADY_PIXELS pixels), averaging out as they narrow below that, what the
// pixel averages of their slopes hiding in their variance; its heights'
// variance a sine's as tall.
fn water_resolved(sum: WaterStanding, gradient: vec2<f32>, height: f32, amplitude: f32,
        width: f32, pixel: f32) -> WaterStanding {
    let kept = saturate(width / (STEADY_PIXELS * max(pixel, 1.0e-6)));
    let steep = dot(gradient, gradient);
    var added = sum;
    added.drawn.slopes += gradient * kept;
    added.drawn.hidden += steep * (1.0 - kept * kept);
    added.drawn.height += height * kept;
    added.drawn.heights += 0.5 * amplitude * amplitude * kept * kept;
    added.spread += steep;
    added.heights += 0.5 * amplitude * amplitude;
    return added;
}

// The waves standing in water `depth` meters deep flowing along `flow`
// (meters a second) at Froude number `froude`, about the obstacle nearest
// it, `obstacle` (its offset from the point along x and z and its radius,
// meters, and how strongly a drop or jump lies just upstream), the share
// `anchored` of them where a pixel's samples agree on that obstacle, for a
// pixel spanning `pixel` meters (`water_resolved`): fixed where the bed is,
// which the flow never smears. A pillow heaps up just upstream of the
// obstacle, as wide as it and as high as the water's velocity head U²/2g,
// or WATER_PILLOW.z of its radius; in supercritical water, a front trails
// each side at the half angle asin(1/Fr) (30° at Froude 2), a narrow crest
// falling off downstream; in slower water, waves stand in its wake 2πU²/g
// apart, the length of the waves its current holds still; and below a drop
// or jump, where the water runs slower than Froude 1.7 as below an undular
// jump, a train stands of waves about five depths long.
fn water_standing(obstacle: vec4<f32>, flow: vec2<f32>, depth: f32, froude: f32,
        pixel: f32, anchored: f32) -> WaterStanding {
    var standing: WaterStanding;
    let radius = obstacle.z;
    let speed = length(flow);
    if radius < WATER_STANDING_LEAST.x || speed < WATER_STANDING_LEAST.y || anchored <= 0.0 {
        return standing;
    }
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    // Meters downstream of the obstacle's middle, and across the flow.
    let apart = -obstacle.xy;
    let along = dot(apart, heading);
    let across = dot(apart, side);
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let run = WATER_PILLOW.y * radius;
    let ahead = (along + WATER_PILLOW.x * radius) / run;
    let wide = across / radius;
    let peak = min(head, WATER_PILLOW.z * radius) * anchored;
    let pillow = peak * exp(-ahead * ahead - wide * wide);
    standing = water_resolved(standing,
        (heading * (ahead / run) + side * (wide / radius)) * (-2.0 * pillow), pillow, peak, run,
        pixel);
    let supercritical = smoothstep(1.0, 1.3, froude);
    if supercritical > 0.0 {
        let sine = 1.0 / max(froude, 1.0);
        let cosine = sqrt(max(1.0 - sine * sine, 0.0));
        // Meters out from the front's crest, and the way out.
        let beside = (abs(across) - radius) * cosine - along * sine;
        let outward = side * (cosine * select(-1.0, 1.0, across >= 0.0)) - heading * sine;
        let crest = max(WATER_FRONT_CREST.x * depth, WATER_FRONT_CREST.y);
        let trailing = smoothstep(-radius, 0.0, along)
            * exp(-max(along, 0.0) / (WATER_FRONT_REACH * radius));
        let tallest = min(head, depth) * supercritical * trailing * anchored;
        let front = tallest * exp(-(beside / crest) * (beside / crest));
        standing = water_resolved(standing, outward * (-2.0 * front * beside / (crest * crest)),
            front, tallest, crest, pixel);
    }
    let subcritical = 1.0 - smoothstep(0.8, 1.0, froude);
    let past = along - radius;
    if subcritical > 0.0 && past > 0.0 {
        let wavelength = 4.0 * PI * head;
        let wake = radius + WATER_WAKE * past;
        let height = WATER_TRAIN * head * subcritical * smoothstep(0.0, radius, past)
            * exp(-past / (WATER_TRAIN_REACH * wavelength) - (across / wake) * (across / wake))
            * anchored;
        let rate = 2.0 * PI / wavelength;
        standing = water_resolved(standing, heading * (-height * rate * sin(rate * along)),
            height * cos(rate * along), height, 0.25 * wavelength, pixel);
    }
    let undular = obstacle.w * (1.0 - smoothstep(1.4, 1.7, froude));
    if undular > 0.0 {
        let wavelength = WATER_UNDULAR.x * max(depth, WATER_THINNEST);
        let height = WATER_UNDULAR.y * depth * undular
            * exp(-length(apart) / (WATER_TRAIN_REACH * wavelength)) * anchored;
        let rate = 2.0 * PI / wavelength;
        standing = water_resolved(standing, heading * (-height * rate * sin(rate * along)),
            height * cos(rate * along), height, 0.25 * wavelength, pixel);
    }
    return standing;
}

// The waves at `p` (meters along x and z), which changes by `dx` and `dy` a
// pixel, of water as `at` holds it, at Froude number `froude`, its eddies
// turning at `eddy` meters a second (`water_eddies`), `flowing` of the
// way from still to running (0 to 1), the share `anchored` of its nearest
// obstacle's waves a pixel's samples agree on, at `time`, the place's clock
// running `stagger` cycles ahead, each of the wind's tiles as strong as
// `gains` says (`water_wind_gains`), as `WaterWaves` holds them. The boils
// its eddies bring up (`water_boils`) calm the wind's waves first, wherever
// those are read. Still water's are the wind's as they evolve where they
// stand. Running water's ride its current: the wind's
// (`water_carried_waves`) and the boils, as much as the blend shows running
// water; and its eddies roughen it unseen (`water_stirred`). Between, the
// two blend keeping their spread, so nothing changes at once where a river
// meets a lake. Waves its bed holds stand still where it runs faster than
// the slowest of its own waves, by that threshold alone however the blend
// stands (`water_standing_waves`); and waves stand about what it flows past
// (`water_standing`).
fn water_flow_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, at: WaterAt,
        gains: vec4<f32>, eddy: f32, froude: f32, flowing: f32, anchored: f32, time: f32,
        stagger: f32) -> WaterWaves {
    let flow = at.words.flow;
    let depth = at.depth;
    let shear = at.words.state.z;
    // The boils first: within them the surface spreads, calming the wind's
    // waves on it, whether they stand or ride the current, as far as the
    // blend shows running water, where they rise.
    let boils = water_boils(p, dx, dy, flow, depth, eddy, time);
    let calmed = gains * (1.0 - flowing * (1.0 - boils.calm));
    var still: WaterRead;
    if flowing < 1.0 {
        still = water_wind_waves(p, dx, dy, calmed);
    }
    var running: WaterRead;
    if flowing > 0.0 {
        running = water_waves_added(water_carried_waves(p, dx, dy, flow, shear,
            water_wave_life(depth, eddy, shear), time, stagger, calmed), boils.drawn);
    }
    var read = water_waves_mixed(still, running, flowing);
    // The wind's slopes' variance is its calmed gains' squares however the
    // water moves; the boils' and the eddies' show as far as the blend shows
    // running water, the eddies' all hidden.
    let running_share = flowing * flowing
        / ((1.0 - flowing) * (1.0 - flowing) + flowing * flowing);
    let stirred = water_stirred(eddy, depth) * running_share;
    read.hidden += stirred.x;
    var spread = dot(calmed, calmed) + boils.spread * running_share + stirred.x;
    // Their heights' alike: each wind tile's root mean square height at its
    // calmed gain, and the boils' and the eddies' as far as running water
    // shows.
    var heights = boils.heights * running_share + stirred.y;
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let lifted = calmed[tile] * water_tile_height(tile);
        heights += lifted * lifted;
    }
    // The waves its bed holds still, whole: the slowest standing wave's speed
    // is their own threshold.
    let bed = water_standing_waves(p, dx, dy, flow, depth);
    read = water_waves_added(read, bed.drawn);
    spread += bed.spread;
    heights += bed.heights;
    // A pixel of the display (the mip bias), whose steady size their
    // features must span to show whole (`water_resolved`).
    let standing = water_standing(at.obstacle, flow, depth, froude,
        max(length(dx), length(dy)) * material_detail(), anchored);
    read = water_waves_added(read, standing.drawn);
    spread += standing.spread;
    heights += standing.heights;
    var waves: WaterWaves;
    waves.slopes = read.slopes;
    waves.hidden = read.hidden;
    waves.spread = spread;
    waves.heights = heights;
    if read.heights > 1.0e-12 {
        waves.crest = saturate(read.height / (WATER_CREST_HEIGHTS * sqrt(read.heights)));
    }
    return waves;
}
