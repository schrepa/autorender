// Terrains' water's waves (`shade_water`), from its state alone: the
// wind's (`water_spectrum.wgsl`), standing where they evolve on still water
// and carried by the current on running water, in sets living as long as
// its eddies take to turn over; the eddies' own, read from the turbulence
// tile turned to the flow and drawn out along it, one to two depths long and
// as high as the eddies' speed squared over gravity; short waves the bed
// holds still where the water runs near the speed of its own waves; and
// waves standing about what it flows past. Every source is drawn as far as a
// pixel resolves its waves, and what the pixel averages of their slopes
// roughens the surface instead (LEAN mapping), never faded; a source with no
// share, or whose waves a pixel hides whole, reads no texture, its slopes'
// variance known from what raised them.

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
// Turbulent surface waves stand one to two depths long (in depths, from
// the first to the second), as large as the eddies stirring them, and the
// current draws them out into streaks WATER_EDDY_STRETCH times as long along
// it as across it.
const WATER_EDDY_WAVES: vec2<f32> = vec2<f32>(1.0, 2.0);
const WATER_EDDY_STRETCH: f32 = 2.0;
// The steepest slope a wave stands at before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of π/7 at most, 0.32 in
// root mean square).
const WATER_STEEPEST: f32 = 0.44879895;
// Short waves standing on the bed where water runs near or above the speed
// of its own waves: waves running upstream as fast as the current carries
// them down stand still, deep-water waves 2πU²/g long, drawn
// WATER_STAND_STRETCH times as long across the flow as along it so their
// crests lie across it, rising over the Froude numbers WATER_STAND_FROUDE,
// from none to all.
const WATER_STAND_STRETCH: f32 = 2.5;
const WATER_STAND_FROUDE: vec2<f32> = vec2<f32>(0.6, 1.2);
// The share of a rung's period between the corners of the grid the
// turbulence tile's turned reads hang their offsets on
// (`water_turned_tile`): each corner's read shows within half a period of
// its middle, so none of it repeats there.
const WATER_TURN_PATCH: f32 = 0.5;
// How many root mean square heights a sine's crests stand over its mean:
// waves stand at the tops of their crests as high as this over the root of
// their heights' variance.
const WATER_CREST_HEIGHTS: f32 = 1.41421356;
// Standing waves (Neyret and Praizelin; Yu, Neyret and Steed): the least
// radius (meters) of what stands in the water, and the least speed (meters
// a second) of the water flowing past it, that raise any; the pixels a
// standing wave's features must span to show whole, what a pixel averages
// of finer ones roughening the water instead; how far upstream of its
// obstacle's middle a pillow heaps up, how long it runs along the flow and
// how high it rises at most, in radii (the water it heaps up spills round
// the obstacle); a front's crest's half width, in depths and at least
// (meters), and the radii over which fronts fall off downstream; the share
// of the velocity head the waves standing past obstacles in slower water
// rise by, how much their wake widens a meter downstream, and the
// wavelengths over which trains fall off; and the length and height of an
// undular jump's waves, in depths.
const WATER_STANDING_LEAST: vec2<f32> = vec2<f32>(0.0625, 0.2);
const WATER_STANDING_PIXELS: f32 = 2.0;
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
// surface's share steeper than a slope s being e^(−s²/spread); and how near
// the tops of the resolved waves' crests the point stands (0 to 1), their
// summed height over WATER_CREST_HEIGHTS roots of their summed heights'
// variance.
struct WaterWaves {
    slopes: vec2<f32>,
    hidden: f32,
    spread: f32,
    crest: f32,
}

// Waves standing about what stands in the water as a pixel shows them
// (`drawn`), and the variance of all their slopes, resolved and hidden.
struct WaterStanding {
    drawn: WaterRead,
    spread: f32,
}

// Waves of the turbulence tile at two rungs of a ladder of fixed scales an
// octave apart (`water_turned_source`): the shorter's octaves over the
// tile's own scale (`first`), how much each shows, the root mean square of
// each one's slopes as a read unstretched holds them and of its heights
// (meters), how many times as long along the flow and across it they are
// drawn, and the variance of their slopes as drawn.
struct WaterTurnedWaves {
    first: f32,
    shown: vec2<f32>,
    slope: vec2<f32>,
    height: vec2<f32>,
    stretch: vec2<f32>,
    spread: f32,
}

// How fast eddies turn (meters a second) in water running `speed` meters a
// second down a surface falling `slope` meters a meter, `depth` meters deep
// (at least WATER_BREAK_THINNEST): u_t = (g·U·S·h)^(1/3). Running down, it
// loses g·U·S watts a kilogram, which stirs eddies as large as it is deep
// turning at (ε·h)^(1/3) (Kolmogorov), the speed by which the cook breaks it
// white (`engine_terrain::tile_water::breaking`).
fn water_eddy_speed(speed: f32, slope: f32, depth: f32) -> f32 {
    return pow(max(WATER_GRAVITY * speed * slope * max(depth, WATER_BREAK_THINNEST), 0.0),
        1.0 / 3.0);
}

// How long the waves running water carries live (seconds) where its eddies
// turn at `eddy` meters a second and it shears `shear` a second: as long as
// an eddy takes to turn over waves `meters` long, their wavelength over its
// speed, no longer than the shear takes to strain them by WATER_STRAIN, and
// at least WATER_SHORTEST_LIFE.
fn water_wave_life(meters: f32, eddy: f32, shear: f32) -> f32 {
    let turning = meters / max(eddy, 1.0e-6);
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

// The share of their slopes' variance waves of an even spread of directions
// keep, drawn `stretch` times as long along the flow (x) and across it (y):
// each axis's half, over its stretch squared.
fn water_stretched(stretch: vec2<f32>) -> f32 {
    return 0.5 / (stretch.x * stretch.x) + 0.5 / (stretch.y * stretch.y);
}

// The turbulence tile (`WATER_TURBULENCE_TILE`) at rung `rung` of its
// ladder of scales, its waves 2^rung times as long as the tile's own, at
// `p` (meters along x and z), which changes by `dx` and `dy` a pixel, its
// waves running along `heading` and drawn `stretch` times as long along it
// (x) and across it (y): read at the three corners of the triangle of a grid
// holding it, its corners WATER_TURN_PATCH of the rung's period apart, each
// read turned to the heading about its corner's middle and shifted at random
// (salted by `salt`), and blended by their weights keeping their spread, as
// `water_tile` blends its reads (Heitz and Neyret). Turning each read about
// a middle nearby keeps a turning flow from shearing it. Its slopes (xy,
// meters a meter, of unit variance before the stretch draws them out), the
// variance of those the pixel hides (z), and its height in its root mean
// square heights (w).
fn water_turned_tile(rung: f32, p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>,
        heading: vec2<f32>, stretch: vec2<f32>, salt: u32) -> vec4<f32> {
    let tile = WATER_TURBULENCE_TILE;
    let side = vec2<f32>(-heading.y, heading.x);
    // The rung's periods a meter along the heading and across it, as its
    // waves are drawn out; read as finely as the display shows them: the mip
    // bias.
    let repeats = water_tile_repeats(tile) * exp2(-rung);
    let scale = repeats / stretch;
    let detail = material_detail();
    let turned_dx = vec2<f32>(dot(dx, heading), dot(dx, side)) * scale * detail;
    let turned_dy = vec2<f32>(dot(dy, heading), dot(dy, side)) * scale * detail;
    let spacing = WATER_TURN_PATCH / repeats;
    let triangle = water_triangle(p / spacing);
    var corners = triangle.corners;
    let weights = triangle.weights;
    var sum = vec4<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let skewed = vec2<f32>(corners[corner]);
        let middle = vec2<f32>(skewed.x, (skewed.y + 0.57735027 * skewed.x) * 0.8660254)
            * spacing;
        let apart = p - middle;
        let moments = water_tile_sample(tile,
            vec2<f32>(dot(apart, heading), dot(apart, side)) * scale
                + water_corner_offset(corners[corner], salt), turned_dx, turned_dy);
        let weight = weights[corner];
        sum += vec4<f32>(moments.xy * weight, moments.z * weight * weight, moments.w * weight);
    }
    let kept = dot(weights, weights);
    // Slopes back from the read's frame, each axis's shortened as far as
    // its waves are drawn out.
    let slopes = sum.xy * inverseSqrt(kept) / stretch;
    return vec4<f32>(heading * slopes.x + side * slopes.y,
        sum.z / kept * water_stretched(stretch), sum.w * inverseSqrt(kept));
}

// The turbulence tile's waves `meters` long at their middle, `height`
// meters high in root mean square and drawn `stretch` times as long along
// the flow and across it, as `WaterTurnedWaves` holds them: at the two rungs
// of the ladder of the tile's scales, octaves of its own, nearest that
// middle, crossfading over the middle share WATER_BAND_BLEND of each octave,
// so they follow the water smoothly while each rung keeps its scale and
// never stretches where the water changes; each rung as high as `height`,
// its slopes as steep as that height gives over its waves, and no steeper,
// as drawn, than waves stand before they break (WATER_STEEPEST, a sine's
// root mean square of it).
fn water_turned_source(meters: f32, height: f32, stretch: vec2<f32>) -> WaterTurnedWaves {
    let tile = WATER_TURBULENCE_TILE;
    let band = water_tile_band(tile);
    let octave = log2(max(meters, 1.0e-6) / sqrt(band.x * band.y));
    let below = floor(octave);
    let toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    // The meters each rung's heights rise for each unit of its slopes' root
    // mean square read unstretched, and the steepest that root mean square
    // may be for the waves as drawn to stand no steeper than breaking waves.
    let lifts = water_tile_height(tile) * exp2(vec2<f32>(below, below + 1.0));
    let steepest = WATER_STEEPEST * 0.70710678 * inverseSqrt(water_stretched(stretch));
    var waves: WaterTurnedWaves;
    waves.first = below;
    waves.shown = vec2<f32>(1.0 - toward, toward);
    waves.slope = min(vec2<f32>(height) / max(lifts, vec2<f32>(1.0e-9)), vec2<f32>(steepest));
    waves.height = waves.slope * lifts;
    waves.stretch = stretch;
    waves.spread = dot(waves.shown * waves.shown, waves.slope * waves.slope)
        / dot(waves.shown, waves.shown) * water_stretched(stretch);
    return waves;
}

// Waves `waves` (`water_turned_source`'s) at `p` (meters along x and z),
// which changes by `dx` and `dy` a pixel, running along `heading` and
// salted by `salt`, as a pixel reads them (`WaterRead`): each rung read
// turned (`water_turned_tile`), salted by its place on the ladder, so a rung
// shows the same waves however the water around it changes; a rung whose
// longest waves, as drawn, span no more than the pixel's narrower side
// hidden whole, its slopes' variance known without reading it; and the
// rungs blended keeping their spread, their heights' variance whole while
// their read heights average out as the pixel comes to span many waves.
fn water_turned_read(waves: WaterTurnedWaves, p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>,
        heading: vec2<f32>, salt: u32) -> WaterRead {
    // The meters the pixel spans across its narrower side, as finely as the
    // display shows them: the mip bias.
    let narrowest = min(length(dx), length(dy)) * material_detail();
    let longest = water_tile_band(WATER_TURBULENCE_TILE).x
        * max(waves.stretch.x, waves.stretch.y);
    let stretched = water_stretched(waves.stretch);
    var read: WaterRead;
    for (var rung = 0u; rung < 2u; rung++) {
        let shown = waves.shown[rung];
        let slope = waves.slope[rung];
        if shown <= 0.0 || slope <= 0.0 {
            continue;
        }
        let ladder = waves.first + f32(rung);
        if longest * exp2(ladder) <= narrowest {
            read.hidden += shown * shown * slope * slope * stretched;
            continue;
        }
        let turned = water_turned_tile(ladder, p, dx, dy, heading, waves.stretch,
            salt ^ (bitcast<u32>(i32(ladder)) * 0x85ebca6bu));
        let height = waves.height[rung];
        read.slopes += turned.xy * (shown * slope);
        read.hidden += turned.z * shown * shown * slope * slope;
        read.height += turned.w * (shown * height);
        read.heights += shown * shown * height * height;
    }
    let kept = dot(waves.shown, waves.shown);
    read.slopes *= inverseSqrt(kept);
    read.hidden /= kept;
    read.height *= inverseSqrt(kept);
    read.heights /= kept;
    return read;
}

// The waves running water carries at `p` (meters along x and z), which
// changes by `dx` and `dy` a pixel, where it flows along `flow` (meters a
// second, `heading` its way) shearing `shear` a second, its waves living
// `life` seconds (`water_wave_life`), at `time`, the place's clock running
// `stagger` cycles ahead: the wind's, each tile as strong as `gains` says
// (`water_wind_waves`), which ride the current (Gonzalez-Ochoa, Uncharted 4's
// rapids), and its eddies' (`eddies`, `water_turned_read`), read where the
// flow carried them from in both sets of both octaves of its lives
// (`water_bands`, `water_set`), faded as the flow strains them and blended
// keeping their spread, Σwᵢxᵢ/√Σwᵢ² of waves whose mean is none (Yu, Neyret,
// Bruneton and Holzschuch). What strain fades of their slopes hides in the
// variance, roughening the water instead.
fn water_carried_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flow: vec2<f32>,
        heading: vec2<f32>, shear: f32, life: f32, time: f32, stagger: f32,
        gains: vec4<f32>, eddies: WaterTurnedWaves) -> WaterRead {
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
            let read = water_waves_added(water_wind_waves(at, dx, dy, gains),
                water_turned_read(eddies, at, dx, dy, heading, salt));
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
// spanning `pixel` meters shows it: whole where they span
// WATER_STANDING_PIXELS pixels, averaging out as they narrow below that,
// what the pixel averages of their slopes hiding in their variance; its
// heights' variance a sine's as tall.
fn water_resolved(sum: WaterStanding, gradient: vec2<f32>, height: f32, amplitude: f32,
        width: f32, pixel: f32) -> WaterStanding {
    let kept = saturate(width / (WATER_STANDING_PIXELS * max(pixel, 1.0e-6)));
    let steep = dot(gradient, gradient);
    var added = sum;
    added.drawn.slopes += gradient * kept;
    added.drawn.hidden += steep * (1.0 - kept * kept);
    added.drawn.height += height * kept;
    added.drawn.heights += 0.5 * amplitude * amplitude * kept * kept;
    added.spread += steep;
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
// turning at `eddy` meters a second (`water_eddy_speed`), `flowing` of the
// way from still to running (0 to 1), the share `anchored` of its nearest
// obstacle's waves a pixel's samples agree on, at `time`, the place's clock
// running `stagger` cycles ahead, each of the wind's tiles as strong as
// `gains` says (`water_wind_gains`), as `WaterWaves` holds them. Still
// water's are the wind's as they evolve where they stand. Running water's
// ride its current (`water_carried_waves`): the wind's, and those its eddies
// raise, one to two depths long, their heights' root mean square u_t²/g as
// the eddies' dynamic pressure lifts the surface against gravity (Froude
// scaling), so shallow fast riffles run rough and deep slow pools stay
// glassy. Between, the two blend keeping their spread, so nothing changes at
// once where a river meets a lake. Where the water runs near or above the
// speed of its own waves, short waves stand on its bed, fixed where the bed
// is: 2πU²/g long, rising with its Froude number, their heights' root mean
// square the velocity head U²/2g at most and no more than a fifth of the
// depth (an undular jump's waves). And waves stand about what it flows past
// (`water_standing`).
fn water_flow_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, at: WaterAt,
        gains: vec4<f32>, eddy: f32, froude: f32, flowing: f32, anchored: f32, time: f32,
        stagger: f32) -> WaterWaves {
    let flow = at.words.flow;
    let speed = length(flow);
    let heading = select(vec2<f32>(1.0, 0.0), flow / max(speed, 1.0e-6), speed > 1.0e-4);
    let depth = at.depth;
    let shear = at.words.state.z;
    var still: WaterRead;
    if flowing < 1.0 {
        still = water_wind_waves(p, dx, dy, gains);
    }
    var running: WaterRead;
    var stirred = 0.0;
    if flowing > 0.0 {
        let middle = max(depth, WATER_BREAK_THINNEST)
            * sqrt(WATER_EDDY_WAVES.x * WATER_EDDY_WAVES.y);
        let eddies = water_turned_source(middle, eddy * eddy / WATER_GRAVITY,
            vec2<f32>(WATER_EDDY_STRETCH, 1.0));
        running = water_carried_waves(p, dx, dy, flow, heading, shear,
            water_wave_life(middle, eddy, shear), time, stagger, gains, eddies);
        stirred = eddies.spread;
    }
    var read = water_waves_mixed(still, running, flowing);
    // The wind's slopes' variance is its gains' squares however the water
    // moves; the eddies' shows as far as the blend shows running water.
    var spread = dot(gains, gains) + stirred * flowing * flowing
        / ((1.0 - flowing) * (1.0 - flowing) + flowing * flowing);
    let rise = smoothstep(WATER_STAND_FROUDE.x, WATER_STAND_FROUDE.y, froude);
    if rise > 0.0 {
        let head = speed * speed / (2.0 * WATER_GRAVITY);
        let bed = water_turned_source(4.0 * PI * head, rise * min(head, WATER_UNDULAR.y * depth),
            vec2<f32>(1.0, WATER_STAND_STRETCH));
        read = water_waves_added(read, water_turned_read(bed, p, dx, dy, heading, 0x5bd1e995u));
        spread += bed.spread;
    }
    // As finely as the display shows them: the mip bias.
    let standing = water_standing(at.obstacle, flow, depth, froude,
        max(length(dx), length(dy)) * material_detail(), anchored);
    read = water_waves_added(read, standing.drawn);
    spread += standing.spread;
    var waves: WaterWaves;
    waves.slopes = read.slopes;
    waves.hidden = read.hidden;
    waves.spread = spread;
    if read.heights > 1.0e-12 {
        waves.crest = saturate(read.height / (WATER_CREST_HEIGHTS * sqrt(read.heights)));
    }
    return waves;
}
