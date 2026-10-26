// Running water's own waves, which `water_flow.wgsl` adds to the wind's:
// short waves its bed holds standing (`water_standing_waves`), and boils,
// the upwellings its bed's turbulence brings up, riding its current
// (`water_boils`). Both are sparse convolution noise (Lagae, Lefebvre,
// Drettakis and Dutré 2009, "Procedural noise using sparse Gabor
// convolution"): kernels placed by hash on grids fixed to the world, so
// nothing tiles, patches or repeats, their cells on a ladder of sizes an
// octave apart, each place blending the two rungs nearest its own scale
// keeping their spread, so the pattern never stretches where the water
// changes; each kernel's parameters taken at the shading point (Lagae's
// rule for slowly varying ones) and its shape known in closed form, so a
// pixel turns what it cannot draw at the steady size (STEADY_PIXELS) into
// its exact variance instead of reading it. It is all arithmetic: no
// texture and no binding.

// The meters a second over which water goes from still to running, from the
// first to the second (`shade_water`'s `flowing`): boils rise only in
// running water.
const WATER_FLOWING: vec2<f32> = vec2<f32>(0.05, 0.3);
// Water's surface tension over its density, σ/ρ (cubic meters a second
// squared): gravity over the square of the wave number at which surface
// tension speeds its waves as much as gravity does (WATER_RIPPLE_NUMBER,
// `engine_core::wind_waves::RIPPLE_NUMBER`, the waves' tiles' own). Its
// waves run slowest there, 1.7 cm long, at
// c_min = (4 g σ/ρ)^(1/4) = √(2 g / WATER_RIPPLE_NUMBER), 0.23 m/s.
const WATER_TENSION: f32 = WATER_GRAVITY / (WATER_RIPPLE_NUMBER * WATER_RIPPLE_NUMBER);
// The k h past which tanh(k h) rounds to 1 in single precision, where waves
// feel no bottom: clamped there, no device's exponentials overflow.
const WATER_TANH_ROUNDED: f32 = 9.0;
// The Newton steps finding a wave number from deep water's
// (`water_wave_number`): four bring the longest waves standing, at shallow
// water's limit, within a percent, and those running nearly as slowly as
// c_min within a few.
const WATER_WAVE_STEPS: u32 = 4u;

// Waves the bed holds standing (Dolcetti, Horoshenkov, Krynkin and Tait
// 2016, a rough-bed flume 4 to 10 cm deep running 0.19 to 0.6 m/s): where
// water runs faster than c_min, waves running upstream as fast as it carries
// them down stand still over its bed, rising from none at c_min to whole at
// twice it, their heights' root mean square WATER_BED_HEAD of the velocity
// head U²/2g, 0.05 F² of the depth: the flume's 0.5 to 2% of its depths
// (0.25 to 2 mm). Slower water keeps only the tiny patterns its current
// carries.
const WATER_BED_HEAD: f32 = 0.1;
// The k h of the longest waves dispersion holds apart: past twenty depths
// long (shallow water's limit, h/L under 1/20) every wave runs at √(g h), so
// no length stands apart from the rest. Water running faster than those
// waves stands its waves only obliquely; the fronts it stands about what it
// flows past are the longer ones (`water_standing`).
const WATER_BED_SHALLOW: f32 = PI / 10.0;
// The Gabor kernels a cell holds, and the standard deviations of their
// envelopes across a cell, where each is cut: about 4π of them, a dozen,
// reach any point, enough that their sum, of independent headings and
// phases, is about Gaussian (the central limit theorem); and cut where its
// envelope has fallen to a hundredth (e^(−9/2)), every kernel reaching a
// point lies in the 3×3 cells about it.
const WATER_BED_KERNELS: u32 = 4u;
const WATER_BED_ENVELOPE: f32 = 3.0;
// The points of the midpoint rule by which the kernels' moments are summed
// over their headings: their heights' squares (cos⁴θ) and their slopes'
// (cos⁴θ k²), all and as far as a pixel draws them. Their slopes stand
// about alike in every heading, so few points sum them closely.
const WATER_BED_POINTS: u32 = 4u;

// Boils, the upwellings the bed's turbulent bursts bring to the surface
// (Nimmo Smith, Thorpe and Graham 1999): round, WATER_BOIL_ACROSS depths
// across (0.9 ± 0.2), covering WATER_BOIL_COVER of the surface at any time
// (20 to 30%), and smooth: inside one the surface spreads, smoothing away
// the short wind ripples riding it, WATER_BOIL_CALM of their slopes at its
// middle as it stands tallest.
const WATER_BOIL_ACROSS: f32 = 0.9;
const WATER_BOIL_COVER: f32 = 0.25;
const WATER_BOIL_CALM: f32 = 0.8;
// The depths of travel a boil lives, its period scaling with depth over
// speed as the bed's bursts raising it do (Jackson 1976). Its slot recurs
// every life but holds a boil in only WATER_BOIL_HELD of them, so boils pass
// over a place every two or three depths of travel (its cell's width over
// that share), as the bed bursts every 1.5 to 3 depths over the surface
// speed (Nakagawa and Nezu 1978).
const WATER_BOIL_LIFE: f32 = 5.0;
// A boil's means over its life, its radius growing from half to whole over
// the first half and its height rising and falling as sin(π age / life):
// its area's, as a share of its whole one (19/24); and over the area it
// covers, its height's and its height's square's, as shares of its tallest.
const WATER_BOIL_AREA: f32 = 19.0 / 24.0;
const WATER_BOIL_RISE: f32 = (5.0 / (4.0 * PI) + 2.0 / (PI * PI) - 2.0 / (PI * PI * PI))
    / WATER_BOIL_AREA;
const WATER_BOIL_RISE_SQUARED: f32 = (19.0 / 48.0 + 3.0 / (8.0 * PI * PI)) / WATER_BOIL_AREA;
// The share of its lives a slot holds a boil: as many as cover
// WATER_BOIL_COVER of the surface, each slot's cell as wide as its boil.
const WATER_BOIL_HELD: f32 = 4.0 * WATER_BOIL_COVER / (PI * WATER_BOIL_AREA);
// The most columns of cells along the current's longer axis the search for
// the boils over a point reads, each two to four cells across it: a life
// carries a boil 5/0.9 cells (WATER_BOIL_LIFE over WATER_BOIL_ACROSS), the
// octave of lives above at most twice as far, and its radius reaches half a
// cell past either end.
const WATER_BOIL_COLUMNS: u32 = 14u;

// What a source of running water's waves draws at a pixel, and all it holds.
struct WaterStream {
    // What the pixel shows (water_spectrum.wgsl's WaterRead).
    drawn: WaterRead,
    // The source's whole slope variance, resolved and hidden.
    spread: f32,
    // The source's whole heights' variance (square meters).
    heights: f32,
    // What the wind's waves' slopes scale by here: 1 leaves them be.
    calm: f32,
}

// The meters a pixel changing by `dx` and `dy` (meters along x and z) spans
// across its footprint's narrowest way and its widest (x and y): the
// footprint's axes, its derivatives' singular values.
fn water_stream_footprint(dx: vec2<f32>, dy: vec2<f32>) -> vec2<f32> {
    let squares = dot(dx, dx) + dot(dy, dy);
    let area = abs(dx.x * dy.y - dx.y * dy.x);
    let sum = sqrt(squares + 2.0 * area);
    let difference = sqrt(max(squares - 2.0 * area, 0.0));
    return 0.5 * vec2<f32>(sum - difference, sum + difference);
}

// How much of waves `meters` long running along `way` (a unit vector) a
// pixel draws, its footprint changing by `dx` and `dy` across the steady
// size (a pixel's derivatives scaled by `water_steady`): none where they
// span no more than that footprint does along their way, so finer waves
// never alias, all where they span twice it, handing over across that
// octave as the wind's tiles do; what it leaves hides in the variance.
fn water_stream_shown(meters: f32, way: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let span = length(vec2<f32>(dot(dx, way), dot(dy, way)));
    return saturate(meters / max(span, 1.0e-9) - 1.0); // filter: waves no longer than the steady size along their way, their slopes' variance as roughness
}

// The wave number (radians a meter) of the waves running at `speed` meters
// a second on water `depth` meters deep, gravity and surface tension both
// restoring them: the longer root of their phase speed squared,
// (g/k + (σ/ρ) k) tanh(k h) = speed² (ω² = (g k + (σ/ρ) k³) tanh(k h)),
// the gravity one, past which the phase speed grows again; found by
// WATER_WAVE_STEPS Newton steps from deep water's g / speed², none
// lengthening the waves more than fourfold.
fn water_wave_number(speed: f32, depth: f32) -> f32 {
    let squared = speed * speed;
    var number = WATER_GRAVITY / squared;
    for (var refined = 0u; refined < WATER_WAVE_STEPS; refined++) {
        let bottom = tanh(min(number * depth, WATER_TANH_ROUNDED));
        let deep = WATER_GRAVITY / number + WATER_TENSION * number;
        let change = (WATER_TENSION - WATER_GRAVITY / (number * number)) * bottom
            + deep * depth * (1.0 - bottom * bottom);
        number = max(number - (deep * bottom - squared) / min(change, -1.0e-12), 0.25 * number);
    }
    return number;
}

// The waves of one rung of the bed's standing waves at `p` (meters along x
// and z), its cells `size` meters across and hashed by `salt`, each holding
// WATER_BED_KERNELS Gabor kernels at hashed places, read in the 3×3 cells
// about the point; in water `depth` meters deep flowing along `flow`
// (meters a second), its waves standing at phase speeds from `top` down by
// `reach` meters a second; each kernel `amplitude` meters high at its middle
// times its heading's share of the velocity head; for a pixel whose
// footprint changes by `dx` and `dy` across the steady size. Each kernel
// heads at θ from straight upstream, its phase speed U cos θ drawn below
// `top` by `reach` times a hashed share's square, its side and its phase
// hashed too; its wave number is the point's own for that speed
// (`water_wave_number`); its envelope is Gaussian, a cell spanning
// WATER_BED_ENVELOPE of its standard deviations, cut there; and it draws as
// far as its waves span the steady size along its way (`water_stream_shown`).
// Their slopes (xy, meters a meter: the kernels' gradients) and height (z,
// meters), summed.
fn water_bed_kernels(p: vec2<f32>, size: f32, salt: u32, flow: vec2<f32>, top: f32,
        reach: f32, depth: f32, amplitude: f32, dx: vec2<f32>, dy: vec2<f32>) -> vec3<f32> {
    let speed = length(flow);
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    let envelope = size / WATER_BED_ENVELOPE;
    let narrowing = 1.0 / (envelope * envelope);
    // The cell holding the point, and where in it the point lies.
    let grid = p / size;
    let base = floor(grid);
    let within = grid - base;
    let corner = vec2<i32>(base);
    var sum = vec3<f32>(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let cell = corner + vec2<i32>(i, j);
            let key = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u) ^ salt;
            for (var kernel = 0u; kernel < WATER_BED_KERNELS; kernel++) {
                let mixed = key ^ ((kernel + 1u) * 0x9e3779b9u);
                // Meters from the kernel's middle to the point: none it
                // reaches lies a cell away.
                let apart = (within - vec2<f32>(f32(i), f32(j)) - terrain_hash2(mixed)) * size;
                let squared = dot(apart, apart);
                if squared >= size * size {
                    continue;
                }
                let drawn = terrain_hash2(mixed ^ 0x68e31da4u);
                let lean = 2.0 * drawn.x - 1.0;
                let standing = top - reach * lean * lean;
                let along = standing / speed;
                let way = side * (sign(lean) * sqrt(max(1.0 - along * along, 0.0)))
                    - heading * along;
                let number = water_wave_number(standing, depth);
                let shown = water_stream_shown(2.0 * PI / number, way, dx, dy);
                if shown <= 0.0 {
                    continue;
                }
                let phase = number * dot(apart, way) + 2.0 * PI * drawn.y;
                let wave = vec2<f32>(cos(phase), sin(phase));
                let lifted = amplitude * along * along * shown
                    * exp(-0.5 * squared * narrowing);
                sum += vec3<f32>(-lifted * (apart * (wave.x * narrowing) + way * (number * wave.y)),
                    lifted * wave.x);
            }
        }
    }
    return sum;
}

// The waves the bed holds standing at `p` (meters along x and z), which
// changes by `dx` and `dy` a pixel, in water `depth` meters deep (at least
// WATER_THINNEST, as Froude numbers read it) flowing along `flow` (meters a
// second), as `WaterStream` holds them: fixed to the bed, so they never
// change with time, their slopes and height the kernels' own gradients and
// values. A wave heading at θ from straight upstream stands still over the
// bed where its phase speed is U cos θ: the longer, gravity root
// (`water_wave_number`); the shorter, capillary one is millimetric and
// counts for nothing, as viscosity damps waves that short within a few of
// their lengths (2νk² a second). They stand from straight upstream, or in
// water running faster than the longest waves dispersion holds apart
// (WATER_BED_SHALLOW), from the heading whose U cos θ is their speed, out to
// θ_max, where U cos θ_max = c_min: each kernel's U cos θ is drawn from the
// fastest standing down to c_min, nearer the fastest by a hashed share's
// square, so in slower water their headings thin away from straight
// upstream as cos(θ/2) and their crests lie mostly across the flow. Each
// kernel is as high as its heading's share of the velocity head (cos²θ), so
// in deep water each is as steep as the next, and their sum's root mean
// square is WATER_BED_HEAD of the velocity head, risen from none at c_min to
// whole at twice it and no steeper than breaking waves (WATER_STEEPEST, a
// sine's root mean square of it), by their density and energy (Campbell's
// theorem: the heights' variance is the kernels' density, WATER_BED_KERNELS
// over a cell's area, times each one's energy, A² πσ² / 2). The cells are
// as wide as the longest waves standing, on the ladder of 2^rung meters,
// each place blending the two rungs nearest (WATER_BAND_BLEND) keeping their
// spread (`water_waves_mixed`). A kernel draws as far as its waves span the
// steady size along its way (`water_stream_shown`); what it leaves hides in
// the variance, which a Gabor field holds exactly: its heights' variance
// times k², along each heading's way, plus its envelope's 1/σ², alike every
// way, summed over the headings by the midpoint rule (WATER_BED_POINTS). A
// rung whose longest waves span no more than the steady size across the
// pixel's narrowest way reads no kernel, all its slopes' variance hidden.
fn water_standing_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flow: vec2<f32>,
        depth: f32) -> WaterStream {
    var stream: WaterStream;
    stream.calm = 1.0;
    let speed = length(flow);
    // The slowest speed waves stand at, c_min, and the fastest: that of the
    // longest waves dispersion holds apart.
    let least = sqrt(2.0 * WATER_GRAVITY / WATER_RIPPLE_NUMBER);
    let deep = max(depth, WATER_THINNEST);
    let shallowest = WATER_BED_SHALLOW / deep;
    let fastest = sqrt((WATER_GRAVITY / shallowest + WATER_TENSION * shallowest)
        * tanh(WATER_BED_SHALLOW));
    let top = min(speed, fastest);
    if top <= least {
        return stream;
    }
    let heading = flow / speed;
    let side = vec2<f32>(-heading.y, heading.x);
    let reach = top - least;
    let tallest = smoothstep(least, 2.0 * least, speed) * WATER_BED_HEAD * speed * speed
        / (2.0 * WATER_GRAVITY);
    // The wave number of the longest waves standing.
    var longest = shallowest;
    if speed < fastest {
        longest = water_wave_number(speed, deep);
    }
    // The pixel's derivatives across the steady size (`water_steady`), and
    // the meters its footprint spans its narrowest way there.
    let steady = water_steady();
    let steady_dx = dx * steady;
    let steady_dy = dy * steady;
    let narrowest = water_stream_footprint(dx, dy).x * steady;
    // The kernels' moments over their headings, either side of straight
    // upstream alike: their heights' squares and their slopes' (but their
    // envelopes'), all and as far as the pixel draws them; and the
    // covariance of the slopes it leaves undrawn (`water_spread`), each
    // heading's along its way.
    var moments = vec4<f32>(0.0);
    var undrawn = vec3<f32>(0.0);
    for (var point = 0u; point < WATER_BED_POINTS; point++) {
        let drawn = (f32(point) + 0.5) / f32(WATER_BED_POINTS);
        let standing = top - reach * drawn * drawn;
        let number = water_wave_number(standing, deep);
        let along = standing / speed;
        let across = side * sqrt(max(1.0 - along * along, 0.0));
        let meters = 2.0 * PI / number;
        let first = water_stream_shown(meters, across - heading * along, steady_dx, steady_dy);
        let second = water_stream_shown(meters, -across - heading * along, steady_dx, steady_dy);
        let shown = 0.5 * (first * first + second * second);
        let share = along * along * along * along;
        let steep = share * number * number;
        moments += vec4<f32>(share, steep, share * shown, steep * shown);
        undrawn += (0.5 * steep) * ((1.0 - first * first) * water_spread(across - heading * along)
            + (1.0 - second * second) * water_spread(-across - heading * along));
    }
    // The two rungs nearest the longest waves' length.
    let octave = log2(2.0 * PI / longest);
    let below = floor(octave);
    let toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    var reads: array<WaterRead, 2>;
    var spreads = vec2<f32>(0.0);
    var heights = vec2<f32>(0.0);
    for (var rung = 0u; rung < 2u; rung++) {
        if select(1.0 - toward, toward, rung == 1u) <= 0.0 {
            continue;
        }
        let ladder = below + f32(rung);
        let size = exp2(ladder);
        let envelope = size / WATER_BED_ENVELOPE;
        let narrowing = 1.0 / (envelope * envelope);
        // Their slopes' variance for each of their heights', and their
        // heights' root mean square, no steeper than breaking waves.
        let slopes = moments.y + moments.x * narrowing;
        let lift = slopes / moments.x;
        let tall = min(tallest, WATER_STEEPEST * sqrt(0.5 / lift));
        spreads[rung] = tall * tall * lift;
        heights[rung] = tall * tall;
        var read: WaterRead;
        read.hidden = (spreads[rung] / slopes)
            * (undrawn + water_spread_even(narrowing * (moments.x - moments.z)));
        read.heights = heights[rung] * moments.z / moments.x;
        if 2.0 * PI / longest > narrowest { // filter: a rung no wave of which spans the steady size, its slopes' variance as roughness
            let amplitude = tall * WATER_BED_ENVELOPE * sqrt(2.0 * f32(WATER_BED_POINTS)
                / (PI * f32(WATER_BED_KERNELS) * moments.x));
            let salt = (bitcast<u32>(i32(ladder)) * 0x85ebca6bu) ^ 0x5bd1e995u;
            let drawn = water_bed_kernels(p, size, salt, flow, top, reach, deep, amplitude,
                steady_dx, steady_dy);
            read.slopes = drawn.xy;
            read.height = drawn.z;
        }
        reads[rung] = read;
    }
    stream.drawn = water_waves_mixed(reads[0], reads[1], toward);
    let parts = vec2<f32>(1.0 - toward, toward);
    let kept = parts * parts / dot(parts, parts);
    stream.spread = dot(kept, spreads);
    stream.heights = dot(kept, heights);
    return stream;
}

// What the boils of one rung and one octave of lives draw at a point
// (`water_boils_read`): their slopes (meters a meter) and height (meters),
// summed, and what they leave of the wind's waves' slopes, each one's calm
// multiplied (1 where none is).
struct WaterBoiled {
    slopes: vec2<f32>,
    height: f32,
    calm: f32,
}

// The boils over `p` (meters along x and z) of the rung whose cells are
// `size` meters across, as wide as its boils, each cell a slot whose lives
// last `life` seconds, hashed by `salt`, in water flowing along `flow`
// (meters a second), each boil `height` meters high at its tallest, at
// `time` (seconds), as `WaterBoiled` holds them. Each slot's lives start at
// its own hashed phase, so the cover holds steady and nothing pulses
// together; a life holds a boil as often as WATER_BOIL_HELD says, at a
// hashed place in its cell, anew each life. A boil born at x_b stands at
// x_b + u age, u being the flow at the point (Lagrangian: each pixel carries
// it by its own current, so it follows bends, and the current's shear draws
// it out as real boils are drawn out); it grows from half its radius to all
// of it over the first half of its life and rises and falls as
// sin(π age / life), a dome A (1 − (r/R)²)², where the surface spreads: the
// wind's ripples' slopes scale by 1 − WATER_BOIL_CALM × the dome's share at
// the point, (1 − (r/R)²)², × its rise, sin(π age / life). The cells
// searched are those whose boils can stand over the point, within a boil's
// radius of the way back along the current over a life, from p − u life to
// p: column by column along the current's longer axis, at most
// WATER_BOIL_COLUMNS of them, two to four cells across each.
fn water_boils_read(p: vec2<f32>, flow: vec2<f32>, size: f32, life: f32, height: f32,
        time: f32, salt: u32) -> WaterBoiled {
    var boiled = WaterBoiled(vec2<f32>(0.0), 0.0, 1.0);
    // In cells: the point, and how far a life carries a boil.
    let here = p / size;
    let travel = flow * (life / size);
    // Along the current's longer axis (x) and across it (y): the way back
    // over a life, how fast it climbs across the columns, and how far either
    // side of it a boil's radius reaches across a column.
    let swapped = abs(travel.y) > abs(travel.x);
    let point = select(here, here.yx, swapped);
    let carried = select(travel, travel.yx, swapped);
    let lowest = min(point.x - carried.x, point.x);
    let highest = max(point.x - carried.x, point.x);
    let rise = select(0.0, carried.y / carried.x, carried.x != 0.0);
    let thickness = 0.5 * sqrt(1.0 + rise * rise);
    let first = i32(floor(lowest - 0.5));
    let last = min(i32(floor(highest + 0.5)), first + i32(WATER_BOIL_COLUMNS) - 1);
    for (var column = first; column <= last; column++) {
        let enter = point.y + (clamp(f32(column), lowest, highest) - point.x) * rise;
        let leave = point.y + (clamp(f32(column + 1), lowest, highest) - point.x) * rise;
        let bottom = i32(floor(min(enter, leave) - thickness));
        let top = min(i32(floor(max(enter, leave) + thickness)), bottom + 3);
        for (var row = bottom; row <= top; row++) {
            let cell = select(vec2<i32>(column, row), vec2<i32>(row, column), swapped);
            let key = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u) ^ salt;
            // Its slot's lives, its own phase staggering them, and whether
            // this one holds a boil.
            let lives = time / life + terrain_hash_of(key ^ 0x41c64e6du);
            let lived = floor(lives);
            let born = key ^ (bitcast<u32>(i32(lived)) * 0x9e3779b9u);
            if terrain_hash_of(born ^ 0x2c1b3c6du) >= WATER_BOIL_HELD {
                continue;
            }
            // How far through its life it is, and where it stands now (cells
            // from the point), carried from where in its cell it was born.
            let age = lives - lived;
            let apart = here - vec2<f32>(cell) - terrain_hash2(born) - travel * age;
            let radius = 0.5 * (0.5 + min(age, 0.5));
            let squared = dot(apart, apart) / (radius * radius);
            if squared >= 1.0 {
                continue;
            }
            let rising = sin(PI * age);
            let inside = 1.0 - squared;
            let dome = inside * inside;
            boiled.height += height * rising * dome;
            boiled.slopes -= apart * (4.0 * height * rising * inside / (radius * radius * size));
            boiled.calm *= 1.0 - WATER_BOIL_CALM * dome * rising;
        }
    }
    return boiled;
}

// The boils over `p` (meters along x and z), which changes by `dx` and `dy`
// a pixel, in water `depth` meters deep flowing along `flow` (meters a
// second), running (faster than WATER_FLOWING's first) and turning eddies at
// `eddy` meters a second (`water_eddies`), at `time` (seconds), as
// `WaterStream` holds them: each boil as high as the head of water upwelling
// at the eddies' speed, w²/2g (Bernoulli), and as wide as WATER_BOIL_ACROSS
// depths, on the ladder of cells 2^rung meters across, each place blending
// the two rungs nearest its boils' width (WATER_BAND_BLEND) keeping their
// spread. A rung's slots live WATER_BOIL_LIFE depths of travel of water as
// deep as its boils are wide, in whole octaves of WATER_SHORTEST_LIFE
// (`water_bands`), the two nearest blended keeping their spread, so every
// pixel a boil covers counts its lives alike (`water_boils_read`). A rung
// whose boils span no more than the steady size across the pixel's widest
// way (`water_steady`), or live no longer than WATER_SHORTEST_LIFE, over
// which their coming and going reads as flicker, reads none, handing over
// across an octave: what it leaves hides its slopes' variance, alike every
// way about its round boils, and its calm averages over its cover,
// 1 − cover × WATER_BOIL_CALM × the dome's mean share (a third) × its
// rise's mean over the area it covers (WATER_BOIL_RISE). Their variances
// over space and their lives are in closed form: their slopes', (2/3) cover
// A² / (R² WATER_BOIL_AREA), and their heights' about the surface they rise
// from, cover A² WATER_BOIL_RISE_SQUARED / 5.
fn water_boils(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, flow: vec2<f32>, depth: f32,
        eddy: f32, time: f32) -> WaterStream {
    var stream: WaterStream;
    stream.calm = 1.0;
    let speed = length(flow);
    if speed <= WATER_FLOWING.x || eddy <= 0.0 {
        return stream;
    }
    let height = eddy * eddy / (2.0 * WATER_GRAVITY);
    let calmed = 1.0 - WATER_BOIL_COVER * WATER_BOIL_CALM * WATER_BOIL_RISE / 3.0;
    // The meters the pixel's footprint spans its widest way at the steady
    // size.
    let widest = water_stream_footprint(dx, dy).y * water_steady();
    // The two rungs nearest the boils' width.
    let octave = log2(WATER_BOIL_ACROSS * max(depth, WATER_THINNEST));
    let below = floor(octave);
    let toward = smoothstep(WATER_BAND_BLEND.x, WATER_BAND_BLEND.y, octave - below);
    var reads: array<WaterRead, 2>;
    var calms = vec2<f32>(1.0);
    var spreads = vec2<f32>(0.0);
    var heights = vec2<f32>(0.0);
    for (var rung = 0u; rung < 2u; rung++) {
        if select(1.0 - toward, toward, rung == 1u) <= 0.0 {
            continue;
        }
        let ladder = below + f32(rung);
        let size = exp2(ladder);
        spreads[rung] = 8.0 * WATER_BOIL_COVER * height * height
            / (3.0 * WATER_BOIL_AREA * size * size);
        heights[rung] = 0.2 * WATER_BOIL_COVER * WATER_BOIL_RISE_SQUARED * height * height;
        calms[rung] = calmed;
        // How long its slots live, and how much of its boils the pixel draws.
        let life = WATER_BOIL_LIFE * size / (WATER_BOIL_ACROSS * speed);
        let resolved = saturate(size / max(widest, 1.0e-9) - 1.0); // filter: boils no wider than the steady size, their slopes' variance as roughness and their calm as its mean
        let shown = resolved * saturate(life / WATER_SHORTEST_LIFE - 1.0);
        var read: WaterRead;
        read.hidden = water_spread_even((1.0 - shown * shown) * spreads[rung]);
        read.heights = shown * shown * heights[rung];
        if shown > 0.0 {
            let bands = water_bands(life);
            let parts = vec2<f32>(1.0 - bands.toward, bands.toward);
            let shares = parts * inverseSqrt(dot(parts, parts));
            let salt = (bitcast<u32>(i32(ladder)) * 0x85ebca6bu) ^ 0xb5297a4du;
            var drawn = WaterBoiled(vec2<f32>(0.0), 0.0, 0.0);
            for (var band = 0u; band < 2u; band++) {
                if parts[band] <= 0.0 {
                    continue;
                }
                let boiled = water_boils_read(p, flow, size, bands.first * f32(1u + band),
                    height, time, salt ^ ((bands.index + band) * 0x632be5abu));
                drawn.slopes += boiled.slopes * shares[band];
                drawn.height += boiled.height * shares[band];
                drawn.calm += boiled.calm * parts[band];
            }
            read.slopes = drawn.slopes * shown;
            read.height = drawn.height * shown;
            calms[rung] = mix(calmed, drawn.calm, shown);
        }
        reads[rung] = read;
    }
    stream.drawn = water_waves_mixed(reads[0], reads[1], toward);
    let parts = vec2<f32>(1.0 - toward, toward);
    let kept = parts * parts / dot(parts, parts);
    stream.spread = dot(kept, spreads);
    stream.heights = dot(kept, heights);
    stream.calm = mix(calms.x, calms.y, toward);
    return stream;
}
