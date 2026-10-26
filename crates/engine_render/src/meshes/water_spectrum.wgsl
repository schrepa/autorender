// Water's waves as its tiles hold them (`water_waves.rs`): WATER_WIND_TILES
// tiles of the wind's waves of fixed periods, each holding its own band of
// lengths at unit slope variance. The wind's
// tiles' periods never divide one another (each the last's over the cube
// of the golden ratio) and each one's lattice turns by its own angle from
// the world's axes (`water_tile_turn`), so no tile's repeat lines up with
// another's or with the grids of the ground and the water: a pixel reads
// each wind tile once, straight at its point turned so, with no patches. A
// pixel reads a tile at the steady size (`water_steady`): the tile's levels
// and anisotropy whose texels span STEADY_PIXELS pixels average what they
// span, so it draws no waves finer than temporal antialiasing holds
// steadily, the variance of the slopes they average away roughens the
// water (LEAN mapping), and nothing fades by distance. Each place sizes
// the wind's tiles by the sea its wind raises there
// (`engine_core::wind_waves`): Elfouhaily, Chapron, Katsaros and
// Vandemark's unified spectrum (1997, "A unified directional spectrum for
// long and short wind-driven waves") for the wind's speed there, gusts and
// all, over the fetch its water's span gives, cut by its depth (TMA) and
// past the waves viscosity takes faster than the wind raises them, each
// tile scaled by the root of the slope variance that sea holds in the
// tile's band; the slopes of its waves finer than the tiles hold roughen
// the water, and light wind, raising no capillary ripples, leaves it
// glassy.
// Terrain upwind shelters the wind over water: behind a crest it slows,
// recovering downwind (`terrain_shelter`), so valley streams lie calm while
// open lakes keep their breeze.

// The wind's tiles (`gpu_layouts::constants::WIND_TILES`).
const WATER_WIND_TILES: u32 = 4u;
// The most anisotropy the waves' sampler filters to
// (`gpu_layouts::constants::WAVE_ANISOTROPY`): along a pixel's longer side
// it averages at most this many times as finely as across.
const WATER_WAVE_ANISOTROPY: f32 = 8.0;
// Meters across the patches of stronger and calmer wind sweeping over
// water.
const WATER_GUST_SIZE: f32 = 40.0;
// The sea the wind raises over a fetch F (`engine_core::wind_waves`,
// Elfouhaily et al. 1997). Its inverse wave age U/c_p,
// Ω = WATER_RAISED_AGE tanh((X/WATER_AGE_FETCH)^WATER_AGE_FETCH_POWER)
// to the power WATER_AGE_POWER, of the fetch over the length the wind's
// speed sets, X = g F / U²: WATER_RAISED_AGE once fully raised, more for
// younger seas, whose peak runs slower.
const WATER_RAISED_AGE: f32 = 0.84;
const WATER_AGE_FETCH: f32 = 2.2e4;
const WATER_AGE_FETCH_POWER: f32 = 0.4;
const WATER_AGE_POWER: f32 = -0.75;
// Its long waves' strength, α_p = WATER_LONG_STRENGTH √Ω; how much higher it
// stands at its peak, γ = WATER_PEAK_ENHANCEMENT up to Ω = 1 and
// WATER_PEAK_ENHANCEMENT_GROWTH log₁₀ Ω more past it; and its peak's width
// in √(k/k_p), σ = WATER_PEAK_WIDTH (1 + WATER_PEAK_WIDTH_GROWTH Ω⁻³).
const WATER_LONG_STRENGTH: f32 = 6.0e-3;
const WATER_PEAK_ENHANCEMENT: f32 = 1.7;
const WATER_PEAK_ENHANCEMENT_GROWTH: f32 = 6.0;
const WATER_PEAK_WIDTH: f32 = 0.08;
const WATER_PEAK_WIDTH_GROWTH: f32 = 4.0;
// The wave number (radians a meter) at which surface tension speeds waves
// as much as gravity does (`WAVE_CAPILLARY`), and the phase speed (meters a
// second) of the slowest waves, there: k_m and c_m, about which its short
// waves gather; and their strength,
// α_m = WATER_SHORT_STRENGTH (1 + ln(u*/c_m)) for the wind's friction
// velocity u* up to c_m and WATER_SHORT_STRENGTH (1 + WATER_SHORT_GROWTH
// ln(u*/c_m)) past it, none where that is negative: light wind raises no
// capillary ripples.
const WATER_RIPPLE_NUMBER: f32 = 370.0;
const WATER_RIPPLE_SPEED: f32 = 0.23;
const WATER_SHORT_STRENGTH: f32 = 0.01;
const WATER_SHORT_GROWTH: f32 = 3.0;
// The sea's drag on the wind 10 meters up, C_D = WATER_DRAG +
// WATER_DRAG_GROWTH U (Wu 1982), by which the wind's friction velocity is
// u* = √C_D U.
const WATER_DRAG: f32 = 8.0e-4;
const WATER_DRAG_GROWTH: f32 = 6.5e-5;
// How fast the wind feeds a wave's energy, WATER_WIND_GROWTH (u*/c)² ω
// (Plant 1982), and water's kinematic viscosity ν (square meters a second,
// at 20 °C), by which it takes it at 4 ν k² (Lamb): past the angular
// frequency where the two meet, WATER_WIND_GROWTH u*² / (4 ν), the sea
// holds no waves.
const WATER_WIND_GROWTH: f32 = 0.04;
const WATER_VISCOSITY: f32 = 1.0e-6;
// The peak's widths within which its enhancement is drawn: past five, it
// lifts the spectrum by less than a hundred-thousandth.
const WATER_PEAK_REACH: f32 = 5.0;
// Points an octave by which a band of wave numbers sums the sea's slopes
// (the midpoint rule in the logarithm of the wave number): the peak's
// enhancement, a quarter of an octave wide at its narrowest, always meets
// one, so a band's share errs by a few hundredths at most.
const WATER_BAND_POINTS: f32 = 2.0;
// The finest of the wind's waves summed (`water_wind_finer`), in k_m: past
// ten times it, the short waves' spectrum holds less than 2×10⁻⁹ of its
// peak, and viscosity takes a ripple's energy within a few of its periods.
const WATER_FINEST: f32 = 10.0;
// The meters of fetch and depth the open sea's wind blows over: farther
// than any wind needs to raise its sea fully, its inverse wave age reaching
// WATER_RAISED_AGE to f32's precision (g F / U² past about 5×10⁶), as the
// sea's own waves take it over unlimited fetch
// (`engine_core::Ocean::spectrum`), and deeper than the tiles' longest
// waves feel (two thirds of their length).
const WATER_OPEN_FETCH: f32 = 1.0e10;
const WATER_OPEN_DEPTH: f32 = 1.0e4;
// The least and the most roughness water shades with: as smooth as every
// surface may be (`meshes/shading.wgsl`), where GGX's lobe stays finite in
// f32 though rounding takes a unit half vector's cosine with the normal an
// ulp or two past 1; and the most its waves' hidden slopes roughen it.
const WATER_SMOOTHEST: f32 = 0.03;
const WATER_ROUGHEST: f32 = 0.6;

// Waves as a pixel reads them: their slopes (meters a meter), the variance
// of the slopes the pixel hides, their height (meters) and the variance of
// the heights it resolves, summed over sources as independent fields:
// slopes and heights add, and so do variances.
struct WaterRead {
    slopes: vec2<f32>,
    hidden: f32,
    height: f32,
    heights: f32,
}

// How many times a meter tile `tile` repeats.
fn water_tile_repeats(tile: u32) -> f32 {
    return water.waves[tile].x;
}

// The longest and the shortest waves tile `tile` holds (x and y, meters)
// as it repeats `water_tile_repeats` times a meter.
fn water_tile_band(tile: u32) -> vec2<f32> {
    return water.waves[tile].yz;
}

// The root mean square height of tile `tile`'s waves at unit slope
// variance (meters), as it repeats `water_tile_repeats` times a meter.
fn water_tile_height(tile: u32) -> f32 {
    return water.waves[tile].w;
}

// The cosine and sine (x and y) of the angle tile `tile`'s lattice turns by
// from the world's axes, from +x toward +z: water reads the tile at each
// world point turned by it, its product with that unit complex number
// (`complex_product`), and turns its slopes back by its conjugate.
fn water_tile_turn(tile: u32) -> vec2<f32> {
    return water.turns[tile].xy;
}

// A random point of a pattern (in its periods) by which reads at corner
// `corner` of a grid of patches shift it, salted by `salt`, so neighboring
// patches show different parts of it.
fn water_corner_offset(corner: vec2<i32>, salt: u32) -> vec2<f32> {
    let key = (u32(corner.x) * 0x8da6b343u) ^ (u32(corner.y) * 0xd8163841u) ^ (salt * 0xcb1ab31fu);
    return vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u));
}

// The triangle of a grid of equilateral triangles, skewed into squares,
// holding `grid` (in its cells' sides): its three corners (in the skewed
// grid) and the weights `grid` takes of each.
struct WaterTriangle {
    corners: array<vec2<i32>, 3>,
    weights: vec3<f32>,
}

fn water_triangle(grid: vec2<f32>) -> WaterTriangle {
    let skewed = vec2<f32>(grid.x, 1.15470054 * grid.y - 0.57735027 * grid.x);
    let cell = floor(skewed);
    let local = skewed - cell;
    let base = vec2<i32>(cell);
    let third = 1.0 - local.x - local.y;
    var triangle: WaterTriangle;
    triangle.weights = vec3<f32>(third, local.y, local.x);
    triangle.corners = array<vec2<i32>, 3>(base, base + vec2<i32>(0, 1), base + vec2<i32>(1, 0));
    if third <= 0.0 {
        triangle.weights = vec3<f32>(-third, 1.0 - local.y, 1.0 - local.x);
        triangle.corners = array<vec2<i32>, 3>(base + vec2<i32>(1, 1), base + vec2<i32>(1, 0),
            base + vec2<i32>(0, 1));
    }
    return triangle;
}

// How many times a pixel's derivatives scale for water's patterns to be
// filtered at the steady size: STEADY_PIXELS times the render size's share
// of the display (`material_detail`, the mip bias), so the levels and
// anisotropy they read average all they hold finer than that many pixels of
// the display.
fn water_steady() -> f32 {
    return STEADY_PIXELS * material_detail();
}

// One filtered read of tile `tile` at `uv` (in its periods), which changes
// by `dx` and `dy` across the steady size (in its periods: a pixel's
// derivatives scaled by `water_steady`): what its levels and anisotropy
// average of its moments over that span. Its slopes (xy, for unit slope
// variance), the variance of those the span averages away (z, what the
// levels lose of the slopes' square) and its height in its own root mean
// square heights (w, averaging toward none as the span covers many waves).
// Where its texels span more than the steady size, its levels' averages
// hide nothing of it: what filtering between texels loses of the slopes'
// square (lines along the texels' edges) is no waves' variance, so it fades
// out as texels grow past it.
fn water_tile_sample(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let moments = textureSampleGrad(water_waves, water_wave_sampler, uv, tile, dx, dy);
    let texels = sqrt(length(dx) * length(dy)) * f32(textureDimensions(water_waves).x);
    let minified = saturate(log2(max(texels, 1.0e-6)) + 1.0);
    return vec4<f32>(moments.xy, max(moments.z - dot(moments.xy, moments.xy), 0.0) * minified,
        moments.w);
}

// The meters a pixel spans as the waves' sampler filters what it covers,
// the pixel changing by `dx` and `dy` (meters): its shorter side, or its
// longer side over WATER_WAVE_ANISOTROPY where that is more, by which the
// sampler picks the level it reads.
fn water_wave_span(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let first = length(dx);
    let second = length(dy);
    return max(min(first, second), max(first, second) / WATER_WAVE_ANISOTROPY);
}

// The wind's speed (meters a second) over water at `world` at `time`, its
// gusts included and the terrain upwind sheltering it: patches of stronger
// and calmer wind, WATER_GUST_SIZE meters across and drawn out along it,
// sweep over the water with it (cat's paws), varying its speed by as much as
// it gusts either way (`water.wind.z`, `Wind::gusts`), as gusts vary the
// push placed objects sway by; and behind upwind terrain it slows, gusts and
// all, recovering downwind (`terrain_shelter`).
fn water_wind_speed(world: vec3<f32>, time: f32) -> f32 {
    let wind = water.wind.xy;
    let speed = length(wind);
    let heading = select(vec2<f32>(1.0, 0.0), wind / speed, speed > 1.0e-3);
    let moving = world.xz - wind * time;
    let streaked = vec2<f32>(dot(moving, heading) / 2.5,
        dot(moving, vec2<f32>(-heading.y, heading.x))) / WATER_GUST_SIZE;
    let patches = water_noise(streaked).x * 0.65
        + water_noise(streaked * 2.3 + vec2<f32>(5.2, 1.3)).x * 0.35;
    let gusting = smoothstep(0.2, 0.8, patches);
    return speed * (1.0 + water.wind.z * (2.0 * gusting - 1.0)) * terrain_shelter(world, heading);
}

// The sea the wind raises over a stretch of water (`water_wind_sea`), as
// `water_wind_curvature` reads it: the inverse root of its peak's wave
// number (1/√k_p), its long and its short waves' strengths times their
// phase speeds (α_p c_p / 2 and α_m c_m / 2, meters a second), how fast its
// long waves fall away past its peak (Ω/√10), the binary logarithm of its
// peak's enhancement (log₂ γ) and its peak's width (σ), the natural
// logarithm of the wave number of its viscous end (radians a meter), past
// which it holds no waves, and the root of the water's depth. A sea of no
// long waves holds none.
struct WaterSea {
    peak_root: f32,
    long: f32,
    short: f32,
    fall: f32,
    enhancement: f32,
    width: f32,
    viscous: f32,
    deep: f32,
}

// The sea wind of `speed` meters a second (as it reaches the water, gusting
// and sheltered by the terrain upwind: `water_wind_speed`) raises over
// `fetch` meters of water `depth` meters deep (`engine_core::wind_waves`):
// its inverse wave age Ω by the fetch over the length the wind's speed
// sets, its peak k_p = g Ω²/U², which runs at c_p = U/Ω, its short waves'
// strength by the wind's friction velocity u*, and its viscous end, the wave
// number whose angular frequency ω_v = WATER_WIND_GROWTH u*² / (4 ν) solves
// ω² = g k (1 + (k/k_m)²): k = (k_m/√3)(w − 1/w), w = ∛(z + √(z² + 1)),
// z = (3√3/2) ω_v² / (g k_m) (Cardano's root). Past it viscosity takes a
// wave's energy faster than the wind feeds it, and the sea holds none
// (Donelan and Pierson 1987), so light wind leaves water still. A fetch
// shorter than the shortest waves the tiles hold raises none of them;
// before the tiles arrive, they hold none.
fn water_wind_sea(speed: f32, fetch: f32, depth: f32) -> WaterSea {
    var sea: WaterSea;
    let shortest = water_tile_band(WATER_WIND_TILES - 1u).y;
    if speed <= 0.0 || shortest <= 0.0 {
        return sea;
    }
    // The fetch over the length the wind's speed sets (g F / U²), and the
    // sea's inverse wave age, its tanh by the exponential, which no fetch
    // overflows.
    let reach = WATER_GRAVITY * max(fetch, shortest) / (speed * speed);
    let grown = exp(-2.0 * pow(reach / WATER_AGE_FETCH, WATER_AGE_FETCH_POWER));
    let age = WATER_RAISED_AGE * pow((1.0 - grown) / (1.0 + grown), WATER_AGE_POWER);
    sea.peak_root = speed / (age * sqrt(WATER_GRAVITY));
    sea.long = 0.5 * WATER_LONG_STRENGTH * sqrt(age) * speed / age;
    sea.fall = age / sqrt(10.0);
    sea.enhancement = log2(WATER_PEAK_ENHANCEMENT
        + WATER_PEAK_ENHANCEMENT_GROWTH * max(log(age) / log(10.0), 0.0));
    sea.width = WATER_PEAK_WIDTH * (1.0 + WATER_PEAK_WIDTH_GROWTH / (age * age * age));
    // The wind's friction velocity, and the short waves' strength by its
    // ratio to the slowest waves' speed.
    let friction = sqrt(WATER_DRAG + WATER_DRAG_GROWTH * speed) * speed;
    let rise = log(friction / WATER_RIPPLE_SPEED);
    let strength = WATER_SHORT_STRENGTH
        * (1.0 + select(1.0, WATER_SHORT_GROWTH, rise > 0.0) * rise);
    sea.short = 0.5 * max(strength, 0.0) * WATER_RIPPLE_SPEED;
    // The viscous end's angular frequency, and its wave number.
    let frequency = WATER_WIND_GROWTH * friction * friction / (4.0 * WATER_VISCOSITY);
    let cubic = 1.5 * sqrt(3.0) * frequency * frequency
        / (WATER_GRAVITY * WATER_RIPPLE_NUMBER);
    let turned = pow(cubic + sqrt(cubic * cubic + 1.0), 1.0 / 3.0);
    sea.viscous = log(max(WATER_RIPPLE_NUMBER / sqrt(3.0) * (turned - 1.0 / turned),
        1.0e-30));
    sea.deep = sqrt(max(depth, 0.0));
    return sea;
}

// The curvature of sea `sea` at the wave number k whose root is `root` (√k,
// k radians a meter): the variance of its slopes by the logarithm of the
// wave number, its waves of every heading counted, B = B_l + B_h
// (Elfouhaily et al. 1997). Its long waves' B_l = (α_p/2)(c_p/c) L_PM J_p
// e^(−(Ω/√10)(√(k/k_p) − 1)) and its short waves'
// B_h = (α_m/2)(c_m/c) L_PM J_p e^(−(k/k_m − 1)²/4), where
// c = √((g/k)(1 + (k/k_m)²)) is the waves' phase speed,
// L_PM = e^(−5/4 (k_p/k)²) the fully raised sea's shape, and J_p = γ^Γ,
// Γ = e^(−(√(k/k_p) − 1)² / 2σ²), its peak's enhancement, drawn within
// WATER_PEAK_REACH widths of the peak. Water too shallow to hold the longer
// waves cuts them by Kitaigorodskii's factor of √(k h), ω √(h/g) in deep
// water's waves (TMA).
fn water_wind_curvature(sea: WaterSea, root: f32) -> f32 {
    let number = root * root;
    // √(k/k_p), ω/ω_p in deep water, and k_p/k, by which the long and the
    // short waves both fade below the peak.
    let over = root * sea.peak_root;
    let ratio = 1.0 / (over * over);
    let fade = -1.25 * ratio * ratio;
    var enhanced = 1.0;
    let off = (over - 1.0) / sea.width;
    if off * off < WATER_PEAK_REACH * WATER_PEAK_REACH {
        enhanced = exp2(sea.enhancement * exp(-0.5 * off * off));
    }
    // k/k_m, and the waves' slowness 1/c (seconds a meter).
    let scaled = number / WATER_RIPPLE_NUMBER;
    let slowness = root * inverseSqrt(WATER_GRAVITY * (1.0 + scaled * scaled));
    let waves = sea.long * exp(fade - sea.fall * (over - 1.0))
        + sea.short * exp(fade - 0.25 * (scaled - 1.0) * (scaled - 1.0));
    let shallow = min(root * sea.deep, 2.0);
    let held = select(1.0 - 0.5 * (2.0 - shallow) * (2.0 - shallow), 0.5 * shallow * shallow,
        shallow <= 1.0);
    return waves * enhanced * slowness * held;
}

// The slope variance sea `sea` holds over `points` points `step` apart in
// the logarithm of the wave number, the first at `first`, by the midpoint
// rule: each point counted as far as its step lies past `counted` (the
// logarithm of the wave number of the longest waves counted) and short of
// the sea's viscous end, so the sum changes smoothly with both.
fn water_wind_sum(sea: WaterSea, first: f32, step: f32, points: f32, counted: f32) -> f32 {
    // The root of each point's wave number, growing by `rising` a point.
    let rising = exp(0.5 * step);
    var root = exp(0.5 * first);
    var sum = 0.0;
    for (var point = 0.0; point < points; point += 1.0) {
        let middle = first + point * step;
        let share = saturate((middle - counted) / step + 0.5)
            * saturate((sea.viscous - middle) / step + 0.5);
        if share > 0.0 {
            sum += water_wind_curvature(sea, root) * share;
        }
        root *= rising;
    }
    return sum * step;
}

// For each wind tile, the root of the slope variance its band holds of the
// waves no longer than `longest` meters in sea `sea` (`water_wind_sea`),
// summed over the band from the middle of its first step, nearest its
// longest waves, WATER_BAND_POINTS points an octave (`water_wind_sum`): what
// the tile's slopes scale by there, 0 where it holds none.
fn water_wind_bands(sea: WaterSea, longest: f32) -> vec4<f32> {
    var gains = vec4<f32>(0.0);
    if sea.long <= 0.0 {
        return gains;
    }
    // The logarithm of the wave number of the longest waves counted.
    let counted = log(2.0 * PI / longest);
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let band = water_tile_band(tile);
        if band.y <= 0.0 || band.x <= band.y {
            continue;
        }
        let points = ceil(WATER_BAND_POINTS * log2(band.x / band.y));
        let step = log(band.x / band.y) / points;
        gains[tile] = sqrt(water_wind_sum(sea, log(2.0 * PI / band.x) + 0.5 * step, step,
            points, counted));
    }
    return gains;
}

// For each wind tile, the root of the slope variance its band holds in sea
// `sea` (`water_wind_sea`), every wave the tiles hold counted
// (`water_wind_bands`): what the tile's slopes scale by there, 0 where it
// holds none.
fn water_wind_gains(sea: WaterSea) -> vec4<f32> {
    return water_wind_bands(sea, water_tile_band(0u).x);
}

// The slope variance of the waves of sea `sea` finer than the wind's tiles
// hold, from the last tile's shortest waves to WATER_FINEST k_m, summed
// WATER_BAND_POINTS points an octave (`water_wind_sum`) as far as the sea's
// viscous end: where light wind's falls short of them, it raises none, and
// the water lies glassy.
fn water_wind_finer(sea: WaterSea) -> f32 {
    if sea.long <= 0.0 {
        return 0.0;
    }
    let start = log(2.0 * PI / water_tile_band(WATER_WIND_TILES - 1u).y);
    let span = log(WATER_FINEST * WATER_RIPPLE_NUMBER) - start;
    if span <= 0.0 {
        return 0.0;
    }
    let points = ceil(WATER_BAND_POINTS * span / log(2.0));
    let step = span / points;
    return water_wind_sum(sea, start + 0.5 * step, step, points, start);
}

// The wind's waves at world `p` (meters along x and z), which changes by
// `dx` and `dy` a pixel (meters): each wind tile read once, straight, at `p`
// turned by its lattice's angle (`water_tile_turn`) times its repeats, at
// the steady size (`water_tile_sample`, its derivatives turned alike and
// scaled by `water_steady`, so it draws no waves finer than its levels'
// texels spanning STEADY_PIXELS pixels), its slopes turned back into the
// world (a wave the tile holds along the wind's heading turned by that
// angle so runs along the wind), and scaled by its gain in `gains`
// (`water_wind_gains`). Slopes and heights add, the hidden variances as
// each gain's square times its read's, and of the heights' variance each
// tile's waves longer than two of those texels hold theirs (its heights'
// variance an octave falling as the square of their length). A tile of no
// gain reads nothing. One whose slopes stay as gentle as the smoothest
// water's microfacets (WATER_SMOOTHEST's square), or whose longest waves
// the steady size spans twice over (as the sampler picks its level,
// `water_wave_span`), reads no texture and hides the square of its gain:
// each hands over across an octave, from twice the gentlest to it and from
// once the longest waves to twice, what it takes of the slopes hiding in
// the variance, so nothing pops.
fn water_wind_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, gains: vec4<f32>) -> WaterRead {
    var read = WaterRead(vec2<f32>(0.0), 0.0, 0.0, 0.0);
    // At the steady size: the meters it spans as the sampler filters.
    let steady = water_steady();
    let span = water_wave_span(dx, dy) * steady;
    let calm = WATER_SMOOTHEST * WATER_SMOOTHEST;
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let gain = gains[tile];
        if gain <= 0.0 {
            continue;
        }
        let band = water_tile_band(tile);
        // How much of its slopes it draws, the rest hiding in its variance.
        let kept = saturate(2.0 - span / band.x) * saturate(gain / calm - 1.0);
        var moments = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        if kept > 0.0 {
            let repeats = water_tile_repeats(tile);
            let turn = water_tile_turn(tile);
            let sampled = water_tile_sample(tile, complex_product(p, turn) * repeats,
                complex_product(dx, turn) * (repeats * steady),
                complex_product(dy, turn) * (repeats * steady));
            // Its slopes turned back into the world.
            let slopes = complex_product(sampled.xy, vec2<f32>(turn.x, -turn.y));
            moments = vec4<f32>(slopes * kept,
                sampled.z + (1.0 - kept * kept) * dot(slopes, slopes), sampled.w * kept);
        }
        // The share of its heights' variance its waves longer than two of
        // the texels it reads hold.
        let cut = clamp(2.0 * span, band.y, band.x);
        let resolved = (band.x * band.x - cut * cut)
            / max(band.x * band.x - band.y * band.y, 1.0e-12);
        let height = water_tile_height(tile) * gain;
        read.slopes += moments.xy * gain;
        read.hidden += moments.z * gain * gain;
        read.height += moments.w * height;
        read.heights += height * height * kept * kept * resolved;
    }
    return read;
}

// The variance water's shading normal `normal` adds to its slopes' square
// by turning across a pixel (in `water_roughness`'s units): what the waves
// it draws vary within the pixel, whose glints its samples would otherwise
// catch in one frame and miss in the next (geometric specular antialiasing,
// Kaplanyan, Hill, Patney and Lefohn 2016: twice the pixel footprint's
// variance SPECULAR_VARIANCE times the normal's squared change across and
// down the pixel, clamped to SPECULAR_LIMIT before it joins the slopes' as
// Tokuyoshi and Kaplanyan 2019 clamp it), every surface's kernel
// (`widened`). Its derivatives need it called in uniform control flow.
fn water_turning(normal: vec3<f32>) -> f32 {
    let across = dpdx(normal);
    let down = dpdy(normal);
    return min(2.0 * SPECULAR_VARIANCE * (dot(across, across) + dot(down, down)),
        SPECULAR_LIMIT);
}

// The roughness of water whose waves vary `variance` of their slopes'
// square within a pixel, those it hides and those it draws turning across
// it (`water_turning`), in sea `sea` (`water_wind_sea`): the variance of
// the sea's waves finer than the tiles hold (`water_wind_finer`) adds to
// it, as variances add in a microfacet distribution (LEAN mapping, Olano
// and Baker), and the roughness is the sum's fourth root, from
// WATER_SMOOTHEST to WATER_ROUGHEST.
fn water_roughness(variance: f32, sea: WaterSea) -> f32 {
    return clamp(sqrt(sqrt(variance + water_wind_finer(sea))), WATER_SMOOTHEST,
        WATER_ROUGHEST);
}
