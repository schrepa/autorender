// Water's waves as its tiles hold them (`water_waves.rs`): WATER_WIND_TILES
// tiles of the wind's waves of fixed periods, each holding its own band of
// lengths at unit slope variance. The wind's
// tiles' periods never divide one another (each the last's over the cube
// of the golden ratio) and each one's lattice turns by its own angle from
// the world's axes (`water_tile_turn`), so no tile's repeat lines up with
// another's or with the grids of the ground and the water: a pixel reads
// each wind tile once, straight at its point turned so, with no patches. A
// pixel reads a tile through its own footprint (`water_footprint`), as every
// texture reads: the tile's levels and their sixteenfold anisotropy filter
// it by the pixel's derivatives, resolving it across the view by its short
// side or a sixteenth of its long side (`detail_resolved`) and averaging it
// along its long side, so it draws every wave the screen shows however far
// along the view it lies. The tiles hold the moments of their slopes and
// heights (LEAN mapping, Olano and Baker 2010), so what filtering averages
// of them gives the mean of the slopes over the footprint and the whole
// covariance of those it averages away, which roughens the water along and
// across the view as it projects there (Bruneton, Neyret and Holzschuch
// 2010), the more along the view the more the footprint stretches, so far
// water streaks and glints along the view; nothing fades by distance.
// Each place sizes the wind's tiles by the sea its wind raises there
// (`engine_core::wind_waves`): Elfouhaily, Chapron, Katsaros and
// Vandemark's unified spectrum (1997, "A unified directional spectrum for
// long and short wind-driven waves") for the wind's speed there, gusts and
// all, over the fetch its water's span gives, cut by its depth (TMA) and
// past the waves viscosity takes faster than the wind raises them, each
// tile scaled by the root of the slope variance that sea holds in the
// tile's band; the slopes of its waves finer than the tiles hold roughen
// the water, and light wind, raising no capillary ripples, leaves it
// glassy. Terrains' water shows the share of that sea its settings ask for
// (`water_wind_shown`); the sea, all of its own.
// Terrain upwind shelters the wind over water: behind a crest it slows,
// recovering downwind (`terrain_shelter`), so valley streams lie calm while
// open lakes keep their breeze.

// The wind's tiles (`gpu_layouts::constants::WIND_TILES`).
const WATER_WIND_TILES: u32 = 4u;
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
// The least and the most roughness water shades with, each way: as smooth
// as every surface may be (`meshes/shading.wgsl`), where GGX's lobe stays
// finite in f32 though rounding takes a unit half vector's cosine with the
// normal an ulp or two past 1; and the most its waves' hidden slopes
// roughen it.
const WATER_SMOOTHEST: f32 = 0.03;
const WATER_ROUGHEST: f32 = 0.6;

// Waves as a pixel reads them: their slopes (meters a meter), the
// covariance of the slopes the pixel hides (`water_spread`), their height
// (meters) and the variance of the heights it resolves, summed over sources
// as independent fields: slopes and heights add, and so do variances.
struct WaterRead {
    slopes: vec2<f32>,
    hidden: vec3<f32>,
    height: f32,
    heights: f32,
}

// The covariance of slopes (xx, zz and xz, along x and z) that slopes
// `slopes` (meters a meter) hold as one sample of their spread: their outer
// product, whose trace is their square.
fn water_spread(slopes: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(slopes.x * slopes.x, slopes.y * slopes.y, slopes.x * slopes.y);
}

// The covariance of slopes whose variance `variance` spreads alike every
// way: half of it along each axis.
fn water_spread_even(variance: f32) -> vec3<f32> {
    return vec3<f32>(0.5 * variance, 0.5 * variance, 0.0);
}

// The variance along the unit vector `way` (along x and z) of slopes of
// covariance `spread`.
fn water_spread_toward(spread: vec3<f32>, way: vec2<f32>) -> f32 {
    return spread.x * way.x * way.x + spread.y * way.y * way.y + 2.0 * spread.z * way.x * way.y;
}

// The covariance of slopes of covariance `spread` turned by the unit complex
// number `turn`, as `complex_product` turns the slopes themselves: its mean
// variance stays, and its anisotropic part, half the difference of its
// variances along x and z beside its covariance, turns by twice the angle.
fn water_spread_turned(spread: vec3<f32>, turn: vec2<f32>) -> vec3<f32> {
    let mean = 0.5 * (spread.x + spread.y);
    let anisotropic = complex_product(vec2<f32>(0.5 * (spread.x - spread.y), spread.z),
        complex_product(turn, turn));
    return vec3<f32>(mean + anisotropic.x, mean - anisotropic.x, anisotropic.y);
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

// The covariance (`water_spread`) of tile `tile`'s slopes at unit slope
// variance over its whole repeat, in its own axes, as the last of its
// levels holds it: half of that variance along each axis, drawn out and
// turned by the anisotropic part the wind's spread gives its spectrum
// (`water.turns`' zw: half the difference of its variances along its x and
// z, and its covariance, `water_waves.rs`).
fn water_tile_spread(tile: u32) -> vec3<f32> {
    let anisotropic = water.turns[tile].zw;
    return vec3<f32>(0.5 + anisotropic.x, 0.5 - anisotropic.x, anisotropic.y);
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

// A pixel's footprint as water's patterns filter it: how its point changes
// across a pixel of the display (`dx` and `dy`, its derivatives scaled by
// the render size's share of the display, `material_detail`, as material
// textures' mip bias samples them: meters along x and z, or a pattern's own
// units), by which its texture reads filter and its computed patterns draw
// (`water_footprint_toward`, `water_shown`); the meters its short side and
// its long side span (`across` and `along`, the singular values of those
// derivatives); and the meters it resolves (`detail_resolved` of its sides:
// its short side, or its long side over DETAIL_ANISOTROPY where it is more
// stretched, as the waves' sampler's levels and anisotropy resolve it), by
// which features varying every way draw, as a read of their texture would
// resolve them.
struct WaterFootprint {
    dx: vec2<f32>,
    dy: vec2<f32>,
    across: f32,
    along: f32,
    resolved: f32,
}

// The footprint of a pixel whose point changes by `dx` and `dy` a pixel.
fn water_footprint(dx: vec2<f32>, dy: vec2<f32>) -> WaterFootprint {
    let detail = material_detail();
    var footprint: WaterFootprint;
    footprint.dx = dx * detail;
    footprint.dy = dy * detail;
    let squares = dot(footprint.dx, footprint.dx) + dot(footprint.dy, footprint.dy);
    let area = abs(footprint.dx.x * footprint.dy.y - footprint.dx.y * footprint.dy.x);
    let sum = sqrt(squares + 2.0 * area);
    let difference = sqrt(max(squares - 2.0 * area, 0.0));
    footprint.across = 0.5 * (sum - difference);
    footprint.along = 0.5 * (sum + difference);
    footprint.resolved = detail_resolved(vec2<f32>(footprint.across, footprint.along));
    return footprint;
}

// The meters footprint `footprint` (`water_footprint`) spans along the unit
// vector `way` (along x and z, or a pattern's own axes): how far a pattern
// varying that way moves across the pixel, the length of the pixel's
// derivatives projected on it, from the footprint's short side, where `way`
// runs across the view, to its long side.
fn water_footprint_toward(footprint: WaterFootprint, way: vec2<f32>) -> f32 {
    return length(vec2<f32>(dot(footprint.dx, way), dot(footprint.dy, way)));
}

// How much of a computed pattern's part `meters` long along the way it
// varies a pixel draws, its footprint spanning `span` meters that way
// (`water_footprint_toward`; the size it resolves for parts varying every
// way): all of it while it spans two pixels or more, none where it spans
// one, so nothing finer than the Nyquist limit draws, as mipmaps keep a
// texture's; what it leaves of the part's slopes hides in their variance.
fn water_shown(meters: f32, span: f32) -> f32 {
    return saturate(meters / max(span, 1.0e-9) - 1.0);
}

// One filtered read of tile `tile` at `uv` (in its periods), which changes
// by `dx` and `dy` across a pixel (in its periods: a footprint's
// derivatives, `water_footprint`), at unit slope variance and in its own
// axes: what its levels and anisotropy average of its moments over that
// footprint (LEAN mapping). The mean of its slopes; the covariance of those
// the footprint averages away (`water_spread`), what the filtering loses of
// their second moments; its height in its own root mean square heights,
// averaging toward none as the footprint covers many waves; and the
// variance of the heights it draws, the tile's whole, one, less what the
// filtering loses of the height's square. Rounding leaves no variance
// negative, nor the slopes' covariance past what their variances allow.
// Where its texels span more than the size the footprint resolves
// (`detail_resolved`, by which the sampler picks its level), its levels'
// averages hide nothing of it: what filtering between texels loses of the
// squares (lines along the texels' edges) is no waves' variance, so it
// fades out as texels grow past it.
fn water_tile_sample(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> WaterRead {
    let slopes = textureSampleGrad(water_waves, water_wave_sampler, uv, tile, dx, dy);
    let heights = textureSampleGrad(water_waves, water_wave_sampler, uv,
        tile + WATER_WIND_TILES, dx, dy);
    let texels = detail_resolved(vec2<f32>(length(dx), length(dy)))
        * f32(textureDimensions(water_waves).x);
    let minified = saturate(log2(max(texels, 1.0e-6)) + 1.0);
    let lost = (vec3<f32>(slopes.zw, heights.x) - water_spread(slopes.xy)) * minified;
    let variances = max(lost.xy, vec2<f32>(0.0));
    let bound = sqrt(variances.x * variances.y);
    var read: WaterRead;
    read.slopes = slopes.xy;
    read.hidden = vec3<f32>(variances, clamp(lost.z, -bound, bound));
    read.height = heights.y;
    read.heights = max(1.0 - max(heights.z - heights.y * heights.y, 0.0) * minified, 0.0);
    return read;
}

// One filtered read of wind tile `tile`'s whitecaps at `uv` (in its
// periods), which changes by `dx` and `dy` across a pixel (in its periods),
// as `water_tile_sample` reads its moments: what its levels and anisotropy
// average over that footprint of the largest compression its texels have
// had at unit gain, fading (x, `water_waves.wgsl`), and the variance of it
// they average away (y, what they lose of its square).
// Where its texels span more than the size the footprint resolves
// (`detail_resolved`), what filtering between them loses of the square is
// no variance of it, so that fades out as they grow past it.
fn water_tile_whitecaps(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec2<f32> {
    let state = textureSampleGrad(water_whitecaps, water_wave_sampler, uv, tile, dx, dy).xy;
    let texels = detail_resolved(vec2<f32>(length(dx), length(dy)))
        * f32(textureDimensions(water_whitecaps).x);
    let minified = saturate(log2(max(texels, 1.0e-6)) + 1.0); // filter: the whitecaps' levels averaging what the footprint spans, the variance between texels none where they outgrow it
    return vec2<f32>(state.x, max(state.y - state.x * state.x, 0.0) * minified);
}

// The wind's whitecaps at world `p` (meters along x and z) for a pixel whose
// footprint is `footprint` (`water_footprint`): each wind tile read where
// `water_wind_waves` reads its slopes, at `p` turned by its lattice's angle
// times its repeats, through that footprint (`water_tile_whitecaps`, its
// derivatives turned alike), and scaled by its gain in `gains`
// (`water_wind_gains`). The sum over the tiles of each one's gain times its
// faded compression (x), past 1 where the surface has folded, and the
// variance of that sum the reads average away (y), the tiles independent. A
// tile of no gain reads nothing; every other reads at any size, even where
// `water_wind_waves` reads none of its slopes, so far water reads the mean
// of what near water holds.
fn water_wind_whitecaps(p: vec2<f32>, footprint: WaterFootprint, gains: vec4<f32>)
        -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let gain = gains[tile];
        if gain <= 0.0 {
            continue;
        }
        let repeats = water_tile_repeats(tile);
        let turn = water_tile_turn(tile);
        let read = water_tile_whitecaps(tile, complex_product(p, turn) * repeats,
            complex_product(footprint.dx, turn) * repeats,
            complex_product(footprint.dy, turn) * repeats);
        sum += read * vec2<f32>(gain, gain * gain);
    }
    return sum;
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

// Sea `sea` (`water_wind_sea`) as terrains' water shows it: the share of its
// waves the water's settings ask for (`water.wind.w`, `Water::wind_waves`),
// its curvature scaled by that share's square, so each of the wind's tiles'
// gains scales by the share (`water_wind_gains`), and with them the slopes
// and heights the tiles give the water and how far they fold it
// (`water_wind_whitecaps`), and the variance of the slopes they hide and of
// its waves finer than the tiles hold (`water_wind_finer`) by its square. A
// share of none leaves no waves in it, and the water reads no tile.
fn water_wind_shown(sea: WaterSea) -> WaterSea {
    let kept = water.wind.w * water.wind.w;
    var shown = sea;
    shown.long *= kept;
    shown.short *= kept;
    return shown;
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

// The wind's waves at world `p` (meters along x and z) for a pixel whose
// footprint is `footprint` (`water_footprint`): each wind tile read once,
// straight, at `p` turned by its lattice's angle (`water_tile_turn`) times
// its repeats, through that footprint (`water_tile_sample`, its derivatives
// turned alike), so its levels and anisotropy average all it holds that
// the footprint spans and give the covariance of the slopes they average
// away; its slopes and that covariance turned back into the world (a wave
// the tile holds along the wind's heading turned by that angle so runs
// along the wind, `water_spread_turned`), and scaled by its gain in `gains`
// (`water_wind_gains`). Slopes and heights add, the hidden covariances as
// each gain's square times its read's, what it leaves undrawn of the slopes
// it reads lying along them; and so do the variances of the heights each
// read draws. A tile of no gain reads nothing. One whose slopes stay as
// gentle as the smoothest water's microfacets (WATER_SMOOTHEST's square),
// or whose longest waves the size the footprint resolves spans twice over,
// every wave it holds shorter than half a pixel, reads no texture and hides
// the square of its gain as its whole covariance, which the last of its
// levels holds (`water_tile_spread`): each hands over across an octave,
// from twice the gentlest to it and from once the longest waves to twice,
// what it takes of the slopes hiding in the variance, so nothing pops.
fn water_wind_waves(p: vec2<f32>, footprint: WaterFootprint, gains: vec4<f32>) -> WaterRead {
    var read = WaterRead(vec2<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
    let calm = WATER_SMOOTHEST * WATER_SMOOTHEST;
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let gain = gains[tile];
        if gain <= 0.0 {
            continue;
        }
        let band = water_tile_band(tile);
        let turn = water_tile_turn(tile);
        // How much of its slopes it draws, the rest hiding in its variance.
        let kept = saturate(2.0 - footprint.resolved / band.x) * saturate(gain / calm - 1.0);
        // What it draws at unit gain, in its own axes: none of its slopes
        // or its height where it reads no texture, all its covariance
        // hidden.
        var drawn = WaterRead(vec2<f32>(0.0), water_tile_spread(tile), 0.0, 0.0);
        if kept > 0.0 {
            let repeats = water_tile_repeats(tile);
            let sampled = water_tile_sample(tile, complex_product(p, turn) * repeats,
                complex_product(footprint.dx, turn) * repeats,
                complex_product(footprint.dy, turn) * repeats);
            drawn.slopes = sampled.slopes * kept;
            drawn.hidden = sampled.hidden + (1.0 - kept * kept) * water_spread(sampled.slopes);
            drawn.height = sampled.height * kept;
            drawn.heights = sampled.heights * kept * kept;
        }
        // Turned back into the world by its turn's conjugate.
        let back = vec2<f32>(turn.x, -turn.y);
        let height = water_tile_height(tile) * gain;
        read.slopes += complex_product(drawn.slopes, back) * gain;
        read.hidden += water_spread_turned(drawn.hidden, back) * (gain * gain);
        read.height += drawn.height * height;
        read.heights += drawn.heights * (height * height);
    }
    return read;
}

// The covariance (`water_spread`) water's drawn slopes `slopes` (meters a
// meter) add to those it hides by turning across a pixel: what the waves it
// draws vary within the pixel, whose glints its samples would otherwise
// catch in one frame and miss in the next (geometric specular antialiasing,
// Kaplanyan, Hill, Patney and Lefohn 2016: twice the pixel footprint's
// variance SPECULAR_VARIANCE times the outer products of the slopes' change
// across and down the pixel, so a pixel drawn out along the view widens the
// lobe along the view; scaled to at most SPECULAR_LIMIT of their square
// before it joins the slopes' as Tokuyoshi and Kaplanyan 2019 clamp it),
// every surface's kernel (`widened`). Its derivatives need it called in
// uniform control flow.
fn water_turning(slopes: vec2<f32>) -> vec3<f32> {
    let turning = 2.0 * SPECULAR_VARIANCE
        * (water_spread(dpdx(slopes)) + water_spread(dpdy(slopes)));
    return turning * min(SPECULAR_LIMIT / max(turning.x + turning.y, 1.0e-12), 1.0);
}

// The roughness along and across the view (x and y) of water whose waves'
// slopes vary within a pixel by the covariance `hidden` (`water_spread`:
// those it hides and those it draws turning across it, `water_turning`),
// seen heading along `heading` (a unit vector along x and z: the view laid
// on the water, away from the eye), in sea `sea` (`water_wind_sea`): the
// variance of the sea's waves finer than the tiles hold
// (`water_wind_finer`) adds to it alike every way, as variances add in a
// microfacet distribution (LEAN mapping, Olano and Baker); the ellipse of
// the sum projected into the view, its variance along the heading and
// across it (Bruneton, Neyret and Holzschuch 2010), sets each way's GGX
// alpha as the root of twice that variance, as a round lobe's square alpha
// is its slopes' whole variance; and each way's roughness is its alpha's
// root, from WATER_SMOOTHEST to WATER_ROUGHEST.
fn water_roughness(hidden: vec3<f32>, sea: WaterSea, heading: vec2<f32>) -> vec2<f32> {
    let spread = hidden + water_spread_even(water_wind_finer(sea));
    let variances = vec2<f32>(water_spread_toward(spread, heading),
        water_spread_toward(spread, vec2<f32>(-heading.y, heading.x)));
    return clamp(sqrt(sqrt(2.0 * max(variances, vec2<f32>(0.0)))), vec2<f32>(WATER_SMOOTHEST),
        vec2<f32>(WATER_ROUGHEST));
}
