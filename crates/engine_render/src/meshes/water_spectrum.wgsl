// Water's waves as its tiles hold them (`water_waves.rs`): WATER_WIND_TILES
// tiles of the wind's waves of fixed periods, each holding its own band of
// lengths, then the tile of the current's turbulence
// (WATER_TURBULENCE_TILE), every tile at unit slope variance. A pixel reads
// a tile at the steady size (`water_steady`): the tile's levels and
// anisotropy whose texels span STEADY_PIXELS pixels average what they span,
// so it draws no waves finer than temporal antialiasing holds steadily, the
// variance of the slopes they average away roughens the water (LEAN
// mapping), and nothing fades by distance; three reads at random offsets,
// blended keeping their variance, keep it from repeating (tiling and
// blending, Heitz and Neyret). Each place sizes
// the wind's tiles by the sea its wind raises there (Horvath 2015,
// "Empirical directional wave spectra for computer graphics",
// `engine_core::wind_waves`): JONSWAP's spectrum for the wind's speed
// there, gusts and all, over the fetch its water's span gives, cut by its
// depth (TMA), each tile scaled by the root of the slope variance that sea
// holds in the tile's band.
// Terrain upwind shelters the wind over water: behind a crest it slows,
// recovering downwind (`terrain_shelter`), so valley streams lie calm while
// open lakes keep their breeze.

// The wind's tiles, the tile of the current's turbulence after them, and
// every tile (`gpu_layouts::constants::WIND_TILES`, `TURBULENCE_TILE`,
// `WAVE_TILES`).
const WATER_WIND_TILES: u32 = 4u;
const WATER_TURBULENCE_TILE: u32 = 4u;
const WATER_TILES: u32 = 5u;
// The share of a tile's period between the corners of the grid its random
// offsets hang on (`water_tile`).
const WATER_PATCH: f32 = 0.5;
// The most anisotropy the waves' sampler filters to
// (`gpu_layouts::constants::WAVE_ANISOTROPY`): along a pixel's longer side
// it averages at most this many times as finely as across.
const WATER_WAVE_ANISOTROPY: f32 = 8.0;
// Meters across the patches of stronger and calmer wind sweeping over
// water.
const WATER_GUST_SIZE: f32 = 40.0;
// The sea the wind raises over a fetch F (`engine_core::wind_waves`):
// Phillips's constant α = WATER_FETCH_ALPHA (U²/(F g))^WATER_FETCH_ALPHA_POWER
// and its peak's angular frequency WATER_FETCH_PEAK (g²/(U F))^(1/3),
// however far it blows no less than the fully raised sea's α
// (WATER_RAISED_ALPHA) and peak (WATER_RAISED_PEAK g/U); how much higher the
// spectrum stands at its peak (γ), and the peak's widths below and above
// its frequency, as shares of it (σ).
const WATER_FETCH_ALPHA: f32 = 0.076;
const WATER_FETCH_ALPHA_POWER: f32 = 0.22;
const WATER_FETCH_PEAK: f32 = 22.0;
const WATER_RAISED_ALPHA: f32 = 0.0081;
const WATER_RAISED_PEAK: f32 = 0.877;
const WATER_PEAK_ENHANCEMENT: f32 = 3.3;
const WATER_PEAK_BELOW: f32 = 0.07;
const WATER_PEAK_ABOVE: f32 = 0.09;
// The peak's widths within which its enhancement is drawn: past five, it
// lifts the spectrum by less than a hundred-thousandth.
const WATER_PEAK_REACH: f32 = 5.0;
// Points an octave by which a tile's band sums the sea's slopes (the
// midpoint rule in the logarithm of the wave number): the peak's
// enhancement, under half an octave wide, always meets one, so a band's
// share errs by a few hundredths at most.
const WATER_BAND_POINTS: f32 = 2.0;
// The meters of fetch and depth the open sea's wind blows over: farther
// than any wind needs to raise its sea fully (g F / U² past about 26,000),
// and deeper than the tiles' longest waves feel (two thirds of their
// length).
const WATER_OPEN_FETCH: f32 = 1.0e7;
const WATER_OPEN_DEPTH: f32 = 1.0e4;
// Calm water's roughness, and the most its waves' hidden slopes roughen
// it: calm water's microfacets slope about as much as the first's square.
const WATER_SMOOTH: f32 = 0.05;
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
// as it repeats `water_tile_repeats` times a meter; read at another scale,
// they scale alike.
fn water_tile_band(tile: u32) -> vec2<f32> {
    return water.waves[tile].yz;
}

// The root mean square height of tile `tile`'s waves at unit slope
// variance (meters), as it repeats `water_tile_repeats` times a meter.
fn water_tile_height(tile: u32) -> f32 {
    return water.waves[tile].w;
}

// Where corner `corner` of the grid `water_tile` hangs tile `tile`'s
// offset: a random point of the tile (in its periods).
fn water_corner_offset(corner: vec2<i32>, tile: u32) -> vec2<f32> {
    let key = (u32(corner.x) * 0x8da6b343u) ^ (u32(corner.y) * 0xd8163841u) ^ (tile * 0xcb1ab31fu);
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
// average of its moments over that span, without tiling. Its slopes (xy,
// for unit slope variance), the variance of those the span averages away
// (z, what the levels lose of the slopes' square) and its height in its own
// root mean square heights (w, averaging toward none as the span covers
// many waves). Where its texels span more than the steady size, its levels'
// averages hide nothing of it: what filtering between texels loses of the
// slopes' square (lines along the texels' edges) is no waves' variance, so
// it fades out as texels grow past it.
fn water_tile_sample(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let moments = textureSampleGrad(water_waves, water_wave_sampler, uv, tile, dx, dy);
    let texels = sqrt(length(dx) * length(dy)) * f32(textureDimensions(water_waves).x);
    let minified = saturate(log2(max(texels, 1.0e-6)) + 1.0);
    return vec4<f32>(moments.xy, max(moments.z - dot(moments.xy, moments.xy), 0.0) * minified,
        moments.w);
}

// Tile `tile`'s waves at `uv` (in its periods), which changes by `dx` and
// `dy` across the steady size (in its periods, scaled by the caller), in
// `water_tile_sample`'s units: read at the random offsets of the three
// corners of the triangle of a skewed grid holding it, blended by its
// weights there so that their variance stays the tile's (tiling and
// blending, Heitz and Neyret; Deliot and Heitz), slopes and heights by
// the weights over the root of their squares' sum and the variances by
// the squares' shares: the waves, a Gaussian field, look the same
// everywhere and never repeat.
fn water_tile(tile: u32, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let triangle = water_triangle(uv / WATER_PATCH);
    var corners = triangle.corners;
    let weights = triangle.weights;
    var sum = vec4<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let read = water_tile_sample(tile, uv + water_corner_offset(corners[corner], tile), dx,
            dy);
        let weight = weights[corner];
        sum += read * vec4<f32>(weight, weight, weight * weight, weight);
    }
    let kept = dot(weights, weights);
    let spread = inverseSqrt(kept);
    return vec4<f32>(sum.xy * spread, sum.z / kept, sum.w * spread);
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

// For each wind tile, the root of the slope variance its band holds of the
// waves no longer than `longest` meters where wind of `speed` meters a
// second blows over `fetch` meters of water `depth` meters deep. That wind
// raises a sea of Phillips's constant α and peak k_p over that fetch
// (Horvath's JONSWAP, the fully raised sea's at most), whose slopes'
// variance by the logarithm of the wave number k (deep water's ω²/g) is
// α/2 e^(−5/4 (k_p/k)²) γ^r, r = e^(−(√(k/k_p) − 1)² / 2σ²): α/2 past the
// peak, where the saturation range runs, the tiles' own spectrum. Water
// too shallow to hold the longer waves cuts them by Kitaigorodskii's factor
// of √(k h), ω √(h/g) in deep water's waves (TMA). Each band sums it by the
// midpoint rule, WATER_BAND_POINTS points an octave, a point past `longest`
// counted as far as it lies within it, so the gains change smoothly with
// all four.
fn water_wind_bands(speed: f32, fetch: f32, depth: f32, longest: f32) -> vec4<f32> {
    var gains = vec4<f32>(0.0);
    // A fetch shorter than the shortest waves the tiles hold raises none of
    // them; before the tiles arrive, they hold none.
    let shortest = water_tile_band(WATER_WIND_TILES - 1u).y;
    if speed <= 0.0 || shortest <= 0.0 {
        return gains;
    }
    // The fetch over the length the wind's speed sets (g F / U²), and the
    // sea it raises: Phillips's constant, the wave number of its peak and
    // that number's inverse root.
    let reach = WATER_GRAVITY * max(fetch, shortest) / (speed * speed);
    let alpha = max(WATER_FETCH_ALPHA * pow(reach, -WATER_FETCH_ALPHA_POWER),
        WATER_RAISED_ALPHA);
    let peak = max(WATER_FETCH_PEAK * pow(reach, -1.0 / 3.0), WATER_RAISED_PEAK)
        * WATER_GRAVITY / speed;
    let peak_number = peak * peak / WATER_GRAVITY;
    let peak_root = inverseSqrt(peak_number);
    let deep = sqrt(max(depth, 0.0));
    // The logarithm of the wave number of the longest waves counted.
    let counted = log(2.0 * PI / longest);
    let enhancement = log2(WATER_PEAK_ENHANCEMENT);
    for (var tile = 0u; tile < WATER_WIND_TILES; tile++) {
        let band = water_tile_band(tile);
        if band.y <= 0.0 || band.x <= band.y {
            continue;
        }
        // Its points, `step` apart in the logarithm of the wave number from
        // the middle of the first, nearest its longest waves, the root of
        // each point's wave number growing by `rising`.
        let points = ceil(WATER_BAND_POINTS * log2(band.x / band.y));
        let step = log(band.x / band.y) / points;
        let first = log(2.0 * PI / band.x) + 0.5 * step;
        let rising = exp(0.5 * step);
        var root = exp(0.5 * first);
        var sum = 0.0;
        for (var point = 0.0; point < points; point += 1.0) {
            let share = saturate((first + (point + 0.5) * step - counted) / step);
            if share > 0.0 {
                // k_p/k, and √(k/k_p): ω/ω_p.
                let ratio = peak_number / (root * root);
                let over = root * peak_root;
                var slopes = exp(-1.25 * ratio * ratio);
                let off = (over - 1.0) / select(WATER_PEAK_ABOVE, WATER_PEAK_BELOW, over <= 1.0);
                if off * off < WATER_PEAK_REACH * WATER_PEAK_REACH {
                    slopes *= exp2(enhancement * exp(-0.5 * off * off));
                }
                let shallow = min(root * deep, 2.0);
                let held = select(1.0 - 0.5 * (2.0 - shallow) * (2.0 - shallow),
                    0.5 * shallow * shallow, shallow <= 1.0);
                sum += slopes * held * share;
            }
            root *= rising;
        }
        gains[tile] = sqrt(0.5 * alpha * step * sum);
    }
    return gains;
}

// For each wind tile, the root of the slope variance its band holds where
// wind of `speed` meters a second (as it reaches the water, gusting and
// sheltered by the terrain upwind: `water_wind_speed`) blows over `fetch`
// meters of water `depth` meters deep, every wave the tiles hold counted
// (`water_wind_bands`): what the tile's slopes scale by there, 0 where it
// holds none.
fn water_wind_gains(speed: f32, fetch: f32, depth: f32) -> vec4<f32> {
    return water_wind_bands(speed, fetch, depth, water_tile_band(0u).x);
}

// The wind's waves at world `p` (meters along x and z), which changes by
// `dx` and `dy` a pixel (meters): each wind tile read at `p` times its
// repeats at the steady size (`water_tile`, its derivatives scaled by
// `water_steady`, so it draws no waves finer than its levels' texels
// spanning STEADY_PIXELS pixels) and scaled by its gain in `gains`
// (`water_wind_gains`). Slopes and heights add, the hidden variances as
// each gain's square times its read's, and of the heights' variance each
// tile's waves longer than two of those texels hold theirs (its heights'
// variance an octave falling as the square of their length). A tile of no
// gain reads nothing. One whose slopes stay as gentle as calm water's
// microfacets (WATER_SMOOTH's square), or whose longest waves the steady
// size spans twice over (as the sampler picks its level,
// `water_wave_span`), reads no texture and hides the square of its gain:
// each hands over across an octave, from twice the gentlest to it and from
// once the longest waves to twice, what it takes of the slopes hiding in
// the variance, so nothing pops.
fn water_wind_waves(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, gains: vec4<f32>) -> WaterRead {
    var read = WaterRead(vec2<f32>(0.0), 0.0, 0.0, 0.0);
    // At the steady size: the meters it spans as the sampler filters.
    let steady = water_steady();
    let span = water_wave_span(dx, dy) * steady;
    let calm = WATER_SMOOTH * WATER_SMOOTH;
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
            let tiled = water_tile(tile, p * repeats, dx * (repeats * steady),
                dy * (repeats * steady));
            moments = vec4<f32>(tiled.xy * kept,
                tiled.z + (1.0 - kept * kept) * dot(tiled.xy, tiled.xy), tiled.w * kept);
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
// it (`water_turning`): calm water's, widened by that variance as a
// microfacet distribution's (LEAN mapping, Olano and Baker).
fn water_roughness(variance: f32) -> f32 {
    let calm = WATER_SMOOTH * WATER_SMOOTH;
    return min(sqrt(sqrt(calm * calm + variance)), WATER_ROUGHEST);
}
