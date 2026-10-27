// Water's waves (`water_waves.rs`): WAVE_WIND_TILES tiles of the wind's
// waves, each a spectrum of WAVE_TEXELS by WAVE_TEXELS wave numbers evolving
// in time, turned into the slopes and heights water shades with by an
// inverse fast Fourier transform, rows then columns, a workgroup a line
// (Stockham's radix-2 form in workgroup memory, which needs no reordering).
// The slopes along x and z are real fields, so one complex transform
// carries both: x's as its real part, z's as its imaginary one; a second
// beside it carries the heights, in the tile's own root mean square
// heights. Each texel keeps the moments of its slopes and of its height
// (LEAN mapping, Olano and Baker 2010), which the mip levels average: the
// slopes along x and z and their squares in a layer a tile, and the slopes'
// product, the height and its square in a layer a tile after all those.
// What averaging loses of the slopes' second moments is the covariance of
// the slopes a pixel averages away, how rough and how drawn out the water
// looks from afar; what it loses of the height's square is the variance of
// the heights it averages away; and the height averages away as the waves
// do.
//
// The wind's tiles' waves also move the water toward their crests, as
// Gerstner's do: choppy waves carry it by g D where their gain is g, D = i
// (k/|k|) h (Tessendorf 2001's −i (k/|k|) h, its sign turned so crests
// gather water). Their compression T = −∂D/∂x, from (kx²/|k|) h,
// (kz²/|k|) h and (kx kz/|k|) h, three real fields carried by two more
// transforms at unit gain, is symmetric, and the surface's area changes by
// det(I − g T) = (1 − g λ₁)(1 − g λ₂), λ₁ ≥ λ₂ its eigenvalues, positive
// where it compresses, at the crests: it folds where g λ₁ reaches 1. Each
// texel's whitecaps keep the largest λ₁ it has had, fading by e every
// WATER_WHITECAP_FADE seconds of the waves' clock (`water_waves.rs`) and
// never less than none, so the foam stays where a crest folded as the crest
// runs on; and its square, which the levels average as the moments', how
// far it varies within what a level averages.

const WAVE_TEXELS: u32 = 256u;
const WAVE_STAGES: u32 = 8u;
const WAVE_WIND_TILES: u32 = 4u;
const WAVE_TAU: f32 = 6.2831855;
// Meters a second squared of gravity (`engine_terrain::hydraulics::GRAVITY`),
// and the radians per meter at which surface tension speeds waves as much as
// gravity does (capillary waves, shorter than about 1.7 cm).
const WAVE_GRAVITY: f32 = 9.81;
const WAVE_CAPILLARY: f32 = 370.0;

// This frame's waves: each tile's radians of wave number a texel (x) and
// the inverse of its waves' root mean square height at unit slope variance
// (y, none before its spectrum arrives); and the loop's phase now, coarse
// and fine (x and y: seconds into the loop over its seconds, split so high
// frequencies keep their precision), its seconds over two pi (z), and the
// share of the whitecaps the last frame left that this one keeps (w,
// e^(−Δt/τ) over the waves' seconds since; none once new spectra arrive).
struct WaveFrame {
    tiles: array<vec4<f32>, WAVE_WIND_TILES>,
    clock: vec4<f32>,
}

@group(0) @binding(0) var<uniform> waves: WaveFrame;
// Each tile's spectrum, row by row: at each wave number its amplitude and
// the amplitude of the opposite one (complex, xy and zw).
@group(0) @binding(1) var<storage, read> wave_spectrum: array<vec4<f32>>;
// What a pass reads: the rows transformed (the columns' passes) or a level
// of the moments or the whitecaps (each level's).
@group(0) @binding(2) var wave_source: texture_2d_array<f32>;
// What a pass writes: the rows transformed, the moments, the whitecaps, or
// a level of them.
@group(0) @binding(3) var wave_target: texture_storage_2d_array<rgba16float, write>;
// The whitecaps' first level as the last frame left it, which the folds'
// columns' pass fades.
@group(0) @binding(4) var wave_before: texture_2d_array<f32>;

// Two lines of the transform, the slopes' (xy) and the heights' (zw), or
// the compression's (`wave_folding`), in two halves the stages pass between.
var<workgroup> wave_line: array<vec4<f32>, 512>;

// The inverse transform of the two lines in wave_line[0..WAVE_TEXELS], each
// of the WAVE_TEXELS / 2 invocations taking one butterfly of both a stage;
// returns where in wave_line the result starts (0 or WAVE_TEXELS).
fn wave_transform(thread: u32) -> u32 {
    var source = 0u;
    for (var stage = 0u; stage < WAVE_STAGES; stage++) {
        let span = 1u << stage;
        let index = thread & (span - 1u);
        let to = WAVE_TEXELS - source;
        let a = wave_line[source + thread];
        let b = wave_line[source + thread + WAVE_TEXELS / 2u];
        let angle = WAVE_TAU * f32(index) / f32(2u * span);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let turned = vec4<f32>(complex_product(b.xy, twiddle), complex_product(b.zw, twiddle));
        let out = (thread - index) * 2u + index;
        wave_line[to + out] = a + turned;
        wave_line[to + out + span] = a - turned;
        source = to;
        workgroupBarrier();
    }
    return source;
}

// Tile `tile`'s wave at texel `texel` (x along the row) now: its wave number
// (radians a meter along x and z), and its height's amplitude (complex), the
// wave number's amplitude turned back by its frequency over the time since
// the loop began, so its wave runs along it, and its opposite's the other
// way, so the field stays real.
struct WaveAmplitude {
    number: vec2<f32>,
    height: vec2<f32>,
}

fn wave_amplitude(tile: u32, texel: vec2<u32>) -> WaveAmplitude {
    let amplitudes = wave_spectrum[(tile * WAVE_TEXELS + texel.y) * WAVE_TEXELS + texel.x];
    var wave: WaveAmplitude;
    wave.number = (vec2<f32>(texel) - f32(WAVE_TEXELS / 2u)) * waves.tiles[tile].x;
    let k = length(wave.number);
    let frequency = sqrt(WAVE_GRAVITY * k * (1.0 + k * k / (WAVE_CAPILLARY * WAVE_CAPILLARY)));
    // Frequencies rounded to whole cycles of the loop, so the waves repeat
    // seamlessly when it does.
    let cycles = round(frequency * waves.clock.z);
    let turns = fract(fract(cycles * waves.clock.x) + cycles * waves.clock.y);
    let angle = turns * WAVE_TAU;
    let turn = vec2<f32>(cos(angle), -sin(angle));
    wave.height = complex_product(amplitudes.xy, turn)
        + complex_product(vec2<f32>(amplitudes.z, -amplitudes.w), vec2<f32>(turn.x, -turn.y));
    return wave;
}

// Tile `tile`'s spectrum at texel `texel` now (`wave_amplitude`): times i k
// along x plus i times i k along z, the slopes' spectra packed as one (xy),
// and over the tile's root mean square height, the heights' (zw).
fn wave_evolved(tile: u32, texel: vec2<u32>) -> vec4<f32> {
    let wave = wave_amplitude(tile, texel);
    // i kx h + i (i kz h) = h (-kz + i kx)
    return vec4<f32>(complex_product(wave.height, vec2<f32>(-wave.number.y, wave.number.x)),
        wave.height * waves.tiles[tile].y);
}

// Wind tile `tile`'s spectrum at texel `texel` now (`wave_amplitude`) as its
// waves' compression T at unit gain: (kx²/|k|) h plus i times (kz²/|k|) h,
// T's two diagonal fields packed as one (xy), and (kx kz/|k|) h, the field
// off its diagonal (zw). Wave number zero holds no wave.
fn wave_folding(tile: u32, texel: vec2<u32>) -> vec4<f32> {
    let wave = wave_amplitude(tile, texel);
    let along = wave.number / max(length(wave.number), 1.0e-6);
    return vec4<f32>(complex_product(wave.height, wave.number * along),
        wave.height * (wave.number.x * along.y));
}

// Row `row` of tile `tile` as the workgroup's invocations wrote it into
// wave_line, transformed along x, invocation `thread` storing its two
// texels.
fn wave_rows(row: u32, tile: u32, thread: u32) {
    workgroupBarrier();
    let at = wave_transform(thread);
    for (var half = 0u; half < 2u; half++) {
        let column = thread + half * (WAVE_TEXELS / 2u);
        textureStore(wave_target, vec2<u32>(column, row), tile, wave_line[at + column]);
    }
}

// Column `column` of tile `tile`'s rows transformed (`wave_source`) into
// wave_line, invocation `thread` loading its two texels, transformed along
// z; returns where in wave_line the result starts.
fn wave_columns(column: u32, tile: u32, thread: u32) -> u32 {
    for (var half = 0u; half < 2u; half++) {
        let row = thread + half * (WAVE_TEXELS / 2u);
        wave_line[row] = textureLoad(wave_source, vec2<u32>(column, row), tile, 0);
    }
    workgroupBarrier();
    return wave_transform(thread);
}

// The point at row `row` of column `column` transformed, wave_line holding
// the column from `at` (`wave_columns`). The spectrum's middle texel is wave
// number zero, which flips every other point's sign.
fn wave_point(at: u32, column: u32, row: u32) -> vec4<f32> {
    let sign = select(1.0, -1.0, ((column + row) & 1u) == 1u);
    return wave_line[at + row] * sign;
}

// Each row (x) of each tile (y): evolved, then transformed along x.
@compute @workgroup_size(128)
fn evolve_rows(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x;
    let tile = group.y;
    let thread = local.x;
    for (var half = 0u; half < 2u; half++) {
        let column = thread + half * (WAVE_TEXELS / 2u);
        wave_line[column] = wave_evolved(tile, vec2<u32>(column, row));
    }
    wave_rows(row, tile, thread);
}

// Each column (x) of each tile's (y) rows: transformed along z, then the
// moments of its slopes and its height into the first level: the slopes
// and their squares in the tile's layer, and the slopes' product, the
// height and its square in the layer WAVE_WIND_TILES on.
@compute @workgroup_size(128)
fn transform_columns(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let column = group.x;
    let tile = group.y;
    let at = wave_columns(column, tile, local.x);
    for (var half = 0u; half < 2u; half++) {
        let row = local.x + half * (WAVE_TEXELS / 2u);
        let line = wave_point(at, column, row);
        let texel = vec2<u32>(column, row);
        textureStore(wave_target, texel, tile, vec4<f32>(line.xy, line.xy * line.xy));
        textureStore(wave_target, texel, tile + WAVE_WIND_TILES,
            vec4<f32>(line.x * line.y, line.z, line.z * line.z, 0.0));
    }
}

// Each row (x) of each wind tile (y) of its waves' compression
// (`wave_folding`): evolved, then transformed along x.
@compute @workgroup_size(128)
fn evolve_fold_rows(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x;
    let tile = group.y;
    let thread = local.x;
    for (var half = 0u; half < 2u; half++) {
        let column = thread + half * (WAVE_TEXELS / 2u);
        wave_line[column] = wave_folding(tile, vec2<u32>(column, row));
    }
    wave_rows(row, tile, thread);
}

// Each column (x) of each wind tile's (y) rows of its compression:
// transformed along z into T, whose largest eigenvalue, (Txx + Tzz)/2 plus
// the root of ((Txx − Tzz)/2)² + Txz², the whitecaps keep where it passes
// what they kept, faded by the share of it this frame keeps
// (`waves.clock.w`), into the first level: what they keep rounded to a half
// float (x), its square (y), and what the rounding lost (z), so the fade's
// small steps from frame to frame keep their precision however many frames
// a second run.
@compute @workgroup_size(128)
fn transform_fold_columns(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let column = group.x;
    let tile = group.y;
    let at = wave_columns(column, tile, local.x);
    for (var half = 0u; half < 2u; half++) {
        let row = local.x + half * (WAVE_TEXELS / 2u);
        let compression = wave_point(at, column, row).xyz;
        let largest = 0.5 * (compression.x + compression.y)
            + length(vec2<f32>(0.5 * (compression.x - compression.y), compression.z));
        let before = textureLoad(wave_before, vec2<u32>(column, row), tile, 0);
        let kept = max((before.x + before.z) * waves.clock.w, largest);
        let rounded = quantizeToF16(kept);
        textureStore(wave_target, vec2<u32>(column, row), tile,
            vec4<f32>(rounded, kept * kept, kept - rounded, 0.0));
    }
}

// Each texel of each layer (z) of a level: the mean of the four of the
// level above.
@compute @workgroup_size(8, 8)
fn average_level(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(wave_target);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let first = id.xy * 2u;
    var sum = vec4<f32>(0.0);
    for (var step = 0u; step < 4u; step++) {
        sum += textureLoad(wave_source, first + vec2<u32>(step & 1u, step >> 1u), id.z, 0);
    }
    textureStore(wave_target, id.xy, id.z, sum * 0.25);
}
