// Water's waves (`water_waves.rs`): WAVE_TILES tiles of waves, each a
// spectrum of WAVE_TEXELS by WAVE_TEXELS wave numbers, turned into the
// slopes and heights water shades with by an inverse fast Fourier
// transform, rows then columns, a workgroup a line (Stockham's radix-2
// form in workgroup memory, which needs no reordering). The first
// WAVE_WIND_TILES, the wind's, evolve in time; the last, the current's
// turbulence, stands as drawn, its waves moving only as the current carries
// them. The slopes along x and z are real fields, so one complex transform
// carries both: x's as its real part, z's as its imaginary one; a second
// beside it carries the heights, in the tile's own root mean square
// heights. Each texel keeps its slopes' moments (the slopes along x and z,
// and the square of their length) and its height, which the mip levels
// average: what averaging loses of the square is how rough the water looks
// from afar (LEAN mapping), and the height averages away as the waves do.

const WAVE_TEXELS: u32 = 256u;
const WAVE_STAGES: u32 = 8u;
const WAVE_TILES: u32 = 5u;
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
// frequencies keep their precision), and its seconds over two pi (z).
struct WaveFrame {
    tiles: array<vec4<f32>, WAVE_TILES>,
    clock: vec4<f32>,
}

@group(0) @binding(0) var<uniform> waves: WaveFrame;
// Each tile's spectrum, row by row: at each wave number its amplitude and
// the amplitude of the opposite one (complex, xy and zw).
@group(0) @binding(1) var<storage, read> wave_spectrum: array<vec4<f32>>;
// What a pass reads: the rows transformed (the columns' pass) or a level of
// the moments (each level's).
@group(0) @binding(2) var wave_source: texture_2d_array<f32>;
// What a pass writes: the rows transformed, the moments, or a level of
// them.
@group(0) @binding(3) var wave_target: texture_storage_2d_array<rgba16float, write>;

// Two lines of the transform, the slopes' (xy) and the heights' (zw), in
// two halves the stages pass between.
var<workgroup> wave_line: array<vec4<f32>, 512>;

// The product of complex numbers `a` and `b`.
fn wave_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

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
        let turned = vec4<f32>(wave_multiply(b.xy, twiddle), wave_multiply(b.zw, twiddle));
        let out = (thread - index) * 2u + index;
        wave_line[to + out] = a + turned;
        wave_line[to + out + span] = a - turned;
        source = to;
        workgroupBarrier();
    }
    return source;
}

// Tile `tile`'s spectrum at texel `texel` (x along the row) now: each wave
// number's amplitude turned back by its frequency over the time since the
// loop began, so its wave runs along it, and its opposite's the other way,
// so the field stays real (the turbulence's as drawn, unturned); then times
// i k along x plus i times i k along z, the slopes' spectra packed as one
// (xy), and over the tile's root mean square height, the heights' (zw).
fn wave_evolved(tile: u32, texel: vec2<u32>) -> vec4<f32> {
    let amplitudes = wave_spectrum[(tile * WAVE_TEXELS + texel.y) * WAVE_TEXELS + texel.x];
    let number = (vec2<f32>(texel) - f32(WAVE_TEXELS / 2u)) * waves.tiles[tile].x;
    let k = length(number);
    let frequency = sqrt(WAVE_GRAVITY * k * (1.0 + k * k / (WAVE_CAPILLARY * WAVE_CAPILLARY)));
    // Frequencies rounded to whole cycles of the loop, so the waves repeat
    // seamlessly when it does.
    let cycles = round(frequency * waves.clock.z);
    let turns = fract(fract(cycles * waves.clock.x) + cycles * waves.clock.y);
    let angle = select(turns * WAVE_TAU, 0.0, tile >= WAVE_WIND_TILES);
    let turn = vec2<f32>(cos(angle), -sin(angle));
    let height = wave_multiply(amplitudes.xy, turn)
        + wave_multiply(vec2<f32>(amplitudes.z, -amplitudes.w), vec2<f32>(turn.x, -turn.y));
    // i kx h + i (i kz h) = h (-kz + i kx)
    return vec4<f32>(wave_multiply(height, vec2<f32>(-number.y, number.x)),
        height * waves.tiles[tile].y);
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
    workgroupBarrier();
    let at = wave_transform(thread);
    for (var half = 0u; half < 2u; half++) {
        let column = thread + half * (WAVE_TEXELS / 2u);
        textureStore(wave_target, vec2<u32>(column, row), tile, wave_line[at + column]);
    }
}

// Each column (x) of each tile's (y) rows: transformed along z, then its
// slopes' moments and its height into the first level. The spectrum's
// middle texel is wave number zero, which flips every other point's sign.
@compute @workgroup_size(128)
fn transform_columns(@builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let column = group.x;
    let tile = group.y;
    let thread = local.x;
    for (var half = 0u; half < 2u; half++) {
        let row = thread + half * (WAVE_TEXELS / 2u);
        wave_line[row] = textureLoad(wave_source, vec2<u32>(column, row), tile, 0);
    }
    workgroupBarrier();
    let at = wave_transform(thread);
    for (var half = 0u; half < 2u; half++) {
        let row = thread + half * (WAVE_TEXELS / 2u);
        let sign = select(1.0, -1.0, ((column + row) & 1u) == 1u);
        let line = wave_line[at + row] * sign;
        textureStore(wave_target, vec2<u32>(column, row), tile,
            vec4<f32>(line.xy, dot(line.xy, line.xy), line.z));
    }
}

// Each texel of each tile (z) of a level: the mean of the four of the
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
