// Terrain surface materials on the GPU: a tile's control
// values from its heights, once when it uploads, into the same layer of the
// control array. Each sample's inputs are read as the CPU reads them for a
// tile's rows (height, gradient and curvature from the heights around it;
// the hydrology bake's flow, wetness and lakes; the nearest river's edge;
// the nearest road), and the surface's rules run as a program for this
// interpreter (`engine_terrain::gpu`), in the order the CPU evaluates them.
// Values match the CPU's but where rounding moves one across a rule's
// threshold; gameplay reads the CPU's.

struct Job {
    // The layer, the tile's first sample (x and z, from the terrain's
    // first, with surroundings' offset), its sample spacing and the
    // hydrology image level its masks read.
    layer: u32,
    level: u32,
    first: vec2<i32>,
    spacing: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> job: Job;
@group(0) @binding(2) var<storage, read> data: array<u32>;
@group(0) @binding(3) var control: texture_storage_2d<r32uint, write>;

const NONE: u32 = 0xffffffffu;
const MAX_STACK: u32 = 16u;
const MAX_MATERIALS: u32 = 64u;
const RIVER_FAR: f32 = 1.0e9;
const PATH_FAR: f32 = 1.0e9;

// The header's words (`engine_terrain::gpu::header`).
const H_PROGRAM: u32 = 0u;
const H_BIOMES: u32 = 2u;
const H_READS: u32 = 4u;
const H_HYDROLOGY: u32 = 5u;
const H_RIVERS: u32 = 6u;
const H_PATHS: u32 = 7u;
const H_FADE: u32 = 8u;

// Opcodes (`engine_terrain::gpu::op`).
const OP_CONSTANT: u32 = 0u;
const OP_HEIGHT: u32 = 1u;
const OP_GRADIENT: u32 = 2u;
const OP_CURVATURE: u32 = 3u;
const OP_FLOW: u32 = 4u;
const OP_WETNESS: u32 = 5u;
const OP_LAKE: u32 = 6u;
const OP_RIVER: u32 = 7u;
const OP_PATH: u32 = 8u;
const OP_NOISE: u32 = 9u;
const OP_MULTIPLY: u32 = 10u;
const OP_ADD: u32 = 11u;
const OP_MIN: u32 = 12u;
const OP_MAX: u32 = 13u;
const OP_INVERT: u32 = 14u;
const OP_REMAP: u32 = 15u;
const OP_END: u32 = 16u;

fn real(at: u32) -> f32 {
    return bitcast<f32>(data[at]);
}

// The engine's lane rules for the smaller and larger of two values
// (`engine_math::min` and `max`), and clamping between them.
fn pmin(a: f32, b: f32) -> f32 {
    return select(a, b, b < a);
}

fn pmax(a: f32, b: f32) -> f32 {
    return select(a, b, a < b);
}

fn clamp_rule(x: f32, low: f32, high: f32) -> f32 {
    return pmin(pmax(x, low), high);
}

fn smooth_unit(t: f32) -> f32 {
    let c = clamp_rule(t, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
}

// 0 at `low`, 1 at `high`, smooth between; a step where they meet.
fn ramp(value: f32, low: f32, high: f32) -> f32 {
    if low == high {
        return select(0.0, 1.0, value >= low);
    }
    return smooth_unit((value - low) / (high - low));
}

// ---------------------------------------------------------------- noise

fn mix_bits(value: u32) -> u32 {
    var h = value;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn lattice_hash(x: i32, z: i32, seed: u32) -> u32 {
    return mix_bits(mix_bits(seed ^ bitcast<u32>(x)) + bitcast<u32>(z));
}

// Unit gradients 22.5 degrees apart, turned half a step off the axes.
fn gradient_at(x: i32, z: i32, seed: u32) -> vec2<f32> {
    var gradients = array<vec2<f32>, 16>(
        vec2<f32>(0.9807853, 0.19509032), vec2<f32>(0.8314696, 0.5555702),
        vec2<f32>(0.5555702, 0.8314696), vec2<f32>(0.19509032, 0.9807853),
        vec2<f32>(-0.19509032, 0.9807853), vec2<f32>(-0.5555702, 0.8314696),
        vec2<f32>(-0.8314696, 0.5555702), vec2<f32>(-0.9807853, 0.19509032),
        vec2<f32>(-0.9807853, -0.19509032), vec2<f32>(-0.8314696, -0.5555702),
        vec2<f32>(-0.5555702, -0.8314696), vec2<f32>(-0.19509032, -0.9807853),
        vec2<f32>(0.19509032, -0.9807853), vec2<f32>(0.5555702, -0.8314696),
        vec2<f32>(0.8314696, -0.5555702), vec2<f32>(0.9807853, -0.19509032));
    return gradients[lattice_hash(x, z, seed) & 15u];
}

fn quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Gradient noise's value at lattice coordinates (x, z), about -1 to 1.
fn gradient_value(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = floor(x);
    let z0 = floor(z);
    let i = i32(x0);
    let j = i32(z0);
    let fx = x - x0;
    let fz = z - z0;
    let ga = gradient_at(i, j, seed);
    let gb = gradient_at(i + 1, j, seed);
    let gc = gradient_at(i, j + 1, seed);
    let gd = gradient_at(i + 1, j + 1, seed);
    let ux = quintic(fx);
    let uz = quintic(fz);
    let va = ga.x * fx + ga.y * fz;
    let vb = gb.x * (fx - 1.0) + gb.y * fz;
    let vc = gc.x * fx + gc.y * (fz - 1.0);
    let vd = gd.x * (fx - 1.0) + gd.y * (fz - 1.0);
    let top = va + ux * (vb - va);
    let bottom = vc + ux * (vd - vc);
    return (top + uz * (bottom - top)) * 1.4142135;
}

fn noise_mask(x: f32, z: f32, spacing: f32, frequency: f32, seed: u32, octaves: u32) -> f32 {
    var sum = 0.0;
    var weight = 0.0;
    var amplitude = 1.0;
    var scale = frequency;
    for (var octave = 0u; octave < octaves; octave++) {
        let shift = f32(octave) * 17.13;
        let fade = clamp_rule(0.5 / (scale * spacing) - 1.0, 0.0, 1.0);
        if fade > 0.0 {
            sum += amplitude * fade
                * gradient_value(x * scale + shift, z * scale - shift, seed + octave * 0x9e3779b9u);
        }
        weight += amplitude;
        amplitude *= 0.5;
        scale *= 2.0;
    }
    return clamp_rule(0.5 + 0.5 * sum / weight, 0.0, 1.0);
}

// ---------------------------------------------------------------- inputs

struct Inputs {
    x: f32,
    z: f32,
    height: f32,
    gradient: f32,
    curvature: f32,
    spacing: f32,
    flow: f32,
    wetness: f32,
    lake: f32,
    river: f32,
    path_distance: f32,
    path_half: f32,
}

fn catmull_rom(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2));
}

// Texel (x, z) of one of an image's levels, edges clamped: `level` points
// at its width, height and first sample.
fn image_texel(level: u32, x: i32, z: i32) -> f32 {
    let width = i32(data[level]);
    let height = i32(data[level + 1u]);
    let cx = clamp(x, 0, width - 1);
    let cz = clamp(z, 0, height - 1);
    return real(data[level + 2u] + u32(cz * width + cx));
}

// Catmull-Rom interpolation of the image whose table starts at `table` at
// image coordinates (u, v), read from level `level` (the image's own where
// it has no such level).
fn image_sample(table: u32, u: f32, v: f32, level: u32) -> f32 {
    let levels = data[table];
    var entry = table + 1u;
    var su = u;
    var sv = v;
    if level > 0u && level < levels {
        entry = table + 1u + 3u * level;
        let scale = 1.0 / f32(1u << level);
        su = u * scale;
        sv = v * scale;
    }
    let u0 = floor(su);
    let v0 = floor(sv);
    let x = i32(u0);
    let z = i32(v0);
    let wx = catmull_rom(su - u0);
    let wz = catmull_rom(sv - v0);
    var sum = 0.0;
    for (var row = 0; row < 4; row++) {
        let zi = z - 1 + row;
        let line = wx.x * image_texel(entry, x - 1, zi) + wx.y * image_texel(entry, x, zi)
            + wx.z * image_texel(entry, x + 1, zi) + wx.w * image_texel(entry, x + 2, zi);
        sum += wz[row] * line;
    }
    return sum;
}

fn decode(sample: f32, low: f32, high: f32) -> f32 {
    return low + (high - low) * sample * (1.0 / 65535.0);
}

// The tables of the hydrology section's three images.
fn image_table(section: u32, image: u32) -> u32 {
    var table = section + 7u;
    for (var index = 0u; index < image; index++) {
        table += 1u + 3u * data[table];
    }
    return table;
}

// Meters from the nearest river's edge within its banks' reach, as the CPU
// finds it: the first nearest among the point's cell's segments.
fn river_edge(point: vec2<f32>) -> f32 {
    let section = data[H_RIVERS];
    if section == NONE {
        return RIVER_FAR;
    }
    let corner = vec2<f32>(real(section), real(section + 1u));
    let columns = data[section + 2u];
    let rows = data[section + 3u];
    let cell_size = real(section + 4u);
    let reach = real(section + 5u);
    let cell = floor((point - corner) / cell_size);
    if !(cell.x >= 0.0 && cell.y >= 0.0) {
        return RIVER_FAR;
    }
    let column = u32(cell.x);
    let row = u32(cell.y);
    if column >= columns || row >= rows {
        return RIVER_FAR;
    }
    let at = row * columns + column;
    let starts = data[section + 6u];
    let list = data[section + 7u];
    let table = data[section + 8u];
    var best = RIVER_FAR;
    var best_half = 0.0;
    var found = false;
    for (var entry = data[starts + at]; entry < data[starts + at + 1u]; entry++) {
        let segment = table + 6u * data[list + entry];
        let a = vec2<f32>(real(segment), real(segment + 1u));
        let b = vec2<f32>(real(segment + 2u), real(segment + 3u));
        let run = b - a;
        let length = sqrt(run.x * run.x + run.y * run.y);
        var t = 0.0;
        if length > 0.0 {
            let toward = point - a;
            t = pmin(pmax((toward.x * run.x + toward.y * run.y) / (length * length), 0.0), 1.0);
        }
        let gap = point - (a + run * t);
        let distance = sqrt(gap.x * gap.x + gap.y * gap.y);
        let ha = real(segment + 4u);
        let half = ha + (real(segment + 5u) - ha) * t;
        if distance < half + reach && (!found || distance < best) {
            best = distance;
            best_half = half;
            found = true;
        }
    }
    if !found {
        return RIVER_FAR;
    }
    return best - best_half;
}

// The nearest road as the point sees it (distance from its line and half
// its width), the first of the stack's roads whose edge lies nearest.
fn path_at(point: vec2<f32>) -> vec2<f32> {
    let section = data[H_PATHS];
    var nearest = vec2<f32>(PATH_FAR, 0.0);
    if section == NONE {
        return nearest;
    }
    let count = data[section];
    for (var index = 0u; index < count; index++) {
        let road = data[section + 1u + index];
        let half = real(road);
        let radius = real(road + 1u);
        let cell_size = real(road + 2u);
        let corner = vec2<f32>(real(road + 3u), real(road + 4u));
        let columns = data[road + 5u];
        let rows = data[road + 6u];
        let starts = data[road + 7u];
        let segments = data[road + 8u];
        let line = data[road + 9u];
        let along = data[road + 10u];
        let spans = data[road + 11u];
        let span_count = data[road + 12u];
        var sample = vec2<f32>(PATH_FAR, 0.0);
        let cell = floor((point - corner) / cell_size);
        if cell.x >= 0.0 && cell.y >= 0.0 && u32(cell.x) < columns && u32(cell.y) < rows {
            let at = u32(cell.y) * columns + u32(cell.x);
            var best = 0.0;
            var best_segment = 0u;
            var best_t = 0.0;
            var found = false;
            for (var entry = data[starts + at]; entry < data[starts + at + 1u]; entry++) {
                let s = data[segments + entry];
                let a = vec2<f32>(real(line + 2u * s), real(line + 2u * s + 1u));
                let b = vec2<f32>(real(line + 2u * s + 2u), real(line + 2u * s + 3u));
                let run = b - a;
                let length = run.x * run.x + run.y * run.y;
                var t = 0.0;
                if length > 0.0 {
                    let toward = point - a;
                    t = pmin(pmax((toward.x * run.x + toward.y * run.y) / length, 0.0), 1.0);
                }
                let gap = point - (a + run * t);
                let distance = sqrt(gap.x * gap.x + gap.y * gap.y);
                if !found || distance < best {
                    best = distance;
                    best_segment = s;
                    best_t = t;
                    found = true;
                }
            }
            if found && best < radius {
                let a_along = real(along + best_segment);
                let distance_along = a_along + (real(along + best_segment + 1u) - a_along) * best_t;
                var bridged = false;
                for (var span = 0u; span < span_count; span++) {
                    let start = real(spans + 2u * span);
                    let end = real(spans + 2u * span + 1u);
                    bridged = bridged || (distance_along > start && distance_along < end);
                }
                if !bridged {
                    sample = vec2<f32>(best, half);
                }
            }
        }
        if sample.x - sample.y < nearest.x - nearest.y {
            nearest = sample;
        }
    }
    return nearest;
}

// How much of the stack's bakes and roads surroundings show:
// 1 within the region they were drawn for, fading out beyond.
fn shown(point: vec2<f32>) -> f32 {
    let section = data[H_FADE];
    if section == NONE {
        return 1.0;
    }
    let low = vec2<f32>(real(section), real(section + 1u));
    let high = vec2<f32>(real(section + 2u), real(section + 3u));
    let blend = real(section + 4u);
    let beyond = max(max(low - point, point - high), vec2<f32>(0.0));
    let outside = sqrt(beyond.x * beyond.x + beyond.y * beyond.y);
    if outside <= 0.0 {
        return 1.0;
    }
    if blend <= 0.0 {
        return 0.0;
    }
    let t = pmin(outside / blend, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

fn sample_inputs(column: u32, row: u32) -> Inputs {
    let last = TILE_STRIDE - 1u;
    let up = select(row - 1u, 0u, row == 0u);
    let down = min(row + 1u, last);
    let left = select(column - 1u, 0u, column == 0u);
    let right = min(column + 1u, last);
    let spacing = job.spacing;
    let height = tile_height(column, row);
    let west = tile_height(left, row);
    let east = tile_height(right, row);
    let north = tile_height(column, up);
    let south = tile_height(column, down);
    let gx = (east - west) / (2.0 * spacing);
    let gz = (south - north) / (2.0 * spacing);
    var inputs: Inputs;
    inputs.x = f32(job.first.x + i32(column) - 1) * spacing;
    inputs.z = f32(job.first.y + i32(row) - 1) * spacing;
    inputs.height = height;
    inputs.gradient = sqrt(gx * gx + gz * gz);
    inputs.curvature = (west + east + north + south - 4.0 * height) / (spacing * spacing);
    inputs.spacing = spacing;
    inputs.river = RIVER_FAR;
    inputs.path_distance = PATH_FAR;
    let point = vec2<f32>(inputs.x, inputs.z);
    let reads = data[H_READS];
    let show = shown(point);
    let hydrology = data[H_HYDROLOGY];
    if (reads & 1u) != 0u && hydrology != NONE && show > 0.0 {
        let image_spacing = real(hydrology);
        let u = inputs.x / image_spacing;
        let v = inputs.z / image_spacing;
        let flow = decode(image_sample(image_table(hydrology, 0u), u, v, job.level),
            real(hydrology + 1u), real(hydrology + 2u));
        let wetness = decode(image_sample(image_table(hydrology, 1u), u, v, job.level),
            real(hydrology + 3u), real(hydrology + 4u));
        let lake = decode(image_sample(image_table(hydrology, 2u), u, v, job.level),
            real(hydrology + 5u), real(hydrology + 6u));
        inputs.flow = flow * show;
        inputs.wetness = wetness * show;
        inputs.lake = lake * show;
        inputs.river = river_edge(point);
    }
    if (reads & 2u) != 0u && show > 0.0 {
        let road = path_at(point);
        inputs.path_distance = road.x;
        inputs.path_half = road.y;
    }
    return inputs;
}

// ---------------------------------------------------------------- program

fn input_ramp(code: u32, inputs: Inputs, low: f32, high: f32) -> f32 {
    switch code {
        case OP_HEIGHT: { return ramp(inputs.height, low, high); }
        case OP_GRADIENT: { return ramp(inputs.gradient, low, high); }
        case OP_CURVATURE: { return ramp(inputs.curvature, low, high); }
        case OP_FLOW: { return ramp(inputs.flow, low, high); }
        case OP_WETNESS: { return ramp(inputs.wetness, low, high); }
        case OP_LAKE: { return ramp(inputs.lake, low, high); }
        case OP_RIVER: { return ramp(inputs.river, low, high); }
        default: {
            // A road's ramp at four points across the sample's footprint,
            // across the road.
            let d = inputs.path_distance;
            let s = inputs.spacing;
            let h = inputs.path_half;
            let sum = ramp(abs(d + -0.375 * s) - h, low, high) + ramp(abs(d + -0.125 * s) - h, low, high)
                + ramp(abs(d + 0.125 * s) - h, low, high) + ramp(abs(d + 0.375 * s) - h, low, high);
            return sum * 0.25;
        }
    }
}

// The mask whose code starts at `start`.
fn evaluate(start: u32, inputs: Inputs) -> f32 {
    var stack: array<f32, MAX_STACK>;
    var top = 0u;
    var pc = start;
    loop {
        let code = data[pc];
        if code == OP_END || top > MAX_STACK {
            break;
        }
        switch code {
            case OP_CONSTANT: {
                stack[top] = real(pc + 1u);
                top += 1u;
                pc += 2u;
            }
            case OP_HEIGHT, OP_GRADIENT, OP_CURVATURE, OP_FLOW, OP_WETNESS, OP_LAKE, OP_RIVER, OP_PATH: {
                stack[top] = input_ramp(code, inputs, real(pc + 1u), real(pc + 2u));
                top += 1u;
                pc += 3u;
            }
            case OP_NOISE: {
                stack[top] = noise_mask(inputs.x, inputs.z, inputs.spacing, real(pc + 1u),
                    data[pc + 2u], data[pc + 3u]);
                top += 1u;
                pc += 4u;
            }
            case OP_MULTIPLY, OP_ADD, OP_MIN, OP_MAX: {
                // Lists fold from their first value, which folding it into
                // the CPU's identity gives back; an empty list gives the
                // identity, the largest finite values standing in for
                // infinities, which constants may not hold.
                let count = data[pc + 1u];
                var value = 1.0;
                if code == OP_ADD {
                    value = -0.0;
                } else if code == OP_MIN {
                    value = 0x1.fffffep+127f;
                } else if code == OP_MAX {
                    value = -0x1.fffffep+127f;
                }
                let first = top - count;
                if count > 0u {
                    value = stack[first];
                }
                for (var index = first + 1u; index < top; index++) {
                    let item = stack[index];
                    if code == OP_MULTIPLY {
                        value = value * item;
                    } else if code == OP_ADD {
                        value = value + item;
                    } else if code == OP_MIN {
                        value = pmin(value, item);
                    } else {
                        value = pmax(value, item);
                    }
                }
                stack[first] = value;
                top = first + 1u;
                pc += 2u;
            }
            case OP_INVERT: {
                stack[top - 1u] = 1.0 - stack[top - 1u];
                pc += 1u;
            }
            case OP_REMAP: {
                stack[top - 1u] = ramp(stack[top - 1u], real(pc + 1u), real(pc + 2u));
                pc += 3u;
            }
            default: {
                return 0.0;
            }
        }
    }
    return stack[0];
}

@compute @workgroup_size(8, 8)
fn tile_control(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let inputs = sample_inputs(id.x, id.y);
    let program = data[H_PROGRAM];
    let biomes = data[H_BIOMES];
    // Later biomes and layers paint over earlier ones: walk them from the
    // last, each taking what the later ones left, until nothing is left;
    // materials' weights in the order they first weigh.
    var materials: array<u32, MAX_MATERIALS>;
    var weights: array<f32, MAX_MATERIALS>;
    var count = 0u;
    var remaining = 1.0;
    var biome = 0u;
    var biome_share = -1.0;
    for (var index = biomes; index > 0u; index--) {
        let current = index - 1u;
        let fitness = data[program + 2u * current];
        var fit = 1.0;
        if current > 0u {
            fit = clamp_rule(evaluate(fitness, inputs), 0.0, 1.0);
        }
        let share = fit * remaining;
        remaining *= 1.0 - fit;
        if share > 0.0 {
            if share > biome_share {
                biome = current;
                biome_share = share;
            }
            var left = share;
            let list = data[program + 2u * current + 1u];
            let layers = data[list];
            for (var layer = layers; layer > 0u; layer--) {
                let l = layer - 1u;
                let entry = list + 1u + 3u * l;
                var cover = 1.0;
                if l > 0u {
                    cover = clamp_rule(evaluate(data[entry + 2u], inputs) * real(entry + 1u), 0.0, 1.0);
                }
                let weight = cover * left;
                left *= 1.0 - cover;
                if weight > 0.0 {
                    let material = data[entry];
                    var placed = false;
                    for (var m = 0u; m < count; m++) {
                        if materials[m] == material {
                            weights[m] += weight;
                            placed = true;
                            break;
                        }
                    }
                    if !placed && count < MAX_MATERIALS {
                        materials[count] = material;
                        weights[count] = weight;
                        count += 1u;
                    }
                }
                if left <= 0.0 {
                    break;
                }
            }
        }
        if remaining <= 0.0 {
            break;
        }
    }
    // The two materials weighing most, the first to weigh on ties.
    var first = 0u;
    var first_weight = -1.0;
    var second = 0u;
    var second_weight = -1.0;
    for (var m = 0u; m < count; m++) {
        let weight = weights[m];
        if weight > first_weight {
            second = first;
            second_weight = first_weight;
            first = materials[m];
            first_weight = weight;
        } else if weight > second_weight {
            second = materials[m];
            second_weight = weight;
        }
    }
    if second_weight <= 0.0 {
        second = first;
        second_weight = 0.0;
    }
    let blend = second_weight / pmax(first_weight + second_weight, 1.0e-12);
    let packed = (first & 0xffu) | ((second & 0xffu) << 8u)
        | (min(u32(blend * 255.0 + 0.5), 255u) << 16u) | ((biome & 0xffu) << 24u);
    textureStore(control, id.xy, vec4<u32>(packed, 0u, 0u, 0u));
}
