import package::place::ground_surface::{
    BIOME_WORDS, H_BIOMES, H_EROSION, H_FADE, H_HYDROLOGY, H_PATHS, H_PROGRAM, H_READS,
    H_WET_BIOMES, MAX_MATERIALS, MAX_STACK, OP_ADD, OP_CONSTANT, OP_CURVATURE, OP_DEPOSIT, OP_END,
    OP_FACING, OP_FLOOD, OP_FLOW, OP_GRADIENT, OP_GRAIN, OP_HEIGHT, OP_INVERT, OP_LAKE, OP_MAX,
    OP_MIN, OP_MULTIPLY, OP_NOISE, OP_PATH, OP_POWER, OP_REMAP, OP_RIVER, OP_VALUE, OP_WETNESS,
    OP_WORN, SurfaceGround, SurfaceInputs,
};
import package::generated::terrain::{
    NONE, SURFACE_PATH_FAR, SURFACE_RIVER_EDGE_FAR, SURFACE_RIVER_EDGE_STEP, SURFACE_RIVER_EDGE_ZERO,
    SURFACE_RIVER_FAR, SurfaceEntry,
};

@group(0) @binding(0) var<storage, read> surface_program: array<u32>;
@group(0) @binding(0) var<storage, read> surface_materials: array<SurfaceEntry>;

var<private> surface_base: u32;

fn word(at: u32) -> u32 {
    return surface_program[surface_base + at];
}

fn real(at: u32) -> f32 {
    return bitcast<f32>(word(at));
}

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

fn ramp(value: f32, low: f32, high: f32) -> f32 {
    if low == high {
        return select(0.0, 1.0, value >= low);
    }
    return smooth_unit((value - low) / (high - low));
}

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

fn image_weights(t: f32) -> vec4<f32> {
    let s = 1.0 - t;
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        s * s * s * (1.0 / 6.0),
        (3.0 * t3 - 6.0 * t2 + 4.0) * (1.0 / 6.0),
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) * (1.0 / 6.0),
        t3 * (1.0 / 6.0));
}

fn image_texel(level: u32, x: i32, z: i32) -> f32 {
    let width = i32(word(level));
    let height = i32(word(level + 1u));
    let cx = clamp(x, 0, width - 1);
    let cz = clamp(z, 0, height - 1);
    return real(word(level + 2u) + u32(cz * width + cx));
}

fn image_sample(table: u32, u: f32, v: f32, level: u32) -> f32 {
    let levels = word(table);
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
    let wx = image_weights(su - u0);
    let wz = image_weights(sv - v0);
    var columns: array<f32, 4>;
    for (var column = 0; column < 4; column++) {
        let xi = x - 1 + column;
        columns[column] = wz.x * image_texel(entry, xi, z - 1) + wz.y * image_texel(entry, xi, z)
            + wz.z * image_texel(entry, xi, z + 1) + wz.w * image_texel(entry, xi, z + 2);
    }
    return wx.x * columns[0] + wx.y * columns[1] + wx.z * columns[2] + wx.w * columns[3];
}

fn decode(sample: f32, low: f32, high: f32) -> f32 {
    return low + (high - low) * sample * (1.0 / 65535.0);
}

fn image_table(section: u32, image: u32) -> u32 {
    var table = section + 7u;
    for (var index = 0u; index < image; index++) {
        table += 1u + 3u * word(table);
    }
    return table;
}

fn image_level(image_spacing: f32, levels: u32, spacing: f32) -> u32 {
    var level = 0u;
    while level < levels && image_spacing * f32(2u << level) <= spacing * 1.0001 {
        level += 1u;
    }
    return level;
}

fn erosion_change(point: vec2<f32>, spacing: f32) -> f32 {
    let section = word(H_EROSION);
    var change = 0.0;
    if section == NONE {
        return change;
    }
    let stages = word(section);
    for (var index = 0u; index < stages; index++) {
        let stage = word(section + 1u + index);
        let scale = real(stage);
        let table = stage + 4u;
        let level = image_level(real(stage + 1u), word(table) - 1u, spacing);
        let sample = image_sample(table, point.x * scale, point.y * scale, level);
        change += decode(sample, real(stage + 2u), real(stage + 3u));
    }
    return change;
}

fn path_at(point: vec2<f32>) -> vec2<f32> {
    let section = word(H_PATHS);
    var nearest = vec2<f32>(SURFACE_PATH_FAR, 0.0);
    if section == NONE {
        return nearest;
    }
    let count = word(section);
    for (var index = 0u; index < count; index++) {
        let road = word(section + 1u + index);
        let half = real(road);
        let radius = real(road + 1u);
        let cell_size = real(road + 2u);
        let corner = vec2<f32>(real(road + 3u), real(road + 4u));
        let columns = word(road + 5u);
        let rows = word(road + 6u);
        let starts = word(road + 7u);
        let segments = word(road + 8u);
        let line = word(road + 9u);
        let along = word(road + 10u);
        let spans = word(road + 11u);
        let span_count = word(road + 12u);
        var sample = vec2<f32>(SURFACE_PATH_FAR, 0.0);
        let cell = floor((point - corner) / cell_size);
        if cell.x >= 0.0 && cell.y >= 0.0 && u32(cell.x) < columns && u32(cell.y) < rows {
            let at = u32(cell.y) * columns + u32(cell.x);
            var best = 0.0;
            var best_segment = 0u;
            var best_t = 0.0;
            var found = false;
            for (var entry = word(starts + at); entry < word(starts + at + 1u); entry++) {
                let s = word(segments + entry);
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

fn shown(point: vec2<f32>) -> f32 {
    let section = word(H_FADE);
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

fn surface_level(terrain: u32, spacing: f32) -> u32 {
    let entry = surface_materials[terrain];
    return image_level(entry.image_spacing, entry.levels, spacing);
}

fn surface_river(held: vec4<u32>, present: bool) -> vec4<f32> {
    var sample = vec4<f32>(SURFACE_RIVER_FAR, 0.0, 0.0, SURFACE_RIVER_FAR);
    if !present {
        return sample;
    }
    if held.x != SURFACE_RIVER_EDGE_FAR {
        sample.x = (f32(held.x) - f32(SURFACE_RIVER_EDGE_ZERO)) * SURFACE_RIVER_EDGE_STEP;
    }
    sample.y = bitcast<f32>(held.y << 16u);
    sample.z = bitcast<f32>(held.z << 16u);
    if held.w != SURFACE_RIVER_EDGE_FAR {
        sample.w = (f32(SURFACE_RIVER_EDGE_ZERO) - f32(held.w)) * SURFACE_RIVER_EDGE_STEP;
    }
    return sample;
}

fn surface_inputs(terrain: u32, ground: SurfaceGround, level: u32, river: vec4<f32>) -> SurfaceInputs {
    surface_base = surface_materials[terrain].base;
    let spacing = ground.spacing;
    var inputs: SurfaceInputs;
    inputs.x = ground.x;
    inputs.z = ground.z;
    inputs.height = ground.height;
    inputs.gradient = ground.gradient;
    inputs.curvature = ground.curvature;
    inputs.spacing = spacing;
    inputs.river = SURFACE_RIVER_FAR;
    inputs.flood = SURFACE_RIVER_FAR;
    inputs.path_distance = SURFACE_PATH_FAR;
    let point = vec2<f32>(inputs.x, inputs.z);
    let reads = word(H_READS);
    let show = shown(point);
    let hydrology = word(H_HYDROLOGY);
    if (reads & 1u) != 0u && hydrology != NONE && show > 0.0 {
        let image_spacing = real(hydrology);
        let u = inputs.x / image_spacing;
        let v = inputs.z / image_spacing;
        let flow = decode(image_sample(image_table(hydrology, 0u), u, v, level),
            real(hydrology + 1u), real(hydrology + 2u));
        let wetness = decode(image_sample(image_table(hydrology, 1u), u, v, level),
            real(hydrology + 3u), real(hydrology + 4u));
        let lake = decode(image_sample(image_table(hydrology, 2u), u, v, level),
            real(hydrology + 5u), real(hydrology + 6u));
        inputs.flow = flow * show;
        inputs.wetness = wetness * show;
        inputs.lake = lake * show;
        inputs.river = river.x;
        inputs.grain = river.y;
        inputs.power = river.z;
        inputs.flood = river.w;
    }
    if (reads & 4u) != 0u && show > 0.0 {
        inputs.erosion = erosion_change(point, spacing) * show;
    }
    if (reads & 2u) != 0u && show > 0.0 {
        let road = path_at(point);
        inputs.path_distance = road.x;
        inputs.path_half = road.y;
    }
    return inputs;
}

fn input_ramp(code: u32, inputs: SurfaceInputs, low: f32, high: f32) -> f32 {
    let g = inputs.gradient;
    switch code {
        case OP_HEIGHT: { return ramp(inputs.height, low, high); }
        case OP_GRADIENT: { return ramp(sqrt(g.x * g.x + g.y * g.y), low, high); }
        case OP_CURVATURE: { return ramp(inputs.curvature, low, high); }
        case OP_WORN: { return ramp(pmax(-inputs.erosion, 0.0), low, high); }
        case OP_DEPOSIT: { return ramp(pmax(inputs.erosion, 0.0), low, high); }
        case OP_FLOW: { return ramp(inputs.flow, low, high); }
        case OP_WETNESS: { return ramp(inputs.wetness, low, high); }
        case OP_LAKE: { return ramp(inputs.lake, low, high); }
        case OP_RIVER: { return ramp(inputs.river, low, high); }
        case OP_GRAIN: { return ramp(inputs.grain, low, high); }
        case OP_POWER: { return ramp(inputs.power, low, high); }
        case OP_FLOOD: { return ramp(inputs.flood, low, high); }
        default: {
            let d = inputs.path_distance;
            let s = inputs.spacing;
            let h = inputs.path_half;
            let sum = ramp(abs(d + -0.375 * s) - h, low, high) + ramp(abs(d + -0.125 * s) - h, low, high)
                + ramp(abs(d + 0.125 * s) - h, low, high) + ramp(abs(d + 0.375 * s) - h, low, high);
            return sum * 0.25;
        }
    }
}

fn input_value(code: u32, inputs: SurfaceInputs) -> f32 {
    let g = inputs.gradient;
    switch code {
        case OP_HEIGHT: { return inputs.height; }
        case OP_GRADIENT: { return sqrt(g.x * g.x + g.y * g.y); }
        case OP_CURVATURE: { return inputs.curvature; }
        case OP_WORN: { return pmax(-inputs.erosion, 0.0); }
        case OP_DEPOSIT: { return pmax(inputs.erosion, 0.0); }
        case OP_FLOW: { return inputs.flow; }
        case OP_WETNESS: { return inputs.wetness; }
        case OP_LAKE: { return inputs.lake; }
        case OP_RIVER: { return inputs.river; }
        case OP_GRAIN: { return inputs.grain; }
        case OP_POWER: { return inputs.power; }
        case OP_FLOOD: { return inputs.flood; }
        default: { return abs(inputs.path_distance) - inputs.path_half; }
    }
}

fn evaluate(start: u32, inputs: SurfaceInputs) -> f32 {
    var stack: array<f32, MAX_STACK>;
    var top = 0u;
    var pc = start;
    loop {
        let code = word(pc);
        if code == OP_END || top > MAX_STACK {
            break;
        }
        switch code {
            case OP_CONSTANT: {
                stack[top] = real(pc + 1u);
                top += 1u;
                pc += 2u;
            }
            case OP_HEIGHT, OP_GRADIENT, OP_CURVATURE, OP_WORN, OP_DEPOSIT, OP_FLOW, OP_WETNESS,
                OP_LAKE, OP_RIVER, OP_PATH, OP_GRAIN, OP_POWER, OP_FLOOD: {
                stack[top] = input_ramp(code, inputs, real(pc + 1u), real(pc + 2u));
                top += 1u;
                pc += 3u;
            }
            case OP_FACING: {
                let tilt = real(pc + 1u) * inputs.gradient.x + real(pc + 2u) * inputs.gradient.y;
                stack[top] = ramp(tilt, real(pc + 3u), real(pc + 4u));
                top += 1u;
                pc += 5u;
            }
            case OP_VALUE: {
                stack[top] = input_value(word(pc + 1u), inputs);
                top += 1u;
                pc += 2u;
            }
            case OP_NOISE: {
                stack[top] = noise_mask(inputs.x, inputs.z, inputs.spacing, real(pc + 1u),
                    word(pc + 2u), word(pc + 3u));
                top += 1u;
                pc += 4u;
            }
            case OP_MULTIPLY, OP_ADD, OP_MIN, OP_MAX: {
                let count = word(pc + 1u);
                var value = 1.0;
                if code == OP_ADD {
                    value = -0.0;
                } else if code == OP_MIN {
                    value = 3.4028235e38f;
                } else if code == OP_MAX {
                    value = -3.4028235e38f;
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

fn surface_control(terrain: u32, inputs: SurfaceInputs) -> u32 {
    surface_base = surface_materials[terrain].base;
    let program = word(H_PROGRAM);
    let biomes = word(H_BIOMES);
    let wet_biomes = word(H_WET_BIOMES);
    var materials: array<u32, MAX_MATERIALS>;
    var weights: array<f32, MAX_MATERIALS>;
    var count = 0u;
    var remaining = 1.0;
    var soaking = select(0.0, 1.0, wet_biomes > 0u);
    var wet = 0.0;
    for (var index = biomes; index > 0u; index--) {
        let current = index - 1u;
        let at = program + BIOME_WORDS * current;
        var fit = 1.0;
        if current > 0u {
            fit = clamp_rule(evaluate(word(at), inputs), 0.0, 1.0);
        }
        let share = fit * remaining;
        remaining *= 1.0 - fit;
        if current < wet_biomes {
            let held = fit * soaking;
            soaking *= 1.0 - fit;
            let rule = word(at + 2u);
            if held > 0.0 && rule != NONE {
                wet += held * clamp_rule(evaluate(rule, inputs), 0.0, 1.0);
            }
        }
        if share > 0.0 {
            var left = share;
            let list = word(at + 1u);
            let layers = word(list);
            for (var layer = layers; layer > 0u; layer--) {
                let l = layer - 1u;
                let entry = list + 1u + 3u * l;
                var cover = 1.0;
                if l > 0u {
                    cover = clamp_rule(evaluate(word(entry + 2u), inputs) * real(entry + 1u), 0.0, 1.0);
                }
                let weight = cover * left;
                left *= 1.0 - cover;
                if weight > 0.0 {
                    let material = word(entry);
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
        if remaining <= 0.0 && soaking <= 0.0 {
            break;
        }
    }
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
    let soaked = clamp_rule(wet, 0.0, 1.0);
    let packed = (first & 0xffu) | ((second & 0xffu) << 8u)
        | (min(u32(blend * 255.0 + 0.5), 255u) << 16u)
        | (min(u32(soaked * 255.0 + 0.5), 255u) << 24u);
    return packed;
}
