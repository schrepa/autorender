// Impostor layers in BC7, encoded where they are baked: each 4 × 4 block
// of a level of a bake's layer as one mode 6 block, sixteen colors on a
// line between two endpoints of seven bits a channel and a bit each
// endpoint shares across its channels, a four-bit entry a texel. The
// endpoints span the block's texels along their principal axis, then move
// to the least-squares fit of the entries they chose, keeping the closer of
// the two fits. Texels are read as stored (sRGB levels undecoded), as BC7
// stores them.

struct Bc7Level {
    // The level's blocks across and down, where its blocks start in the
    // output, and how many blocks apart its rows start there.
    blocks: vec2<u32>,
    first: u32,
    row: u32,
}

@group(0) @binding(0) var bc7_source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bc7_blocks: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> bc7_level: Bc7Level;

// Mode 6's sixteen weights, out of 64.
const BC7_WEIGHTS = array<u32, 16>(0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u,
    51u, 55u, 60u, 64u);

// An endpoint as mode 6 stores it: seven bits a channel and its shared bit.
struct Bc7Endpoint {
    bits: vec4<u32>,
    parity: u32,
}

// The stored endpoint whose expanded value lies nearest `value` (0 to 255).
fn bc7_quantize(value: vec4<f32>) -> Bc7Endpoint {
    var best = Bc7Endpoint(vec4<u32>(0u), 0u);
    var best_error = 3.0e38;
    for (var parity = 0u; parity < 2u; parity++) {
        let bits = vec4<u32>(clamp(round((value - f32(parity)) / 2.0), vec4<f32>(0.0),
            vec4<f32>(127.0)));
        let d = vec4<f32>(bits * 2u + parity) - value;
        let miss = dot(d, d);
        if miss < best_error {
            best_error = miss;
            best = Bc7Endpoint(bits, parity);
        }
    }
    return best;
}

// The 8-bit channels an endpoint decodes to.
fn bc7_expand(endpoint: Bc7Endpoint) -> vec4<u32> {
    return endpoint.bits * 2u + endpoint.parity;
}

// Entry `index` of the palette between `a` and `b`, as decoders compute it.
fn bc7_color(a: vec4<u32>, b: vec4<u32>, index: u32) -> vec4<f32> {
    let weight = BC7_WEIGHTS[index];
    return vec4<f32>(((64u - weight) * a + weight * b + 32u) >> vec4<u32>(6u));
}

// A fit: its endpoints, each texel's entry and the squared error.
struct Bc7Fit {
    a: Bc7Endpoint,
    b: Bc7Endpoint,
    indices: array<u32, 16>,
    miss: f32,
}

// The fit of `texels` between `low` and `high`: each texel's nearest entry.
fn bc7_fit(texels: ptr<function, array<vec4<f32>, 16>>, low: vec4<f32>, high: vec4<f32>)
        -> Bc7Fit {
    var fit: Bc7Fit;
    fit.a = bc7_quantize(low);
    fit.b = bc7_quantize(high);
    fit.miss = 0.0;
    let a = bc7_expand(fit.a);
    let b = bc7_expand(fit.b);
    for (var texel = 0u; texel < 16u; texel++) {
        var nearest = 0u;
        var nearest_miss = 3.0e38;
        for (var entry = 0u; entry < 16u; entry++) {
            let d = bc7_color(a, b, entry) - (*texels)[texel];
            let miss = dot(d, d);
            if miss < nearest_miss {
                nearest_miss = miss;
                nearest = entry;
            }
        }
        fit.indices[texel] = nearest;
        fit.miss += nearest_miss;
    }
    return fit;
}

// Writes the `count` low bits of `value` at bit `at` of `block`.
fn bc7_put(block: ptr<function, array<u32, 4>>, at: u32, count: u32, value: u32) {
    let bits = value & ((1u << count) - 1u);
    let word = at / 32u;
    let bit = at % 32u;
    (*block)[word] |= bits << bit;
    if bit + count > 32u {
        (*block)[word + 1u] |= bits >> (32u - bit);
    }
}

@compute @workgroup_size(8, 8)
fn encode_bc7(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= bc7_level.blocks.x || id.y >= bc7_level.blocks.y {
        return;
    }
    var texels: array<vec4<f32>, 16>;
    var mean = vec4<f32>(0.0);
    for (var texel = 0u; texel < 16u; texel++) {
        let at = vec2<u32>(id.x * 4u + texel % 4u, id.y * 4u + texel / 4u);
        texels[texel] = round(textureLoad(bc7_source, at, 0) * 255.0);
        mean += texels[texel] / 16.0;
    }
    // The principal axis, by power iteration over the covariance.
    var covariance = mat4x4<f32>();
    for (var texel = 0u; texel < 16u; texel++) {
        let d = texels[texel] - mean;
        covariance += mat4x4<f32>(d * d.x, d * d.y, d * d.z, d * d.w);
    }
    var axis = vec4<f32>(1.0);
    var even = false;
    for (var iteration = 0u; iteration < 8u; iteration++) {
        let next = covariance * axis;
        let reach = sqrt(dot(next, next));
        if reach < 1.0e-6 {
            even = true;
            break;
        }
        axis = next / reach;
    }
    var low = mean;
    var high = mean;
    if !even {
        var near = 3.0e38;
        var far = -3.0e38;
        for (var texel = 0u; texel < 16u; texel++) {
            let along = dot(texels[texel] - mean, axis);
            near = min(near, along);
            far = max(far, along);
        }
        low = clamp(mean + axis * near, vec4<f32>(0.0), vec4<f32>(255.0));
        high = clamp(mean + axis * far, vec4<f32>(0.0), vec4<f32>(255.0));
    }
    var best = bc7_fit(&texels, low, high);
    // The endpoints the chosen entries fit best, by least squares.
    if best.miss > 0.0 {
        var aa = 0.0;
        var ab = 0.0;
        var bb = 0.0;
        var ax = vec4<f32>(0.0);
        var bx = vec4<f32>(0.0);
        for (var texel = 0u; texel < 16u; texel++) {
            let beta = f32(BC7_WEIGHTS[best.indices[texel]]) / 64.0;
            let alpha = 1.0 - beta;
            aa += alpha * alpha;
            ab += alpha * beta;
            bb += beta * beta;
            ax += alpha * texels[texel];
            bx += beta * texels[texel];
        }
        let det = aa * bb - ab * ab;
        if abs(det) >= 1.0e-6 {
            let refined_low = clamp((bb * ax - ab * bx) / det, vec4<f32>(0.0), vec4<f32>(255.0));
            let refined_high = clamp((aa * bx - ab * ax) / det, vec4<f32>(0.0),
                vec4<f32>(255.0));
            let refined = bc7_fit(&texels, refined_low, refined_high);
            if refined.miss < best.miss {
                best = refined;
            }
        }
    }
    // The first texel's entry drops its top bit: below 8.
    var a = best.a;
    var b = best.b;
    var indices = best.indices;
    if indices[0] >= 8u {
        a = best.b;
        b = best.a;
        for (var texel = 0u; texel < 16u; texel++) {
            indices[texel] = 15u - indices[texel];
        }
    }
    var block = array<u32, 4>(0u, 0u, 0u, 0u);
    // Mode 6: its bit after six zeros.
    bc7_put(&block, 0u, 7u, 64u);
    var at = 7u;
    for (var channel = 0u; channel < 4u; channel++) {
        bc7_put(&block, at, 7u, a.bits[channel]);
        bc7_put(&block, at + 7u, 7u, b.bits[channel]);
        at += 14u;
    }
    bc7_put(&block, at, 1u, a.parity);
    bc7_put(&block, at + 1u, 1u, b.parity);
    bc7_put(&block, at + 2u, 3u, indices[0]);
    at += 5u;
    for (var texel = 1u; texel < 16u; texel++) {
        bc7_put(&block, at, 4u, indices[texel]);
        at += 4u;
    }
    bc7_blocks[bc7_level.first + id.y * bc7_level.row + id.x] =
        vec4<u32>(block[0], block[1], block[2], block[3]);
}
