struct Quad {
    rect: vec4<f32>,
    uv: vec4<f32>,
    fill: u32,
    edge: u32,
    mode: u32,
    radius: f32,
    stroke_width: f32,
    stroke_inset: f32,
    scale: f32,
    angle: f32,
    case_color: u32,
    case_grow: f32,
    case_offset: vec2<f32>,
    rim: u32,
    rim_grow: f32,
    align: vec2<u32>,
}

struct ScreenQuads {
    canvas: vec4<f32>,
    quads: array<Quad>,
}

@group(2) @binding(0) var<storage, read> screen_quads: ScreenQuads;
@group(2) @binding(1) var screen_icons: texture_2d<f32>;
@group(2) @binding(2) var screen_glyphs_0: texture_2d<f32>;
@group(2) @binding(3) var screen_glyphs_1: texture_2d<f32>;
@group(2) @binding(4) var screen_picture_0: texture_2d<f32>;
@group(2) @binding(5) var screen_picture_1: texture_2d<f32>;
@group(2) @binding(6) var screen_picture_2: texture_2d<f32>;
@group(2) @binding(7) var screen_picture_3: texture_2d<f32>;
@group(2) @binding(8) var screen_picture_4: texture_2d<f32>;
@group(2) @binding(9) var screen_picture_5: texture_2d<f32>;
@group(2) @binding(10) var screen_picture_6: texture_2d<f32>;
@group(2) @binding(11) var screen_picture_7: texture_2d<f32>;
@group(2) @binding(12) var screen_picture_8: texture_2d<f32>;
@group(2) @binding(13) var screen_picture_9: texture_2d<f32>;
@group(2) @binding(14) var screen_sampler: sampler;

const PLATE: u32 = 0u;
const ICON: u32 = 1u;
const IMAGE: u32 = 3u;
const PLATE_PAD: f32 = 1.0;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) index: u32,
}

fn corner_of(vertex: u32) -> vec2<f32> {
    let x = f32((0x32u >> vertex) & 1u);
    let y = f32((0x2cu >> vertex) & 1u);
    return vec2<f32>(x, y);
}

@vertex
fn vs_screen(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> Corner {
    let quad = screen_quads.quads[index];
    let corner = corner_of(vertex);
    let size = quad.rect.zw;
    let half = size * 0.5;
    var low = -half;
    var high = half;
    let framed = framed_image(quad);
    if quad.mode == PLATE || framed {
        if quad.case_color != 0u {
            let grow = vec2<f32>(quad.case_grow);
            var near = quad.case_offset;
            var far = quad.case_offset;
            if doubled(quad) {
                near = min(near, vec2<f32>(0.0));
                far = max(far, vec2<f32>(0.0));
            }
            low = min(low, near - half - grow);
            high = max(high, far + half + grow);
        }
        var soft = quad.scale;
        if framed {
            soft = 1.0;
        }
        let pad = vec2<f32>(PLATE_PAD * soft);
        low = low - pad;
        high = high + pad;
    }
    let local = mix(low, high, corner);
    let turn = vec2<f32>(cos(quad.angle), sin(quad.angle));
    let turned = vec2<f32>(local.x * turn.x - local.y * turn.y, local.x * turn.y + local.y * turn.x);
    let at = quad.rect.xy + size * 0.5 + turned;
    let canvas = max(screen_quads.canvas.xy, vec2<f32>(1.0));
    var out: Corner;
    out.position = vec4<f32>(at.x / canvas.x * 2.0 - 1.0, 1.0 - at.y / canvas.y * 2.0, 0.0, 1.0);
    out.local = local;
    out.uv = quad.uv.xy + (local / max(size, vec2<f32>(1.0e-4)) + vec2<f32>(0.5)) * quad.uv.zw;
    out.index = index;
    return out;
}

fn framed_image(quad: Quad) -> bool {
    return quad.mode == IMAGE && (quad.case_color != 0u || quad.radius > 0.0);
}

fn doubled(quad: Quad) -> bool {
    if quad.mode == IMAGE {
        return quad.stroke_inset > 0.5;
    }
    return quad.uv.y > 0.5;
}

fn rounded_box(point: vec2<f32>, half: vec2<f32>, radius: f32) -> f32 {
    let r = clamp(radius, 0.0, min(half.x, half.y));
    let q = abs(point) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn coverage(distance: f32) -> f32 {
    return clamp(0.5 - distance, 0.0, 1.0);
}

fn over(top: vec4<f32>, below: vec4<f32>) -> vec4<f32> {
    return top + below * (1.0 - top.a);
}

fn premultiplied(color: vec4<f32>, amount: f32) -> vec4<f32> {
    let a = color.a * amount;
    return vec4<f32>(color.rgb * a, a);
}

fn cased(quad: Quad, local: vec2<f32>, soft: f32, shown: vec4<f32>) -> vec4<f32> {
    if quad.case_color == 0u {
        return shown;
    }
    let half = quad.rect.zw * 0.5;
    let grown = half + vec2<f32>(quad.case_grow);
    let rounding = quad.radius + quad.case_grow;
    let color = unpack4x8unorm(quad.case_color);
    var under = color.a * coverage(rounded_box(local - quad.case_offset, grown, rounding) / soft);
    if doubled(quad) {
        let again = color.a * coverage(rounded_box(local, grown, rounding) / soft);
        under = under + again - under * again;
    }
    var below = vec4<f32>(color.rgb * under, under);
    if quad.rim != 0u {
        let rim = unpack4x8unorm(quad.rim);
        let reach = half + vec2<f32>(quad.rim_grow);
        let ring = rim.a * coverage(rounded_box(local, reach, quad.radius + quad.rim_grow) / soft);
        below = over(vec4<f32>(rim.rgb * ring, ring), below);
    }
    return over(shown, below);
}

fn plate(quad: Quad, local: vec2<f32>) -> vec4<f32> {
    let half = quad.rect.zw * 0.5;
    let inside = coverage(rounded_box(local, half, quad.radius) / quad.scale);
    var shown = premultiplied(unpack4x8unorm(quad.fill), inside);
    if quad.stroke_width > 0.0 {
        let inset = quad.stroke_inset;
        let outer = coverage(rounded_box(local, half - vec2<f32>(inset), quad.radius - inset));
        let deep = inset + quad.stroke_width;
        let inner = coverage(rounded_box(local, half - vec2<f32>(deep), quad.radius - deep));
        shown = over(premultiplied(unpack4x8unorm(quad.edge), max(outer - inner, 0.0)), shown);
    }
    shown = cased(quad, local, quad.scale, shown);
    if quad.uv.x > 0.0 {
        let turn = fract(atan2(local.x, -local.y) / 6.2831853 + 1.0);
        shown = shown * select(0.0, 1.0, turn <= 1.0 - quad.uv.x);
    }
    return shown;
}

fn encoded(linear: vec3<f32>) -> vec3<f32> {
    let low = linear * 12.92;
    let high = 1.055 * pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, linear <= vec3<f32>(0.0031308));
}

fn sampled(picture: texture_2d<f32>, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let texels = vec2<f32>(textureDimensions(picture));
    return textureSampleLevel(picture, screen_sampler, uv / texels, lod);
}

fn picture_of(slot: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    switch slot {
        case 0u: { return sampled(screen_picture_0, uv, lod); }
        case 1u: { return sampled(screen_picture_1, uv, lod); }
        case 2u: { return sampled(screen_picture_2, uv, lod); }
        case 3u: { return sampled(screen_picture_3, uv, lod); }
        case 4u: { return sampled(screen_picture_4, uv, lod); }
        case 5u: { return sampled(screen_picture_5, uv, lod); }
        case 6u: { return sampled(screen_picture_6, uv, lod); }
        case 7u: { return sampled(screen_picture_7, uv, lod); }
        case 8u: { return sampled(screen_picture_8, uv, lod); }
        default: { return sampled(screen_picture_9, uv, lod); }
    }
}

fn glyph_of(slot: u32, uv: vec2<f32>, lod: f32) -> f32 {
    switch slot {
        case 0u: { return sampled(screen_glyphs_0, uv, lod).r; }
        default: { return sampled(screen_glyphs_1, uv, lod).r; }
    }
}

fn lod_of(scale: f32) -> f32 {
    return max(0.0, -log2(max(scale, 1.0e-4)));
}

@fragment
fn fs_screen(corner: Corner) -> @location(0) vec4<f32> {
    let quad = screen_quads.quads[corner.index];
    if quad.mode == PLATE {
        return plate(quad, corner.local);
    }
    let tint = unpack4x8unorm(quad.fill);
    let lod = lod_of(quad.scale);
    if quad.mode == IMAGE {
        let picture = picture_of(quad.edge, corner.uv, lod);
        let shown = encoded(picture.rgb);
        let grey = dot(shown, vec3<f32>(0.2126, 0.7152, 0.0722));
        let tinted = vec4<f32>(mix(shown, vec3<f32>(grey), quad.stroke_width), picture.a) * tint;
        if !framed_image(quad) {
            return premultiplied(tinted, 1.0);
        }
        let inside = coverage(rounded_box(corner.local, quad.rect.zw * 0.5, quad.radius));
        return cased(quad, corner.local, 1.0, premultiplied(tinted, inside));
    }
    if quad.mode == ICON {
        let texels = vec2<f32>(textureDimensions(screen_icons));
        let alpha = textureSampleLevel(screen_icons, screen_sampler, corner.uv / texels, lod).a;
        return premultiplied(tint, alpha);
    }
    let distance = glyph_of(quad.edge, corner.uv, lod);
    let spread = quad.radius;
    let pixels = (distance - 0.5) * 2.0 * spread * quad.scale + quad.stroke_width;
    return premultiplied(tint, clamp(pixels / (1.0 + quad.stroke_inset) + 0.5, 0.0, 1.0));
}
