import package::fluid::common::{UiBuffer, UiQuad};

@group(1) @binding(0) var<storage, read> ui_data: UiBuffer;
@group(1) @binding(1) var ui_font: texture_2d<f32>;
@group(1) @binding(2) var ui_sampler: sampler;

struct UiOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) idx: u32,
}

fn ui_corner(vertex: u32) -> vec2<f32> {
    let x = f32((50u >> vertex) & 1u);
    let y = f32((44u >> vertex) & 1u);
    return vec2<f32>(x, y);
}

fn sdf_rrect(p: vec2<f32>, half_s: vec2<f32>, r: f32) -> f32 {
    let rr = clamp(r, 0.0, min(half_s.x, half_s.y));
    let q = abs(p) - half_s + vec2<f32>(rr);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

@vertex
fn vs_ui(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> UiOut {
    let q = ui_data.quads[instance];
    let c = ui_corner(vertex);
    let pad = select(1.5, 10.0, q.params.x < 0.5);
    let size = q.rect.zw;
    let local = mix(-size * 0.5 - pad, size * 0.5 + pad, c);
    let at = q.rect.xy + size * 0.5 + local;
    let canvas = max(ui_data.header.xy, vec2<f32>(1.0));

    var out: UiOut;
    out.clip = vec4<f32>(at.x / canvas.x * 2.0 - 1.0, 1.0 - at.y / canvas.y * 2.0, 0.0, 1.0);
    out.local = local;
    out.uv = q.uv.xy + ((local / max(size, vec2<f32>(1.0e-4))) + 0.5) * q.uv.zw;
    out.idx = instance;
    return out;
}

@fragment
fn fs_ui(in: UiOut) -> @location(0) vec4<f32> {
    let sdf = textureSampleLevel(ui_font, ui_sampler, in.uv, 0.0).r;
    let fw = max(fwidth(sdf), 0.025);
    let q = ui_data.quads[in.idx];
    let kind = q.params.x;
    let half_s = q.rect.zw * 0.5;

    if kind > 2.5 && kind < 3.5 {
        let alpha = smoothstep(0.5 - fw * 0.72, 0.5 + fw * 0.72, sdf) * q.color.a;
        if alpha <= 0.002 {
            discard;
        }
        return vec4<f32>(q.color.rgb * alpha, alpha);
    }

    let d = sdf_rrect(in.local, half_s, q.params.y);

    if kind < 0.5 {
        let shadow_d = sdf_rrect(in.local - vec2<f32>(0.0, 3.5), half_s + 1.0, q.params.y + 1.0);
        let shadow = exp(-max(shadow_d, 0.0) * 0.28) * 0.44;
        let body = clamp(0.5 - d, 0.0, 1.0);
        let grad_t = clamp((in.local.y / max(half_s.y, 1.0)) * 0.5 + 0.5, 0.0, 1.0);
        let top_sheen = mix(vec3<f32>(0.11, 0.13, 0.17), q.color.rgb, smoothstep(0.0, 0.45, grad_t));
        let border = clamp(0.5 - abs(d + 0.65), 0.0, 1.0) * mix(0.28, 0.09, grad_t);
        let fill_a = body * q.color.a;
        let rim_a = border * body;
        let out_rgb = top_sheen * fill_a + vec3<f32>(0.92, 0.96, 1.0) * rim_a;
        let out_a = max(fill_a, rim_a);
        let below_a = shadow * (1.0 - body);
        return vec4<f32>(out_rgb, clamp(out_a + below_a, 0.0, 1.0));
    }

    if kind < 1.5 {
        let body = clamp(0.5 - d, 0.0, 1.0);
        if body <= 0.002 {
            discard;
        }
        let glow = q.params.z;
        let grad_t = clamp((in.local.y / max(half_s.y, 1.0)) * 0.5 + 0.5, 0.0, 1.0);
        let tint = mix(q.color.rgb * 1.18, q.color.rgb * 0.88, grad_t);
        let border = clamp(0.5 - abs(d + 0.6), 0.0, 1.0) * (0.12 + 0.30 * glow);
        let a = body * q.color.a;
        let rgb = tint * a + vec3<f32>(0.92, 0.96, 1.0) * (border * body);
        return vec4<f32>(rgb, clamp(a + border * body * 0.6, 0.0, 1.0));
    }

    if kind < 2.5 {
        let body = clamp(0.5 - d, 0.0, 1.0);
        if body <= 0.002 {
            discard;
        }
        let n = in.local / max(half_s.x, 1.0);
        let spec = exp(-dot(n - vec2<f32>(-0.32, -0.34), n - vec2<f32>(-0.32, -0.34)) * 7.5);
        let shade = 0.72 + 0.28 * (1.0 - n.y);
        let col = q.color.rgb * shade + vec3<f32>(0.9, 0.95, 1.0) * spec * 0.85;
        let a = body * (0.55 + 0.45 * q.params.z);
        return vec4<f32>(col * a, a);
    }

    let body = clamp(0.5 - d, 0.0, 1.0);
    let a = body * q.color.a;
    if a <= 0.002 {
        discard;
    }
    return vec4<f32>(q.color.rgb * a, a);
}
