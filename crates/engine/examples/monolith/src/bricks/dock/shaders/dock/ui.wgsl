import package::generated::dock::UiHeader;
import package::generated::dock::UiQuad;

@group(2) @binding(0) var<uniform> ui_header: UiHeader;
@group(2) @binding(1) var<storage, read> ui_quads: array<UiQuad>;

struct UiOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_px: vec2<f32>,
    @location(1) size_px: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
}

@vertex
fn vs_ui(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> UiOut {
    let q = ui_quads[iid];
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let c = corners[vid];
    let px = q.rect.xy + c * q.rect.zw;
    let screen = max(ui_header.screen.xy, vec2<f32>(1.0));
    let ndc = vec2<f32>(
        (px.x / screen.x) * 2.0 - 1.0,
        1.0 - (px.y / screen.y) * 2.0,
    );
    var out: UiOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_px = c * q.rect.zw;
    out.size_px = q.rect.zw;
    out.color = q.color;
    out.border = q.border;
    out.params = q.params;
    return out;
}

fn glyph_bit(lo18: u32, hi17: u32, col: i32, row: i32) -> f32 {
    if col < 0 || col >= 5 || row < 0 || row >= 7 {
        return 0.0;
    }
    let idx = u32(row * 5 + col);
    if idx < 18u {
        return f32((lo18 >> idx) & 1u);
    }
    return f32((hi17 >> (idx - 18u)) & 1u);
}

fn box_dist(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let d = abs(p) - half;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

@fragment
fn fs_ui(in: UiOut) -> @location(0) vec4<f32> {
    if in.params.z > 0.5 {
        let lo18 = u32(round(in.params.x));
        let hi17 = u32(round(in.params.y));
        let uv = in.local_px / max(in.size_px, vec2<f32>(1.0));
        let g = uv * vec2<f32>(5.0, 7.0);
        let cell = vec2<i32>(floor(g));
        var min_d = 10.0;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cx = cell.x + dx;
                let cy = cell.y + dy;
                if glyph_bit(lo18, hi17, cx, cy) > 0.5 {
                    let center = vec2<f32>(f32(cx) + 0.5, f32(cy) + 0.5);
                    let d = box_dist(g - center, vec2<f32>(0.36, 0.34)) - 0.11;
                    min_d = min(min_d, d);
                }
            }
        }
        let aa = max(length(vec2<f32>(5.0, 7.0) / max(in.size_px, vec2<f32>(1.0))) * 0.52, 0.12);
        let cov = 1.0 - smoothstep(-aa, aa, min_d);
        if cov <= 0.02 {
            discard;
        }
        return vec4<f32>(in.color.rgb, in.color.a * cov);
    }

    let half = in.size_px * 0.5;
    let p = in.local_px - half;
    let r = min(in.params.x, min(half.x, half.y));
    let q = abs(p) - (half - vec2<f32>(r));
    let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    let aa = 0.85;
    let fill_a = 1.0 - smoothstep(-aa, aa, dist);
    if fill_a <= 0.002 {
        discard;
    }

    let bw = max(in.params.y, 1.0);
    let border_t = smoothstep(-(bw + aa), -bw + aa, dist) * fill_a;
    let top_sheen = smoothstep(0.0, -half.y, p.y) * 0.06;
    var rgb = in.color.rgb + vec3<f32>(top_sheen);
    var alpha = in.color.a * fill_a;
    if in.border.a > 0.002 {
        rgb = mix(rgb, in.border.rgb, border_t * in.border.a);
        alpha = max(alpha, border_t * in.border.a);
    }
    return vec4<f32>(rgb, alpha);
}
