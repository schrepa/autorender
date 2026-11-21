import package::engine::view::view_clip;
import package::generated::world::WorldUniforms;
import package::generated::world::VisibleParticle;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> vis_particles: array<VisibleParticle>;

struct PartOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) kind: f32,
}

@vertex
fn vs_particle(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> PartOut {
    let idx = min(iid, arrayLength(&vis_particles) - 1u);
    let rec = vis_particles[idx];
    let pos = rec.pos_size.xyz;
    let size = rec.pos_size.w;

    let rg = unpack2x16float(rec.color_spin.x);
    let ba = unpack2x16float(rec.color_spin.y);
    let cs_sn = unpack2x16snorm(rec.color_spin.z);

    let corners = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(-1.7320508, -1.0),
        vec2<f32>(1.7320508, -1.0),
    );
    let corner = corners[vid];
    let rot_c = vec2<f32>(
        corner.x * cs_sn.x - corner.y * cs_sn.y,
        corner.x * cs_sn.y + corner.y * cs_sn.x,
    );
    let offset = (world.camera_right.xyz * rot_c.x + world.camera_up.xyz * rot_c.y) * size;

    var out: PartOut;
    out.clip_pos = view_clip(pos + offset);
    out.uv = corner;
    out.color = vec4<f32>(rg.x, rg.y, ba.x, ba.y);
    out.kind = f32(rec.color_spin.w);
    return out;
}

@fragment
fn fs_particle(in: PartOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.uv, in.uv);
    if d2 > 1.0 {
        discard;
    }
    if in.kind < 0.5 {
        let core = exp(-d2 * 12.0);
        let halo = exp(-d2 * 3.6) * 0.42;
        let shape = core + halo;
        return vec4<f32>(in.color.rgb * (1.0 + core * 1.2), clamp(shape * in.color.a, 0.0, 1.0));
    }
    let p = vec2<f32>(in.uv.x * 1.15, in.uv.y * 0.82);
    let notch = smoothstep(0.0, 0.25, abs(p.x)) * 0.18 * step(0.0, p.y);
    let petal = 1.0 - smoothstep(0.55, 0.92, length(p) + notch);
    if petal < 0.05 {
        discard;
    }
    return vec4<f32>(in.color.rgb, petal * in.color.a);
}
