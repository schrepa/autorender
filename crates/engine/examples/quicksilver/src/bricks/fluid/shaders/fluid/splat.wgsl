import package::fluid::common::{FrameUniform, Particle, grid_to_world};

@group(1) @binding(0) var<uniform> splat_frame: FrameUniform;
@group(1) @binding(1) var<storage, read> splat_particles: array<Particle>;

struct SplatOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) center_view: vec3<f32>,
    @location(2) @interpolate(flat) radius: f32,
    @location(3) @interpolate(flat) aux: vec4<f32>,
}

struct DepthAuxOut {
    @builtin(frag_depth) hw_depth: f32,
    @location(0) linear_depth: f32,
    @location(1) aux: vec4<f32>,
}

fn quad_corner(vertex: u32) -> vec2<f32> {
    let x = f32((50u >> vertex) & 1u) * 2.0 - 1.0;
    let y = f32((44u >> vertex) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

fn build_splat_vertex(vertex: u32, p: Particle, radius_mult: f32) -> SplatOut {
    let world = grid_to_world(p.pos_foam.xyz, splat_frame.basin, splat_frame.grid_dims);
    let center_view = (splat_frame.view * vec4<f32>(world, 1.0)).xyz;
    let speed = length(p.vel_dens.xyz);
    let ratio = p.extra.w;
    let density_scale = mix(0.86, 1.12, smoothstep(0.25, 1.15, ratio));
    let radius = splat_frame.splat.x * density_scale * radius_mult;

    var out: SplatOut;
    if center_view.z > -0.18 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.center_view = center_view;
        out.radius = radius;
        out.aux = vec4<f32>(0.0);
        return out;
    }

    let c_clip = splat_frame.proj * vec4<f32>(center_view, 1.0);
    let inv_w = 1.0 / max(abs(c_clip.w), 1.0e-5);
    if abs(c_clip.x * inv_w) > 1.25 || abs(c_clip.y * inv_w) > 1.25 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.center_view = center_view;
        out.radius = radius;
        out.aux = vec4<f32>(0.0);
        return out;
    }

    let corner = quad_corner(vertex);
    let pad = 1.05;
    let view_pos = center_view + vec3<f32>(corner * (radius * pad), 0.0);
    out.clip = splat_frame.proj * vec4<f32>(view_pos, 1.0);
    out.uv = corner * pad;
    out.center_view = center_view;
    out.radius = radius;
    out.aux = vec4<f32>(speed, p.pos_foam.w, p.extra.y, ratio);
    return out;
}

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SplatOut {
    let p = splat_particles[instance];
    if p.extra.z < 0.5 {
        var culled: SplatOut;
        culled.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        culled.uv = vec2<f32>(2.0);
        culled.center_view = vec3<f32>(0.0);
        culled.radius = 0.0;
        culled.aux = vec4<f32>(0.0);
        return culled;
    }
    return build_splat_vertex(vertex, p, 1.0);
}

@vertex
fn vs_thick(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SplatOut {
    return build_splat_vertex(vertex, splat_particles[instance * 4u], 1.48);
}

@fragment
fn fs_depth_aux(in: SplatOut) -> DepthAuxOut {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let nz = sqrt(max(1.0 - r2, 0.0));
    let surf_view = in.center_view + vec3<f32>(in.uv * in.radius, nz * in.radius);
    let z_eye = -surf_view.z;
    if z_eye <= splat_frame.splat.z {
        discard;
    }
    let clip = splat_frame.proj * vec4<f32>(surf_view, 1.0);
    let hw_depth = clamp(clip.z / max(clip.w, 1.0e-5), 0.0, 1.0);

    var out: DepthAuxOut;
    out.hw_depth = hw_depth;
    out.linear_depth = z_eye;
    out.aux = vec4<f32>(in.aux.x, in.aux.y, in.aux.z, 1.0);
    return out;
}

@fragment
fn fs_thickness(in: SplatOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let nz = sqrt(max(1.0 - r2, 0.0));
    let smooth_w = nz * nz * (3.0 - 2.0 * nz);
    let count_norm = 250000.0 / max(splat_frame.style.w, 1000.0);
    let scale = pow(count_norm, 0.66);
    let w = smooth_w * 0.16 * scale;
    let thick = nz * in.radius * 0.56 * scale;
    return vec4<f32>(thick, in.aux.y * w, in.aux.x * w, w);
}
