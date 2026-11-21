import package::fluid::common::{FrameUniform, grid_to_world, unpack_meta};

@group(1) @binding(0) var<uniform> splat_frame: FrameUniform;
@group(1) @binding(1) var<storage, read> splat_particles: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> surface_indices: array<u32>;

struct DepthSplatOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) center_z_rad: vec2<f32>,
}

struct ThickSplatOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) rad_meta: vec4<f32>,
}

struct DepthOut {
    @builtin(frag_depth) hw_depth: f32,
    @location(0) linear_depth: f32,
}

fn quad_corner(vertex: u32) -> vec2<f32> {
    let x = f32((50u >> vertex) & 1u) * 2.0 - 1.0;
    let y = f32((44u >> vertex) & 1u) * 2.0 - 1.0;
    return vec2<f32>(x, y);
}

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> DepthSplatOut {
    let pid = surface_indices[instance];
    let p = splat_particles[pid];
    let pm = unpack_meta(p.w);
    let world = grid_to_world(p.xyz, splat_frame.basin, splat_frame.grid_dims);
    let center_view = (splat_frame.view * vec4<f32>(world, 1.0)).xyz;
    let density_scale = mix(0.88, 1.14, smoothstep(0.25, 1.15, pm.z));
    let radius = splat_frame.splat.x * density_scale;

    var out: DepthSplatOut;
    if center_view.z > -0.18 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.center_z_rad = vec2<f32>(0.0);
        return out;
    }

    let c_clip = splat_frame.proj * vec4<f32>(center_view, 1.0);
    let inv_w = 1.0 / max(abs(c_clip.w), 1.0e-5);
    if abs(c_clip.x * inv_w) > 1.25 || abs(c_clip.y * inv_w) > 1.25 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.center_z_rad = vec2<f32>(0.0);
        return out;
    }

    let corner = quad_corner(vertex);
    let pad = 1.01;
    let view_pos = center_view + vec3<f32>(corner * (radius * pad), 0.0);
    out.clip = splat_frame.proj * vec4<f32>(view_pos, 1.0);
    out.uv = corner * pad;
    out.center_z_rad = vec2<f32>(center_view.z, radius);
    return out;
}

@vertex
fn vs_thick(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ThickSplatOut {
    let p = splat_particles[instance * 12u];
    let pm = unpack_meta(p.w);
    let world = grid_to_world(p.xyz, splat_frame.basin, splat_frame.grid_dims);
    let center_view = (splat_frame.view * vec4<f32>(world, 1.0)).xyz;
    let density_scale = mix(0.86, 1.12, smoothstep(0.25, 1.15, pm.z));
    let radius = splat_frame.splat.x * density_scale * 1.62;

    var out: ThickSplatOut;
    if center_view.z > -0.18 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.rad_meta = vec4<f32>(0.0);
        return out;
    }

    let c_clip = splat_frame.proj * vec4<f32>(center_view, 1.0);
    let inv_w = 1.0 / max(abs(c_clip.w), 1.0e-5);
    if abs(c_clip.x * inv_w) > 1.25 || abs(c_clip.y * inv_w) > 1.25 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(2.0);
        out.rad_meta = vec4<f32>(0.0);
        return out;
    }

    let corner = quad_corner(vertex);
    let pad = 1.01;
    let view_pos = center_view + vec3<f32>(corner * (radius * pad), 0.0);
    let aux_g = select(pm.x, pm.y, splat_frame.style.x > 1.5);
    out.clip = splat_frame.proj * vec4<f32>(view_pos, 1.0);
    out.uv = corner * pad;
    out.rad_meta = vec4<f32>(radius, aux_g, pm.w, 0.0);
    return out;
}

@fragment
fn fs_depth_only(in: DepthSplatOut) -> DepthOut {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let nz = sqrt(1.0 - r2);
    let z_eye = -in.center_z_rad.x - nz * in.center_z_rad.y;
    if z_eye <= splat_frame.splat.z {
        discard;
    }
    let hw_depth = clamp(-splat_frame.proj[2][2] + splat_frame.proj[3][2] / z_eye, 0.0, 1.0);
    return DepthOut(hw_depth, z_eye);
}

@fragment
fn fs_thickness(in: ThickSplatOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let nz = sqrt(1.0 - r2);
    let smooth_w = nz * nz * (3.0 - 2.0 * nz);
    let count_norm = 250000.0 / max(splat_frame.style.w, 1000.0);
    let scale = pow(count_norm, 0.66) * 2.50;
    let w = smooth_w * 0.16 * scale;
    let thick = nz * in.rad_meta.x * 0.56 * scale;
    return vec4<f32>(thick, in.rad_meta.y * w, in.rad_meta.z * w, w);
}
