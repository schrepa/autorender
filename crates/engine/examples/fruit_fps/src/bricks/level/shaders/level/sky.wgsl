import package::engine::view::{camera, stable_normalize};
import package::level::light::level_light;

struct SkyCorner {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
}

fn sky_axis(index: u32) -> vec3<f32> {
    return vec3<f32>(select(0.0, 1.0, index == 0u), select(0.0, 1.0, index == 1u),
        select(0.0, 1.0, index == 2u));
}

@vertex fn vs_sky(@builtin(vertex_index) vertex: u32) -> SkyCorner {
    let face = vertex / 6u;
    let corner = vertex % 6u;
    let axis = face / 2u;
    let sign = 1.0 - 2.0 * f32(face % 2u);
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u)) * 2.0 - 1.0;
    let across = sky_axis((axis + 1u) % 3u);
    let up = sky_axis((axis + 2u) % 3u);
    let direction = sky_axis(axis) * sign + across * unit.x + up * unit.y;
    let clip = camera.view_projection * vec4<f32>(direction, 0.0);
    var out: SkyCorner;
    out.clip_position = vec4<f32>(clip.xy, 0.0, clip.w);
    out.direction = direction;
    return out;
}

@fragment fn fs_sky(corner: SkyCorner) -> @location(0) vec4<f32> {
    let toward = stable_normalize(corner.direction);
    let height = max(level_light.zenith.w, 1.0e-3);
    let haze = exp(-max(toward.y, 0.0) / height);
    var color = mix(level_light.zenith.rgb, level_light.horizon.rgb, haze);
    let under = saturate(-toward.y / height);
    color = mix(color, level_light.below.rgb, under);
    let facing = dot(toward, level_light.to_sun.xyz);
    let fog = level_light.fog;
    let glow = level_light.glow;
    let disc = smoothstep(fog.w, fog.z, facing) * glow.x;
    let halo = pow(max(facing, 0.0), glow.y) * glow.z;
    color += level_light.sun.rgb * (disc + halo) * (1.0 - under);
    return vec4<f32>(color, 1.0);
}
