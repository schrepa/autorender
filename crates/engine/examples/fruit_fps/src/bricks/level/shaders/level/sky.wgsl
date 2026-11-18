import package::engine::view::{camera, stable_normalize};
import package::level::light::level_light;

const SKY_TURN: f32 = 6.28318531;

struct SkyCorner {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
}

fn sky_axis(index: u32) -> vec3<f32> {
    return vec3<f32>(select(0.0, 1.0, index == 0u), select(0.0, 1.0, index == 1u),
        select(0.0, 1.0, index == 2u));
}

fn sky_corner(vertex: u32) -> SkyCorner {
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

fn sky_haze(toward: vec3<f32>) -> f32 {
    return exp(-max(toward.y, 0.0) / max(level_light.zenith.w, 1.0e-3));
}

fn sky_under(toward: vec3<f32>) -> f32 {
    return saturate(-toward.y / max(level_light.zenith.w, 1.0e-3));
}

fn sky_gradient(toward: vec3<f32>) -> vec3<f32> {
    let color = mix(level_light.zenith.rgb, level_light.horizon.rgb, sky_haze(toward));
    return mix(color, level_light.below.rgb, sky_under(toward));
}

fn sky_pictured(map: texture_2d<f32>, map_sampler: sampler, toward: vec3<f32>) -> vec3<f32> {
    let shape = level_light.picture;
    let elevation = degrees(asin(clamp(toward.y, -1.0, 1.0)));
    let across = (0.5 + atan2(toward.x, -toward.z) / SKY_TURN) * shape.w;
    let down = saturate((shape.y - elevation) / (shape.y - shape.x));
    let picture = textureSampleLevel(map, map_sampler, vec2<f32>(across, down), 0.0).rgb;
    let hazed = mix(picture, level_light.horizon.rgb, sky_haze(toward) * level_light.reflect.y);
    let top = smoothstep(shape.y - shape.z, shape.y, elevation);
    let bottom = 1.0 - smoothstep(shape.x - shape.z, shape.x, elevation);
    return mix(mix(hazed, sky_gradient(toward), top), level_light.below.rgb, bottom);
}

fn sky_sunned(color: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let facing = dot(toward, level_light.to_sun.xyz);
    let fog = level_light.fog;
    let glow = level_light.glow;
    let disc = smoothstep(fog.w, fog.z, facing) * glow.x;
    let halo = pow(max(facing, 0.0), glow.y) * glow.z;
    return color + level_light.sun.rgb * (disc + halo) * (1.0 - sky_under(toward));
}

@vertex fn vs_sky(@builtin(vertex_index) vertex: u32) -> SkyCorner {
    return sky_corner(vertex);
}

@fragment fn fs_sky(corner: SkyCorner) -> @location(0) vec4<f32> {
    let toward = stable_normalize(corner.direction);
    return vec4<f32>(sky_sunned(sky_gradient(toward), toward), 1.0);
}
