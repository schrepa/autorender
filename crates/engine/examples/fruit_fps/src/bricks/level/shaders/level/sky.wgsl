import package::engine::view::camera;
import package::level::light::level_light;

const SKY_INV_TURN: f32 = 0.15915494;

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
    return exp(-max(toward.y, 0.0) * level_light.below.w);
}

fn sky_under(toward: vec3<f32>) -> f32 {
    return saturate(-toward.y * level_light.below.w);
}

fn sky_gradient_with(haze: f32, under: f32) -> vec3<f32> {
    let color = mix(level_light.zenith.rgb, level_light.horizon.rgb, haze);
    return mix(color, level_light.below.rgb, under);
}

fn sky_gradient(toward: vec3<f32>) -> vec3<f32> {
    return sky_gradient_with(sky_haze(toward), sky_under(toward));
}

fn sky_pictured_with(map: texture_2d<f32>, map_sampler: sampler, toward: vec3<f32>, under: f32)
        -> vec3<f32> {
    let shape = level_light.picture;
    let elevation = degrees(asin(clamp(toward.y, -1.0, 1.0)));
    let across = (0.5 + atan2(toward.x, -toward.z) * SKY_INV_TURN) * shape.w;
    let down = saturate((shape.y - elevation) * level_light.glow.w);
    let picture = textureSampleLevel(map, map_sampler, vec2<f32>(across, down), 0.0).rgb;
    let haze = sky_haze(toward);
    var color = mix(picture, level_light.horizon.rgb, haze * level_light.reflect.y);
    let top = smoothstep(shape.y - shape.z, shape.y, elevation);
    if top > 0.0 {
        color = mix(color, sky_gradient_with(haze, under), top);
    }
    let bottom = 1.0 - smoothstep(shape.x - shape.z, shape.x, elevation);
    return mix(color, level_light.below.rgb, bottom);
}

fn sky_pictured(map: texture_2d<f32>, map_sampler: sampler, toward: vec3<f32>) -> vec3<f32> {
    return sky_pictured_with(map, map_sampler, toward, sky_under(toward));
}

fn sky_sunned_with(color: vec3<f32>, toward: vec3<f32>, under: f32) -> vec3<f32> {
    if under >= 1.0 {
        return color;
    }
    let facing = dot(toward, level_light.to_sun.xyz);
    if facing <= 0.0 {
        return color;
    }
    let fog = level_light.fog;
    let glow = level_light.glow;
    let disc = select(0.0, smoothstep(fog.w, fog.z, facing) * glow.x, facing > fog.w);
    let halo = pow(facing, glow.y) * glow.z;
    return color + level_light.sun.rgb * ((disc + halo) * (1.0 - under));
}

@vertex fn vs_sky(@builtin(vertex_index) vertex: u32) -> SkyCorner {
    return sky_corner(vertex);
}

@fragment fn fs_sky(corner: SkyCorner) -> @location(0) vec4<f32> {
    let toward = normalize(corner.direction);
    let under = sky_under(toward);
    let color = sky_gradient_with(sky_haze(toward), under);
    return vec4<f32>(sky_sunned_with(color, toward, under), 1.0);
}
