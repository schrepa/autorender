struct Post {
    inverse_view_projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    size: vec4<f32>,
    extent: vec4<f32>,
    occlusion: vec4<f32>,
    bloom: vec4<f32>,
    reflections: vec4<f32>,
    haze: vec4<f32>,
    previous_view_projection: mat4x4<f32>,
    previous_eye: vec4<f32>,
    sky: vec4<f32>,
    detail: vec4<f32>,
}

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var scene: texture_2d<f32>;
@group(0) @binding(3) var glow: texture_2d<f32>;
@group(0) @binding(5) var occlusion: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;
@group(0) @binding(8) var reflections: texture_2d<f32>;

fn drawn_uv(pixel: vec2<f32>, settings: Post) -> vec2<f32> {
    return clamp(pixel, vec2<f32>(0.5), settings.size.xy - 0.5) * settings.extent.zw;
}

fn drawn_texels(size: vec2<f32>, settings: Post) -> vec2<f32> {
    let share = max(floor(size * settings.size.xy * settings.extent.zw), vec2<f32>(1.0));
    return select(share, size, settings.size.xy == settings.extent.xy);
}

fn inverse_distance_at(depth: texture_depth_2d, pixel: vec2<i32>, settings: Post) -> f32 {
    let value = textureLoad(depth, pixel, 0);
    if value <= 0.0 {
        return 0.0;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * settings.size.zw;
    let point = settings.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, value, 1.0);
    return 1.0 / max(length(point.xyz / point.w - settings.eye.xyz), 1.0e-4);
}

fn occlusion_at(occlusion: texture_2d<f32>, pixel: vec2<i32>, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return textureLoad(occlusion, pixel, 0).x;
}

fn reflected(color: vec3<f32>, reflections: texture_2d<f32>, linear: sampler,
        uv: vec2<f32>, settings: Post) -> vec3<f32> {
    if settings.reflections.z == 0.0 {
        return color;
    }
    let reflection = textureSampleLevel(reflections, linear,
        drawn_uv(uv * settings.size.xy, settings), 0.0);
    let strength = settings.reflections.y;
    return min(color * (1.0 - saturate(reflection.a * strength)) + reflection.rgb * strength,
        vec3<f32>(1.0));
}

fn effect_shade(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return 1.0 - ambient * (1.0 - visible);
}
