import package::engine::drawn::{drawn_float_count, drawn_floats, drawn_tint};
import package::engine::view::{camera, stable_normalize};
import package::generated::figures::FigureLight;
import package::figures::material::{figure_field_map, figure_field_sampler, material_texels};
import package::play::light::light_field;

@group(2) @binding(0) var<uniform> figure_light: FigureLight;

const FIGURE_PI: f32 = 3.14159265;
const DIELECTRIC_REFLECTANCE: f32 = 0.04;
const SMALLEST_ALPHA: f32 = 0.002;
const SMALLEST_COSINE: f32 = 1.0e-4;

struct FigureCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) slot: u32,
    @location(4) field: vec4<f32>,
}

fn figure_pow5(x: f32) -> f32 {
    let squared = x * x;
    return squared * squared * x;
}

fn figure_field(world: vec3<f32>, slot: u32) -> vec4<f32> {
    let shape = figure_light.field_shape;
    if shape.z == 0.0 {
        return vec4<f32>(0.0);
    }
    let at_camera = drawn_float_count(slot) >= 8u && drawn_floats(slot, 1u).w > 0.5;
    let at = select(world, camera.position.xyz, at_camera);
    let raw = light_field(figure_field_map, figure_field_sampler, at,
        figure_light.direction.xyz, figure_light.field, shape);
    let open = max(figure_light.sky.rgb * figure_light.sky.w, vec3<f32>(1.0e-4));
    return vec4<f32>(raw.rgb / open, raw.a);
}

fn deflected_normal(normal: vec3<f32>, dp_dx: vec3<f32>, dp_dy: vec3<f32>, uv_dx: vec2<f32>,
        uv_dy: vec2<f32>, sampled: vec3<f32>) -> vec3<f32> {
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * frame;
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn hemisphere(direction: vec3<f32>) -> vec3<f32> {
    return mix(figure_light.ground.rgb * figure_light.ground.w,
        figure_light.sky.rgb * figure_light.sky.w, direction.y * 0.5 + 0.5);
}

fn scene_fogged(color: vec3<f32>, distance: f32) -> vec3<f32> {
    let fog = figure_light.fog;
    let reach = distance - fog.x;
    if figure_light.horizon.w <= 0.0 || reach <= 0.0 {
        return color;
    }
    let amount = figure_light.horizon.w * (1.0 - exp(-reach * fog.y));
    return mix(color, figure_light.horizon.rgb, amount);
}

fn lit_color(base_color: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, raw_nv: f32,
        roughness: f32, metallic: f32, field: vec4<f32>) -> vec3<f32> {
    let to_sun = figure_light.direction.xyz;
    let raw_nl = dot(normal, to_sun);
    let n_l = max(raw_nl, 0.0);
    let n_v = max(raw_nv, SMALLEST_COSINE);
    let reflectance = mix(vec3<f32>(DIELECTRIC_REFLECTANCE), base_color, metallic);
    let diffuse = base_color * (1.0 - metallic);
    let wrap = figure_light.sun.w;
    let wrapped = saturate((raw_nl + wrap) * figure_light.fog.z);
    let has_field = figure_light.field_shape.z != 0.0;
    let sun_scale = select(1.0, field.a, has_field);
    var sun = vec3<f32>(0.0);
    if sun_scale > 0.0 && wrapped > 0.0 {
        var direct = diffuse * wrapped;
        if n_l > 0.0 {
            let half_way = normalize(to_sun + view);
            let n_h = max(dot(normal, half_way), 0.0);
            let alpha = max(roughness * roughness, SMALLEST_ALPHA);
            let alpha_2 = alpha * alpha;
            let lobe = n_h * n_h * (alpha_2 - 1.0) + 1.0;
            let distribution = alpha_2 / (FIGURE_PI * lobe * lobe);
            let visibility = 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
            let grazing = figure_pow5(1.0 - max(dot(to_sun, half_way), 0.0));
            let fresnel = reflectance + (1.0 - reflectance) * grazing;
            direct += fresnel * (FIGURE_PI * distribution * visibility * n_l);
        }
        sun = direct * (figure_light.sun.rgb * sun_scale);
    }
    var ambient = hemisphere(normal) * diffuse + hemisphere(reflect(-view, normal)) * reflectance;
    if has_field {
        let share = max(field.rgb, vec3<f32>(figure_light.field_shape.w));
        ambient *= share;
    }
    return sun + ambient;
}

@fragment fn fs_figures(corner: FigureCorner, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let texels = material_texels(corner.uv);
    var base_color = texels.base_color;
    if drawn_float_count(corner.slot) >= 7u {
        let trim = saturate(drawn_floats(corner.slot, 0u).w * texels.trim);
        base_color = mix(base_color, drawn_floats(corner.slot, 1u).xyz, trim);
    }
    let facing = stable_normalize(select(-corner.normal, corner.normal, front));
    let normal = deflected_normal(facing, dpdx(corner.world), dpdy(corner.world),
        dpdx(corner.uv), dpdy(corner.uv), texels.normal);
    let to_eye = camera.position.xyz - corner.world;
    let eye_sq = dot(to_eye, to_eye);
    let inv_eye = inverseSqrt(max(eye_sq, 1.0e-20));
    let view = select(vec3<f32>(0.0, 1.0, 0.0), to_eye * inv_eye, eye_sq > 1.0e-20);
    let raw_nv = dot(normal, view);
    var shown = base_color;
    if figure_light.direction.w != 0.0 {
        shown = lit_color(base_color, normal, view, raw_nv, texels.roughness, texels.metallic,
            corner.field);
    }
    var out = scene_fogged(shown, eye_sq * inv_eye);
    let tint = drawn_tint(corner.slot);
    if tint.a < 1.0 {
        let edge = 1.0 - saturate(raw_nv);
        let rim = (1.0 - tint.a) * (edge * edge * edge);
        out += tint.rgb * rim;
    }
    return vec4<f32>(out, 1.0);
}
