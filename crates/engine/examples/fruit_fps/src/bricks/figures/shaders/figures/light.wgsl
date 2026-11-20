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

fn figure_field(world: vec3<f32>, slot: u32) -> vec4<f32> {
    let shape = figure_light.field_shape;
    if shape.z == 0.0 {
        return vec4<f32>(0.0);
    }
    let at_camera = drawn_float_count(slot) >= 8u && drawn_floats(slot, 1u).w > 0.5;
    let at = select(world, camera.position.xyz, at_camera);
    return light_field(figure_field_map, figure_field_sampler, at,
        normalize(figure_light.direction.xyz), figure_light.field, shape);
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

fn scene_fogged(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let distance = length(world - camera.position.xyz);
    let fog = figure_light.fog;
    let amount = figure_light.horizon.w * (1.0 - exp(-max(distance - fog.x, 0.0) * fog.y));
    return mix(color, figure_light.horizon.rgb, amount);
}

fn lit_color(base_color: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, roughness: f32,
        metallic: f32, field: vec4<f32>) -> vec3<f32> {
    let to_sun = normalize(figure_light.direction.xyz);
    let half_way = normalize(to_sun + view);
    let n_l = max(dot(normal, to_sun), 0.0);
    let n_v = max(dot(normal, view), SMALLEST_COSINE);
    let n_h = max(dot(normal, half_way), 0.0);
    let alpha = max(roughness * roughness, SMALLEST_ALPHA);
    let alpha_2 = alpha * alpha;
    let lobe = n_h * n_h * (alpha_2 - 1.0) + 1.0;
    let distribution = alpha_2 / (FIGURE_PI * lobe * lobe);
    let visibility = 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
    let reflectance = mix(vec3<f32>(DIELECTRIC_REFLECTANCE), base_color, metallic);
    let grazing = pow(1.0 - max(dot(to_sun, half_way), 0.0), 5.0);
    let fresnel = reflectance + (1.0 - reflectance) * grazing;
    let diffuse = base_color * (1.0 - metallic);
    let wrap = figure_light.sun.w;
    let wrapped = saturate((dot(normal, to_sun) + wrap) / (1.0 + wrap));
    let sun = (diffuse * wrapped + FIGURE_PI * distribution * visibility * fresnel * n_l)
        * figure_light.sun.rgb;
    let ambient = hemisphere(normal) * diffuse + hemisphere(reflect(-view, normal)) * reflectance;
    if figure_light.field_shape.z == 0.0 {
        return sun + ambient;
    }
    let open = max(hemisphere(vec3<f32>(0.0, 1.0, 0.0)), vec3<f32>(1.0e-4));
    let share = max(field.rgb / open, vec3<f32>(figure_light.field_shape.w));
    return sun * field.a + ambient * share;
}

@fragment fn fs_figures(corner: FigureCorner, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let texels = material_texels(corner.uv);
    let trimmed = drawn_float_count(corner.slot) >= 7u;
    let trim = saturate(drawn_floats(corner.slot, 0u).w * texels.trim);
    let tinted = mix(texels.base_color, drawn_floats(corner.slot, 1u).xyz, trim);
    let base_color = select(texels.base_color, tinted, trimmed);
    let facing = stable_normalize(select(-corner.normal, corner.normal, front));
    let normal = deflected_normal(facing, dpdx(corner.world), dpdy(corner.world),
        dpdx(corner.uv), dpdy(corner.uv), texels.normal);
    let view = stable_normalize(camera.position.xyz - corner.world);
    let lit = lit_color(base_color, normal, view, texels.roughness, texels.metallic,
        corner.field);
    let shown = select(base_color, lit, figure_light.direction.w != 0.0);
    let tint = drawn_tint(corner.slot);
    let rim = (1.0 - tint.a) * pow(1.0 - saturate(dot(normal, view)), 3.0);
    return vec4<f32>(scene_fogged(shown, corner.world) + tint.rgb * rim, 1.0);
}
