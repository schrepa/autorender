import package::engine::shading::SurfaceInput;
import package::engine::view::stable_normalize;
import package::generated::scatter::PlacedValues;
import package::scatter::values::placed_values_of;

struct Grounding {
    amount: f32,
    texel: vec4<f32>,
    up: vec3<f32>,
}

var<private> held_grounding: Grounding;

fn prepare_grounding(input: SurfaceInput) {
    held_grounding = grounding(input);
}

fn grounding(input: SurfaceInput) -> Grounding {
    return grounding_of(placed_values_of(input.kind, input.object), input.world);
}

fn grounding_of(values: PlacedValues, world: vec3<f32>) -> Grounding {
    var at: Grounding;
    if !(values.reach > 0.0) || values.texel == 0u {
        return at;
    }
    let tilt = unpack2x16snorm(values.tilt);
    let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
    let above = world.y + dot(tilt, world.xz) / up - values.ground;
    at.amount = 1.0 - smoothstep(0.0, values.reach, above);
    at.texel = unpack4x8unorm(values.texel);
    at.up = vec3<f32>(tilt.x, up, tilt.y);
    return at;
}

fn ground_color(at: Grounding, color: vec3<f32>) -> vec3<f32> {
    return mix(color, at.texel.rgb * at.texel.rgb, at.amount);
}

fn ground_normal(at: Grounding, facing: vec3<f32>) -> vec3<f32> {
    if at.amount <= 0.0 {
        return facing;
    }
    return stable_normalize(mix(facing, at.up, at.amount));
}

fn ground_roughness(at: Grounding, rough: f32) -> f32 {
    return mix(rough, max(at.texel.a, 0.03), at.amount);
}

fn ground_metalness(at: Grounding, metal: f32) -> f32 {
    return mix(metal, 0.0, at.amount);
}

fn grounded_color(input: SurfaceInput, color: vec3<f32>) -> vec3<f32> {
    return ground_color(held_grounding, color);
}

fn grounded_normal(input: SurfaceInput, facing: vec3<f32>) -> vec3<f32> {
    return ground_normal(held_grounding, facing);
}

fn grounded_roughness(input: SurfaceInput, rough: f32) -> f32 {
    return ground_roughness(held_grounding, rough);
}

fn grounded_metalness(input: SurfaceInput, metal: f32) -> f32 {
    return ground_metalness(held_grounding, metal);
}
