import package::engine::shading::SurfaceInput;
import package::engine::view::stable_normalize;

struct Grounding {
    amount: f32,
    texel: vec4<f32>,
    up: vec3<f32>,
}

fn grounding(input: SurfaceInput) -> Grounding {
    var at: Grounding;
    let reach = bitcast<f32>(input.words.z);
    if !(reach > 0.0) || input.words.w == 0u {
        return at;
    }
    let tilt = unpack2x16snorm(input.words.x);
    let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
    let above = input.world.y - input.origin.y - bitcast<f32>(input.words.y)
        + dot(tilt, input.world.xz - input.origin.xz) / up;
    at.amount = 1.0 - smoothstep(0.0, reach, above);
    at.texel = unpack4x8unorm(input.words.w);
    at.up = vec3<f32>(tilt.x, up, tilt.y);
    return at;
}

fn grounded_color(input: SurfaceInput, color: vec3<f32>) -> vec3<f32> {
    let at = grounding(input);
    return mix(color, at.texel.rgb * at.texel.rgb, at.amount);
}

fn grounded_normal(input: SurfaceInput, facing: vec3<f32>) -> vec3<f32> {
    let at = grounding(input);
    if at.amount <= 0.0 {
        return facing;
    }
    return stable_normalize(mix(facing, at.up, at.amount));
}

fn grounded_roughness(input: SurfaceInput, rough: f32) -> f32 {
    let at = grounding(input);
    return mix(rough, max(at.texel.a, 0.03), at.amount);
}

fn grounded_metalness(input: SurfaceInput, metal: f32) -> f32 {
    return mix(metal, 0.0, grounding(input).amount);
}
