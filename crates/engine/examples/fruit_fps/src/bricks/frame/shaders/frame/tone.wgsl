import package::engine::color::encode_srgb;
import package::generated::frame::FrameGrade;

@group(0) @binding(2) var scene: texture_2d<f32>;
@group(2) @binding(0) var<uniform> tone_grade: FrameGrade;

const GRADE_MIDDLE: f32 = 0.18;

fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn filmic_scalar(x: f32) -> f32 {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn graded(color: vec3<f32>) -> vec3<f32> {
    let grade = tone_grade.grade;
    let exposed = max(color, vec3<f32>(0.0)) * grade.x;
    let luma = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let saturated = max(mix(vec3<f32>(luma), exposed, grade.z), vec3<f32>(0.0));
    if grade.y == 1.0 {
        return saturated;
    }
    return GRADE_MIDDLE * pow(saturated / GRADE_MIDDLE, vec3<f32>(grade.y));
}

fn tonemap(exposed: vec3<f32>) -> vec3<f32> {
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let shown = filmic_scalar(peak);
    let kept = exposed * (shown / max(peak, 1.0e-6));
    let burnt = 1.0 - 1.0 / (max(peak - shown, 0.0) + 1.0);
    if burnt <= 0.0 {
        return saturate(kept);
    }
    return saturate(mix(kept, filmic(exposed), burnt));
}

@fragment
fn fs_tone(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(scene));
    let pixel = min(vec2<i32>(position.xy), size - 1);
    return vec4<f32>(encode_srgb(tonemap(graded(textureLoad(scene, pixel, 0).rgb))), 1.0);
}
