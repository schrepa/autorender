// Exactly 48 bytes, matching GradeUniform.
struct Grade {
    parameters: vec4<f32>,
    shadows: vec4<f32>,
    highlights: vec4<f32>,
}

@group(0) @binding(0) var radiance: texture_2d<f32>;
@group(0) @binding(1) var<uniform> grade: Grade;
@group(0) @binding(2) var<storage, read_write> radiance_limits: atomic<u32>;

override ENCODE_OUTPUT: bool = false;
override CANVAS_OUTPUT: bool = false;

@fragment
fn fs_grade(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let stored = guard_radiance(textureLoad(radiance, vec2<i32>(position.xy), 0), 2u);
    var color = grade_exposed(stored.rgb * 64.0, grade.parameters.xyz,
        grade.shadows.rgb, grade.highlights.rgb);
    if ENCODE_OUTPUT {
        color = encode_srgb(color);
    }
    return vec4<f32>(color, select(clamp(stored.a, 0.0, 1.0), 1.0, CANVAS_OUTPUT));
}
