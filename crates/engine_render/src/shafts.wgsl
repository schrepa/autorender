// Sun shafts, prepended with scene.wgsl and effects.wgsl: at
// quarter resolution, each view ray marches through the fog to the depth
// buffer's surface or the gathering distance, adding the sunlight each step
// scatters toward the eye where the far shadow maps let the sun through,
// dimmed by the fog in front of it. The ray's length goes with the light, so
// the filter of post.wgsl can bring it to half resolution without crossing
// silhouettes. The composite scales the result by the sun's color.

@group(1) @binding(0) var<uniform> post: Post;
@group(1) @binding(1) var depth_texture: texture_depth_2d;

const SHAFT_STEPS: u32 = 16u;

// Scattered light toward the eye, with the Henyey-Greenstein phase toward
// the sun and the strength applied, and the length of the ray that gathered
// it.
@fragment
fn fs_shafts(@builtin(position) position: vec4<f32>) -> @location(0) vec2<f32> {
    if shadow.state.x == 0.0 {
        return vec2<f32>(0.0);
    }
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = min(shaft_pixel(vec2<i32>(position.xy)), last);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let far_point = post.inverse_view_projection * clip;
    let eye = post.eye.xyz;
    let ray = far_point.xyz / far_point.w - eye;
    let ray_length = length(ray);
    if !(ray_length > 0.0) {
        return vec2<f32>(0.0);
    }
    let direction = ray / ray_length;
    let reach = min(ray_length, post.shafts.z);
    let step = reach / f32(SHAFT_STEPS);
    let noise = pixel_noise(position.xy);
    let density = lighting.fog.y;
    let falloff = lighting.fog.z;
    var scattered = 0.0;
    var optical = 0.0;
    for (var i = 0u; i < SHAFT_STEPS; i++) {
        let point = eye + direction * ((f32(i) + noise) * step);
        let local = density * exp(-falloff * max(point.y, -1000.0));
        scattered += far_visibility(point) * local * exp(-optical) * step;
        optical += local * step;
    }
    // Henyey-Greenstein phase toward the sun.
    let g = post.shafts.y;
    let cosine = dot(direction, lighting.sun.xyz);
    let denominator = 1.0 + g * g - 2.0 * g * cosine;
    let phase = (1.0 - g * g) / (4.0 * PI * denominator * sqrt(denominator));
    return vec2<f32>(scattered * phase * post.shafts.x, reach);
}
