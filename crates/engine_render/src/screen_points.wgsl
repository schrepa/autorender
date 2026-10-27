// The depth buffer's points, which the screen effects (post.wgsl) and
// reflections (reflections.wgsl) share; each declares the effects' uniform
// `post` and the depth buffer `depth_texture` in a group of its own.

// World position of a depth-buffer pixel; w is 0 where only sky is (the
// cleared depth, 0 with reversed depth).
fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = post.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth > 0.0));
}

// The unit normal, facing the eye, of the surface the depth buffer holds at
// `pixel`, whose world position is `p`; none (zero) where its neighbors
// give no direction. The nearer neighbor on each axis keeps silhouettes
// from bending it. On the screen's edges the neighbor past the edge is the
// pixel itself, which gives no direction: the one inside stands in.
fn depth_normal(pixel: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let right = world_at(min(pixel + vec2<i32>(1, 0), last)).xyz - p;
    let left = p - world_at(max(pixel - vec2<i32>(1, 0), vec2<i32>(0))).xyz;
    let down = world_at(min(pixel + vec2<i32>(0, 1), last)).xyz - p;
    let up = p - world_at(max(pixel - vec2<i32>(0, 1), vec2<i32>(0))).xyz;
    let across = select(right, left,
        pixel.x == last.x || (pixel.x > 0 && dot(left, left) < dot(right, right)));
    let along = select(down, up,
        pixel.y == last.y || (pixel.y > 0 && dot(up, up) < dot(down, down)));
    let normal = cross(along, across);
    let length_squared = dot(normal, normal);
    if !(length_squared > 0.0) {
        return vec3<f32>(0.0);
    }
    let unit = normal * inverseSqrt(length_squared);
    return select(unit, -unit, dot(unit, post.eye.xyz - p) < 0.0);
}
