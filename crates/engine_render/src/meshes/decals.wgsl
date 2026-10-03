// Decals (decision 039): a box per decal, drawn after the opaque pass. Its
// fragments rebuild the opaque surface under them from the depth buffer and
// cover it with the decal's material where it lies inside the box.
struct DecalRecord {
    // Transposed top rows of the box's world matrix, unless `pose.x` names
    // its displayed pose in the joint palette.
    model: array<vec4<f32>, 3>,
    // Linear color and alpha.
    color: vec4<f32>,
    // Fade cosine, roughness, metalness and emission.
    params: vec4<f32>,
    // x: palette entry, or NO_DECAL_POSE.
    pose: vec4<u32>,
}
@group(2) @binding(8) var<storage, read> decal_records: array<DecalRecord>;
@group(2) @binding(9) var decal_depth: texture_depth_2d;

const NO_DECAL_POSE: u32 = 0xffffffffu;

// Multiplying decals scale the surface's color instead of covering it;
// canvas pipelines encode what they write.
override DECAL_MULTIPLY: bool = false;
override DECAL_CANVAS: bool = false;

// The unit box's twelve triangles, wound outward: culling front faces keeps
// one layer covering the box's pixels, wherever the eye is.
const DECAL_BOX = array<vec3<f32>, 36>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
);

struct DecalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) decal: u32,
    // Rows of the inverse of the box's world matrix.
    @location(1) @interpolate(flat) inverse_0: vec4<f32>,
    @location(2) @interpolate(flat) inverse_1: vec4<f32>,
    @location(3) @interpolate(flat) inverse_2: vec4<f32>,
    // The box's +y in world space.
    @location(4) @interpolate(flat) up: vec3<f32>,
}

@vertex
fn vs_decal(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> DecalOutput {
    let record = decal_records[instance];
    var rows = record.model;
    if record.pose.x != NO_DECAL_POSE {
        let pose = palette[record.pose.x];
        rows = array<vec4<f32>, 3>(pose.row_0, pose.row_1, pose.row_2);
    }
    let r0 = rows[0].xyz;
    let r1 = rows[1].xyz;
    let r2 = rows[2].xyz;
    let determinant = dot(r0, cross(r1, r2));
    var output: DecalOutput;
    output.decal = instance;
    // A flat box covers nothing.
    if !(abs(determinant) > 1.0e-12) {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }
    // Mirrored boxes turn their triangles inside out: wind them back.
    var corner = vertex;
    if determinant < 0.0 {
        corner = vertex + 2u - 2u * (vertex % 3u);
    }
    let local = vec4<f32>(DECAL_BOX[corner] - 0.5, 1.0);
    let world = vec3<f32>(dot(rows[0], local), dot(rows[1], local), dot(rows[2], local));
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    // The inverse's columns are the rows' cross products over the
    // determinant; its translation moves back.
    let inverse = mat3x3<f32>(cross(r1, r2), cross(r2, r0), cross(r0, r1)) * (1.0 / determinant);
    let translation = vec3<f32>(rows[0].w, rows[1].w, rows[2].w);
    let inverse_rows = transpose(inverse);
    output.inverse_0 = vec4<f32>(inverse_rows[0], -dot(inverse_rows[0], translation));
    output.inverse_1 = vec4<f32>(inverse_rows[1], -dot(inverse_rows[1], translation));
    output.inverse_2 = vec4<f32>(inverse_rows[2], -dot(inverse_rows[2], translation));
    output.up = stable_normalize(vec3<f32>(rows[0].y, rows[1].y, rows[2].y));
    return output;
}

// The opaque surface's world position at `pixel` of depth `depth`, without
// the jitter of temporally antialiased frames (decision 017).
fn decal_surface(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(decal_depth));
    let center = vec2<f32>(pixel) + 0.5;
    let ndc = vec2<f32>(center.x / size.x * 2.0 - 1.0, 1.0 - center.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w;
}

// How the surface at `pixel` moves one pixel along `axis`: toward the
// neighbor whose depth is nearer its own, so edges of other surfaces do not
// tilt it.
fn decal_step(pixel: vec2<i32>, depth: f32, world: vec3<f32>, axis: vec2<i32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(decal_depth)) - 1;
    let after = min(pixel + axis, last);
    let before = max(pixel - axis, vec2<i32>(0));
    let after_depth = textureLoad(decal_depth, after, 0);
    let before_depth = textureLoad(decal_depth, before, 0);
    let at_end = all(after == pixel);
    let at_start = all(before == pixel);
    if !at_end && (at_start || abs(after_depth - depth) <= abs(before_depth - depth)) {
        return decal_surface(after, after_depth) - world;
    }
    return world - decal_surface(before, before_depth);
}

// The decal's point on the opaque surface under this fragment: its albedo,
// coverage (`opacity`), normal and position. Discards pixels outside the
// box, of the sky or of no coverage.
fn decal_point(input: DecalOutput) -> SurfacePoint {
    let pixel = vec2<i32>(input.clip_position.xy);
    let depth = textureLoad(decal_depth, pixel, 0);
    // The sky has no surface to cover: it keeps the cleared depth, 0 with
    // reversed depth (decision 042).
    if !(depth > 0.0) {
        discard;
    }
    let world = decal_surface(pixel, depth);
    let point4 = vec4<f32>(world, 1.0);
    let local = vec3<f32>(dot(input.inverse_0, point4), dot(input.inverse_1, point4),
        dot(input.inverse_2, point4));
    if any(abs(local) > vec3<f32>(0.5)) {
        discard;
    }
    let record = decal_records[input.decal];
    let world_dx = decal_step(pixel, depth, world, vec2<i32>(1, 0));
    let world_dy = decal_step(pixel, depth, world, vec2<i32>(0, 1));
    let to_eye = camera.position.xyz - world;
    var normal = stable_normalize(cross(world_dx, world_dy));
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;
    }
    // Surfaces turning away from the box's +y fade out toward a right angle.
    let facing = saturate(dot(normal, input.up) / max(record.params.x, 1.0e-4));
    // The material spans the box's x and z; its map's footprint follows the
    // surface's steps.
    let uv = local.xz + 0.5;
    let uv_dx = vec2<f32>(dot(input.inverse_0.xyz, world_dx), dot(input.inverse_2.xyz, world_dx));
    let uv_dy = vec2<f32>(dot(input.inverse_0.xyz, world_dy), dot(input.inverse_2.xyz, world_dy));
    let transform = mat2x2<f32>(
        vec2<f32>(material_maps.transforms[0].x, material_maps.transforms[1].x),
        vec2<f32>(material_maps.transforms[0].y, material_maps.transforms[1].y));
    let texel = textureSampleGrad(base_color_texture, base_color_sampler, map_uv(0u, uv, uv),
        transform * uv_dx, transform * uv_dy);
    let alpha = record.color.a * texel.a * facing;
    if !(alpha > 0.0) {
        discard;
    }
    let albedo = record.color.rgb * texel.rgb;
    let metalness = record.params.z;
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = alpha;
    point.emission = record.params.w * texel.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.roughness = max(record.params.y, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), albedo, metalness);
    point.diffuse_color = albedo * (1.0 - metalness);
    point.normal = normal;
    point.geometric = normal;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
}

// Graded color and alpha as the target stores them.
fn decal_output(color: vec3<f32>, alpha: f32) -> vec4<f32> {
    if DECAL_CANVAS {
        return vec4<f32>(encoded(color), alpha);
    }
    return vec4<f32>(color, alpha);
}

@fragment
fn fs_decal(input: DecalOutput) -> @location(0) vec4<f32> {
    let point = decal_point(input);
    if DECAL_MULTIPLY {
        // Fog hides what it covers.
        let visible = point.opacity * (1.0 - fog(camera.position.xyz, point.world));
        return decal_output(mix(vec3<f32>(1.0), point.albedo, visible), 1.0);
    }
    // Covering decals are lit like any surface.
    return decal_output(finish(point, vec3<f32>(0.0)).color, point.opacity);
}

// Covering decals of frames with local lights (decision 023).
@fragment
fn fs_decal_lit(input: DecalOutput) -> @location(0) vec4<f32> {
    let point = decal_point(input);
    return decal_output(finish(point, local_light(point, input.clip_position.xy)).color,
        point.opacity);
}
