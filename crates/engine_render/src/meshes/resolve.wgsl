// The resolve of the visibility buffer: each material bin
// draws its tiles, and each of their pixels that bin's triangle covers
// shades as a forward-drawn fragment of that triangle would. The triangle's
// corners are posed as `vs_virtual` poses them, and its attributes
// interpolated at the pixel center with perspective-correct barycentric
// weights. Neighboring pixels may show other triangles, so derivatives come
// from the same triangle's weights at the pixel's partner in its 2x2 quad,
// across and down, as fine derivatives of forward fragments take them.

struct ResolveOutput {
    @builtin(position) clip_position: vec4<f32>,
}

// Corner `vertex` of two triangles over tile `instance` of the bin.
@vertex
fn vs_resolve(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> ResolveOutput {
    let tile = resolve_tiles[visibility_draw.tile_base + instance];
    let size = vec2<f32>(textureDimensions(visibility_ids));
    let corner = vec2<f32>(f32((0x32u >> vertex) & 1u), f32((0x2cu >> vertex) & 1u));
    let origin = vec2<f32>(f32(tile % visibility_draw.tiles_across),
        f32(tile / visibility_draw.tiles_across));
    let pixel = min((origin + corner) * 8.0, size);
    var output: ResolveOutput;
    output.clip_position = vec4<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0,
        0.0, 1.0);
    return output;
}

// Perspective-correct barycentric weights at normalized device point
// `point` (x, y, 1) of the triangle whose edge functions are `a`, `b` and
// `c` (Olano and Greer's homogeneous rasterization: corners stay in clip
// space, so triangles crossing the eye plane resolve too).
fn barycentric(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let edges = vec3<f32>(dot(a, point), dot(b, point), dot(c, point));
    return edges / (edges.x + edges.y + edges.z);
}

// A pixel's triangle: its fragment, how its world position, texture
// coordinates and interpolated normal change per pixel across and down,
// which way its quad partners lie (1 or -1, across and down), whether it
// faces the camera, and the point's previous clip x, y and w.
struct Resolved {
    fragment: Fragment,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    normal_dx: vec3<f32>,
    normal_dy: vec3<f32>,
    partner: vec2<f32>,
    front: bool,
    previous: vec3<f32>,
}

// The triangle of this draw's bin covering the pixel at `position`;
// discards pixels of other bins and of nothing.
fn resolve_pixel(position: vec4<f32>) -> Resolved {
    let ids = textureLoad(visibility_ids, vec2<i32>(position.xy), 0).xy;
    if ids.x == 0u || (ids.y >> 7u) != visibility_draw.bin {
        discard;
    }
    let entry = visible_clusters[ids.x - 1u];
    let slot = entry.x & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let material = colors[slot];
    let first = (ids.y & 127u) * 3u;
    let a = virtual_corner(entry.y, first);
    let b = virtual_corner(entry.y, first + 1u);
    let c = virtual_corner(entry.y, first + 2u);
    let va = transform_vertex(a.position, a.normal, material.color, material.surface, a.uv,
        model);
    let vb = transform_vertex(b.position, b.normal, material.color, material.surface, b.uv,
        model);
    let vc = transform_vertex(c.position, c.normal, material.color, material.surface, c.uv,
        model);
    let pa = va.clip_position.xyw;
    let pb = vb.clip_position.xyw;
    let pc = vc.clip_position.xyw;
    let ea = cross(pb, pc);
    let eb = cross(pc, pa);
    let ec = cross(pa, pb);
    let size = vec2<f32>(textureDimensions(visibility_ids));
    let point = vec3<f32>(position.x / size.x * 2.0 - 1.0, 1.0 - position.y / size.y * 2.0, 1.0);
    let weights = barycentric(ea, eb, ec, point);
    // Even pixels' partners follow them, odd ones' precede them.
    let partner = select(vec2<f32>(1.0), vec2<f32>(-1.0),
        (vec2<u32>(position.xy) & vec2<u32>(1u)) == vec2<u32>(1u));
    let across = (barycentric(ea, eb, ec, point + vec3<f32>(partner.x * 2.0 / size.x, 0.0, 0.0))
        - weights) * partner.x;
    let down = (barycentric(ea, eb, ec, point - vec3<f32>(0.0, partner.y * 2.0 / size.y, 0.0))
        - weights) * partner.y;
    let normal_u = weights.x * va.normal_u + weights.y * vb.normal_u + weights.z * vc.normal_u;
    let world_v = weights.x * va.world_v + weights.y * vb.world_v + weights.z * vc.world_v;
    var resolved: Resolved;
    resolved.fragment.color = output_color(va.material);
    resolved.fragment.normal = normal_u.xyz;
    resolved.fragment.world = world_v.xyz;
    resolved.fragment.uv = vec2<f32>(normal_u.w, world_v.w);
    resolved.fragment.uv1 = resolved.fragment.uv;
    resolved.fragment.surface = va.material.z;
    resolved.fragment.pixel = position.xy;
    let normal_across = across.x * va.normal_u + across.y * vb.normal_u + across.z * vc.normal_u;
    let normal_down = down.x * va.normal_u + down.y * vb.normal_u + down.z * vc.normal_u;
    let world_across = across.x * va.world_v + across.y * vb.world_v + across.z * vc.world_v;
    let world_down = down.x * va.world_v + down.y * vb.world_v + down.z * vc.world_v;
    resolved.world_dx = world_across.xyz;
    resolved.world_dy = world_down.xyz;
    resolved.uv_dx = vec2<f32>(normal_across.w, world_across.w);
    resolved.uv_dy = vec2<f32>(normal_down.w, world_down.w);
    resolved.normal_dx = normal_across.xyz;
    resolved.normal_dy = normal_down.xyz;
    resolved.partner = partner;
    // Counterclockwise in normalized device coordinates faces the camera.
    resolved.front = dot(pa, ea) > 0.0;
    // Only moving slots read their previous pose, as `vs_virtual_temporal`.
    var previous_world = resolved.fragment.world;
    if (entry.x & MOVED_SLOT) != 0u {
        let pose = previous_poses[slot];
        let mesh_point = vec4<f32>(weights.x * a.position + weights.y * b.position
            + weights.z * c.position, 1.0);
        previous_world = vec3<f32>(dot(pose.row_0, mesh_point), dot(pose.row_1, mesh_point),
            dot(pose.row_2, mesh_point));
    }
    resolved.previous = (camera.previous_view_projection * vec4<f32>(previous_world, 1.0)).xyw;
    return resolved;
}

// The shading normal at a quad partner: the interpolated normal changed by
// `normal_step`, the world position by `world_step` and the texture
// coordinates by `uv_step`, sampling the normal map there.
fn stepped_normal(resolved: Resolved, texels: MaterialTexels, normal_step: vec3<f32>,
        world_step: vec3<f32>, uv_step: vec2<f32>, ripples: f32) -> vec3<f32> {
    let input = resolved.fragment;
    var sampled = texels.normal;
    if material_flag(2u, NORMAL_MAP) {
        let step = map_gradient(2u, uv_step, uv_step);
        sampled = textureSampleGrad(normal_texture, normal_sampler,
            map_uv(2u, input.uv, input.uv1) + step, texels.normal_dx,
            texels.normal_dy).xyz * 2.0 - 1.0;
    }
    return shading_normal(geometric_normal(input.normal + normal_step, resolved.front),
        input.world + world_step, resolved.world_dx, resolved.world_dy, sampled,
        texels.normal_dx, texels.normal_dy, ripples);
}

// `surface_point` of a resolved pixel: explicit derivatives, and the shading
// normal's turn toward its quad partners across and down, as forward
// fragments' derivatives take it.
fn resolved_point(resolved: Resolved) -> SurfacePoint {
    let input = resolved.fragment;
    let gradients = UvGradients(resolved.uv_dx, resolved.uv_dy, resolved.uv_dx, resolved.uv_dy);
    var texels: MaterialTexels;
    if material_flag(4096u, WORLD_MAPPING) {
        texels = world_texels(input.world, stable_normalize(input.normal), resolved.world_dx,
            resolved.world_dy);
    } else {
        texels = material_texels_grad(input.uv, input.uv1, gradients);
    }
    let point = textured_point(input.color, input.normal, input.world, resolved.world_dx,
        resolved.world_dy, texels, input.surface, resolved.front);
    let ripples = unpack4x8unorm(input.surface).w;
    let x = resolved.partner.x;
    let y = resolved.partner.y;
    if material_flag(4096u, WORLD_MAPPING) {
        // World-mapped maps turn toward the partners with
        // the interpolated normal alone.
        let across = stable_normalize(point.normal + resolved.normal_dx * x);
        let down = stable_normalize(point.normal + resolved.normal_dy * y);
        return widened(point, across - point.normal, down - point.normal);
    }
    let across = stepped_normal(resolved, texels, resolved.normal_dx * x, resolved.world_dx * x,
        resolved.uv_dx * x, ripples);
    let down = stepped_normal(resolved, texels, resolved.normal_dy * y, resolved.world_dy * y,
        resolved.uv_dy * y, ripples);
    return widened(point, across - point.normal, down - point.normal);
}

fn shade_resolved(resolved: Resolved) -> Shaded {
    return finish(resolved_point(resolved), vec3<f32>(0.0));
}

fn shade_resolved_lit(resolved: Resolved) -> Shaded {
    let point = resolved_point(resolved);
    return finish(point, local_light(point, resolved.fragment.pixel));
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_resolved(resolve_pixel(position)).color), 1.0);
}

@fragment
fn fs_resolve_lit(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_resolved_lit(resolve_pixel(position)).color), 1.0);
}

@fragment
fn fs_resolve_post(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(shade_resolved(resolve_pixel(position)));
}

@fragment
fn fs_resolve_post_lit(@builtin(position) position: vec4<f32>) -> SceneTargets {
    return scene_targets(shade_resolved_lit(resolve_pixel(position)));
}

@fragment
fn fs_resolve_temporal(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = resolve_pixel(position);
    return temporal_targets(shade_resolved(resolved), position, resolved.previous);
}

@fragment
fn fs_resolve_temporal_lit(@builtin(position) position: vec4<f32>) -> TemporalTargets {
    let resolved = resolve_pixel(position);
    return temporal_targets(shade_resolved_lit(resolved), position, resolved.previous);
}
