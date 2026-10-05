// Octahedral impostors (decision 048): a draw group's mesh baked from
// IMPOSTOR_FRAMES x IMPOSTOR_FRAMES directions over the whole sphere, or
// over the upper hemisphere for placed objects (decision 060), into one
// layer of four texture arrays, drawn for instances too small for its
// texels to show as a card facing the viewer that blends the three frames
// around the view. Texels hold premultiplied coverage: albedo and coverage;
// emissive color and emission mask; normal and depth; roughness and
// metalness factors, occlusion and the sky the surface sees past its own
// mesh (decision 060). Normals are plain vectors, which filter and average
// across levels as octahedral ones, folded across the lower hemisphere's
// seams, cannot.
//
// Frames, records and their directions: `impostor_frames.wgsl`.

// One record per draw group; groups without an impostor never draw one.
@group(1) @binding(11) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(1) @binding(12) var impostor_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var impostor_emissive: texture_2d_array<f32>;
@group(1) @binding(14) var impostor_geometry: texture_2d_array<f32>;
@group(1) @binding(15) var impostor_surface: texture_2d_array<f32>;
@group(1) @binding(16) var impostor_sampler: sampler;

// Baking: every frame renders the mesh orthographically into its cell of the
// layer, instance `frame` for frame `frame`, the sphere filling the cell.
struct ImpostorBake {
    // Mesh-space center and radius of the frames' sphere.
    sphere: vec4<f32>,
    // The record's flags.
    flags: vec4<u32>,
}
@group(2) @binding(20) var<uniform> impostor_bake: ImpostorBake;
// Each frame's nearest depth toward its viewer, at four times the layer's
// texels, drawn before the colors (decision 060).
@group(2) @binding(21) var impostor_shade: texture_depth_2d;

fn bake_hemisphere() -> bool {
    return (impostor_bake.flags.x & IMPOSTOR_HEMISPHERE) != 0u;
}

// The share of the sky above the horizon that mesh-space `position`, with
// unit normal `normal`, sees past its own mesh, weighted by how squarely
// its surface faces each direction: the frames' nearest depths are a depth
// map from every frame's direction (decision 060). Cards shade the sun by
// it, as the mesh's own shadows would darken its insides.
fn impostor_self_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let size = textureDimensions(impostor_shade).x;
    // A texel of the depths, in sphere radii.
    let texel = 2.0 * f32(IMPOSTOR_FRAMES) / f32(size);
    // Moved a texel and a half along its normal, a surface clears its own
    // depth.
    let p = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w
        + normal * (1.5 * texel);
    let hemisphere = bake_hemisphere();
    var seen = 0.0;
    var total = 0.0;
    for (var frame = 0u; frame < IMPOSTOR_FRAMES * IMPOSTOR_FRAMES; frame++) {
        let d = impostor_frame_direction(frame, hemisphere);
        let facing = dot(normal, d);
        if d.y < 0.0 || facing <= 0.0 {
            continue;
        }
        let basis = impostor_basis(d);
        let image = vec2<f32>(dot(p, basis.right), -dot(p, basis.up));
        let cell = vec2<f32>(f32(frame % IMPOSTOR_FRAMES), f32(frame / IMPOSTOR_FRAMES));
        let atlas = (cell + 0.5 + image * 0.5) / f32(IMPOSTOR_FRAMES);
        let at = min(vec2<u32>(max(atlas, vec2<f32>(0.0)) * f32(size)), vec2<u32>(size - 1u));
        // Nearer the frame's viewer is greater; empty texels hold zero.
        let nearest = textureLoad(impostor_shade, at, 0);
        total += facing;
        if 0.5 + 0.5 * dot(p, d) >= nearest {
            seen += facing;
        }
    }
    if total <= 0.0 {
        return 1.0;
    }
    return seen / total;
}

struct BakeOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Mesh-space normal and texture u; mesh-space position and texture v.
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    // Depth toward the frame's viewer, in sphere radii from the center.
    @location(2) depth: f32,
}

struct BakeExtendedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    @location(2) depth: f32,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

fn bake_vertex(position: vec3<f32>, encoded_normal: vec2<f32>, uv: vec2<f32>, frame: u32)
        -> BakeOutput {
    let d = impostor_frame_direction(frame, bake_hemisphere());
    let basis = impostor_basis(d);
    let local = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w;
    let image = vec2<f32>(dot(local, basis.right), -dot(local, basis.up));
    let depth = clamp(dot(local, d), -1.0, 1.0);
    // The frame's cell, from the top left.
    let cell = vec2<f32>(f32(frame % IMPOSTOR_FRAMES), f32(frame / IMPOSTOR_FRAMES));
    let atlas = (cell + 0.5 + image * 0.5) / f32(IMPOSTOR_FRAMES);
    var output: BakeOutput;
    // Nearer the frame's viewer is greater, as reversed depth keeps it.
    output.clip_position = vec4<f32>(atlas.x * 2.0 - 1.0, 1.0 - atlas.y * 2.0,
        0.5 + 0.5 * depth, 1.0);
    output.normal_u = vec4<f32>(octahedral_normal(encoded_normal), uv.x);
    output.local_v = vec4<f32>(position, uv.y);
    output.depth = depth;
    return output;
}

@vertex
fn vs_impostor_bake(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @builtin(instance_index) frame: u32) -> BakeOutput {
    return bake_vertex(position, normal, uv, frame);
}

@vertex
fn vs_impostor_bake_extended(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @location(9) color: vec4<f32>, @location(10) uv1: vec2<f32>,
        @builtin(instance_index) frame: u32) -> BakeExtendedOutput {
    let vertex = bake_vertex(position, normal, uv, frame);
    var output: BakeExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.local_v = vertex.local_v;
    output.depth = vertex.depth;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

// The albedo, emission, geometry and surface factors of the material at a
// baked point, as its fragments would read them; the targets' resolve
// premultiplies them by coverage.
struct BakeTargets {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) geometry: vec4<f32>,
    @location(3) surface: vec4<f32>,
}

// Discards the gaps of masked materials, as their fragments would.
fn bake_mask(texels: MaterialTexels, color: vec4<f32>) {
    if ALPHA_MASK && texels.base.a * color.a * texels.coverage < material_maps.params.z {
        discard;
    }
}

fn bake_targets(input: BakeOutput, color: vec4<f32>, uv1: vec2<f32>, front: bool)
        -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    let texels = material_texels(uv, uv1);
    let local = input.local_v.xyz;
    let local_dx = dpdx(local);
    let local_dy = dpdy(local);
    let geometric = geometric_normal(input.normal_u.xyz, front);
    var normal = geometric;
    if material_flag(2u, NORMAL_MAP) {
        normal = mapped_normal(geometric, local_dx, local_dy, texels.normal_dx,
            texels.normal_dy, texels.normal, material_maps.params.x);
    }
    var emissive = vec3<f32>(0.0);
    var occlusion = 1.0;
    if material_flag(1u, MATERIAL_MAPS) {
        emissive = texels.emissive;
        occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
    }
    bake_mask(texels, color);
    var targets: BakeTargets;
    targets.albedo = vec4<f32>(texels.base.rgb * color.rgb, 1.0);
    // Texture alpha masks emission (decision 019).
    targets.emissive = vec4<f32>(emissive, texels.base.a);
    targets.geometry = vec4<f32>(stable_normalize(normal) * 0.5 + 0.5, 0.5 + 0.5 * input.depth);
    targets.surface = vec4<f32>(texels.factors, occlusion,
        impostor_self_visibility(local, geometric));
    return targets;
}

// The frames' nearest depths alone (decision 060).
@fragment
fn fs_impostor_shade(input: BakeOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, uv), vec4<f32>(1.0));
}

@fragment
fn fs_impostor_shade_extended(input: BakeExtendedOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, input.uv1), input.color);
}

@fragment
fn fs_impostor_bake(input: BakeOutput, @builtin(front_facing) front: bool) -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    return bake_targets(input, vec4<f32>(1.0), uv, front);
}

@fragment
fn fs_impostor_bake_extended(input: BakeExtendedOutput, @builtin(front_facing) front: bool)
        -> BakeTargets {
    var vertex: BakeOutput;
    vertex.clip_position = input.clip_position;
    vertex.normal_u = input.normal_u;
    vertex.local_v = input.local_v;
    vertex.depth = input.depth;
    return bake_targets(vertex, input.color, input.uv1, front);
}

// Drawing: one card per impostor instance, six vertices, from the visible
// slot and its draw group.
struct ImpostorInput {
    @builtin(vertex_index) vertex: u32,
    @location(2) slot: u32,
    @location(3) batch: u32,
}

struct ImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Layer coordinates of the card point in the three blended frames.
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    // The card point in world space.
    @location(2) world: vec3<f32>,
    // Columns of the normal transform (cofactors, signed by the pose's
    // orientation); w: the first two frames' blend weights.
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    // World offset per unit of baked depth toward the viewer, and the
    // card's own depth in those units.
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    // Linear color and opacity as four 16-bit fractions, the packed surface
    // (decision 019), and the three frames (six bits each).
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
}

struct ImpostorTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    // The unjittered previous clip position: x, y and w.
    @location(9) previous: vec3<f32>,
}

// Two triangles of a card, corners in its right and up.
const IMPOSTOR_CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

// Unit direction from `point` toward the viewer: toward the eye with
// perspective, against the view with orthographic projections.
fn toward_viewer(point: vec3<f32>) -> vec3<f32> {
    let projection = camera.view_projection;
    let w_row = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    if length(w_row) > 0.5 {
        return stable_normalize(camera.position.xyz - point);
    }
    // Reversed depth (decision 042): the near plane is 1, the far plane 0.
    let near = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let far = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return stable_normalize(near.xyz / near.w - far.xyz / far.w);
}

// The card of an impostor instance in mesh space, and its outputs besides
// the previous position.
struct ImpostorCard {
    local: vec3<f32>,
    output: ImpostorOutput,
}

fn impostor_card(input: ImpostorInput) -> ImpostorCard {
    let slot = input.slot & ~MOVED_SLOT;
    let model = pose_matrix(poses[slot]);
    let record = impostor_records[impostor_index(input.batch)];
    let center = record.sphere.xyz;
    let radius = record.sphere.w;
    // Cofactor columns: the inverse's rows times the determinant, rescaled
    // as `transform_vertex` does.
    let inverse_scale = 1.0 / max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz * inverse_scale;
    let y = model[1].xyz * inverse_scale;
    let z = model[2].xyz * inverse_scale;
    let yz = cross(y, z);
    let zx = cross(z, x);
    let xy = cross(x, y);
    let orientation = sign(dot(x, yz));
    // The mesh-space direction toward the viewer.
    let toward = toward_viewer((model * vec4<f32>(center, 1.0)).xyz);
    let dir = stable_normalize(vec3<f32>(dot(yz, toward), dot(zx, toward), dot(xy, toward))
        * orientation);
    // The three frames around it: the grid triangle holding it.
    let hemisphere = (record.flags & IMPOSTOR_HEMISPHERE) != 0u;
    let grid = (impostor_grid(dir, hemisphere) * 0.5 + 0.5) * f32(IMPOSTOR_FRAMES - 1u);
    let cell = min(floor(grid), vec2<f32>(f32(IMPOSTOR_FRAMES - 2u)));
    let f = grid - cell;
    let base = vec2<u32>(cell);
    var corners = array<vec2<u32>, 3>(base, base + vec2<u32>(1u, 0u), base + vec2<u32>(0u, 1u));
    var weights = vec3<f32>(1.0 - f.x - f.y, f.x, f.y);
    if f.x + f.y > 1.0 {
        corners = array<vec2<u32>, 3>(base + vec2<u32>(1u), base + vec2<u32>(0u, 1u),
            base + vec2<u32>(1u, 0u));
        weights = vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    // The card: the bounds' outline seen along `dir`, in front of them.
    let basis = impostor_basis(dir);
    let half = record.extent;
    let support = dot(abs(dir), half);
    let outline = vec2<f32>(dot(abs(basis.right), half), dot(abs(basis.up), half));
    let corner = IMPOSTOR_CARD[input.vertex];
    let local = center + dir * support + basis.right * (corner.x * outline.x)
        + basis.up * (corner.y * outline.y);
    let world = model * vec4<f32>(local, 1.0);
    // Each frame shows the card point where the view projects it onto the
    // frame's plane through the center.
    let offset = local - center;
    var uvs: array<vec2<f32>, 3>;
    var frames = 0u;
    for (var k = 0u; k < 3u; k++) {
        let frame = corners[k].y * IMPOSTOR_FRAMES + corners[k].x;
        let d = impostor_frame_direction(frame, hemisphere);
        let frame_basis = impostor_basis(d);
        let projected = offset - dir * (dot(offset, d) / max(dot(dir, d), 1.0e-3));
        let image = vec2<f32>(dot(projected, frame_basis.right), -dot(projected, frame_basis.up))
            / radius;
        uvs[k] = (vec2<f32>(corners[k]) + 0.5 + image * 0.5) / f32(IMPOSTOR_FRAMES);
        frames |= frame << (6u * k);
    }
    let color = colors[slot];
    var output: ImpostorOutput;
    output.clip_position = camera.view_projection * world;
    output.frames_ab = vec4<f32>(uvs[0], uvs[1]);
    output.frame_c = uvs[2];
    output.world = world.xyz;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    output.material = vec4<u32>(color.color, color.surface, frames);
    output.batch = input.batch;
    return ImpostorCard(local, output);
}

@vertex
fn vs_impostor(input: ImpostorInput) -> ImpostorOutput {
    return impostor_card(input).output;
}

@vertex
fn vs_impostor_temporal(input: ImpostorInput) -> ImpostorTemporalOutput {
    let card = impostor_card(input);
    let slot = input.slot & ~MOVED_SLOT;
    let point = vec4<f32>(card.local, 1.0);
    // Only moving slots read their previous poses (decision 017).
    var pose = poses[slot];
    if (input.slot & MOVED_SLOT) != 0u {
        pose = previous_poses[slot];
    }
    let world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    let vertex = card.output;
    var output: ImpostorTemporalOutput;
    output.clip_position = vertex.clip_position;
    output.frames_ab = vertex.frames_ab;
    output.frame_c = vertex.frame_c;
    output.world = vertex.world;
    output.normal_x = vertex.normal_x;
    output.normal_y = vertex.normal_y;
    output.normal_z = vertex.normal_z;
    output.depth_axis = vertex.depth_axis;
    output.material = vertex.material;
    output.batch = vertex.batch;
    output.previous = (camera.previous_view_projection * world).xyw;
    return output;
}

// The surface an impostor fragment shows: the three frames blended by
// weight and coverage, unpremultiplied, on the slot's color and surface;
// discards fragments the blend leaves under half covered, scaling coverage
// up on coarser levels as masked materials do.
fn impostor_point(input: ImpostorOutput) -> SurfacePoint {
    let record = impostor_records[impostor_index(input.batch)];
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let frames = vec3<u32>(packed & 63u, (packed >> 6u) & 63u, (packed >> 12u) & 63u);
    // The level whose texels match the pixels, and half a texel of the
    // coarser level sampled, which keeps each frame's samples in its cell.
    let size = f32(textureDimensions(impostor_albedo).x);
    let footprint = max(length(dpdx(input.frames_ab.xy)), length(dpdy(input.frames_ab.xy)))
        * size;
    let level = clamp(log2(max(footprint, 1.0e-6)), 0.0,
        f32(textureNumLevels(impostor_albedo) - 1u));
    let inset = 0.5 * exp2(ceil(level)) / size;
    let span = 1.0 / f32(IMPOSTOR_FRAMES);
    let unclamped = array<vec2<f32>, 3>(input.frames_ab.xy, input.frames_ab.zw, input.frame_c);
    var uvs: array<vec2<f32>, 3>;
    var albedo = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        let cell = vec2<f32>(f32(frames[k] % IMPOSTOR_FRAMES), f32(frames[k] / IMPOSTOR_FRAMES))
            * span;
        uvs[k] = clamp(unclamped[k], cell + inset, cell + span - inset);
        albedo += weights[k] * textureSampleLevel(impostor_albedo, impostor_sampler, uvs[k],
            record.layer, level);
    }
    // Most of a card around its mesh is empty: those fragments stop before
    // reading the rest.
    let coverage = albedo.a;
    if coverage * (1.0 + MASK_MIP_SCALE * level) < 0.5 {
        discard;
    }
    // Fading in (decision 063), a card covers that share of its pixels; the
    // mesh behind it shows through the rest.
    let fade = impostor_fade(input.batch);
    if fade < 1.0 && fade <= pixel_noise(input.clip_position.xy) {
        discard;
    }
    var emissive = vec4<f32>(0.0);
    var geometry = vec4<f32>(0.0);
    var surface = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        emissive += weights[k] * textureSampleLevel(impostor_emissive, impostor_sampler, uvs[k],
            record.layer, level);
        geometry += weights[k] * textureSampleLevel(impostor_geometry, impostor_sampler, uvs[k],
            record.layer, level);
        surface += weights[k] * textureSampleLevel(impostor_surface, impostor_sampler, uvs[k],
            record.layer, level);
    }
    let unpremultiply = 1.0 / max(coverage, 1.0e-4);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let geometric = stable_normalize(input.normal_x.xyz * mesh_normal.x
        + input.normal_y.xyz * mesh_normal.y + input.normal_z * mesh_normal.z);
    // The surface behind the card point, at the blended depth.
    let depth = geometry.w * unpremultiply * 2.0 - 1.0;
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    // Roughness, metalness, emission over 16 and ripples.
    let slot_surface = unpack4x8unorm(input.material.z);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    let albedo_color = albedo.rgb * unpremultiply * color.rgb;
    let metalness = slot_surface.y * surface.y * unpremultiply;
    var point: SurfacePoint;
    point.albedo = albedo_color;
    point.opacity = color.a;
    point.emission = slot_surface.z * 16.0 * emissive.a * unpremultiply;
    point.emissive = record.emissive * emissive.rgb * unpremultiply;
    point.occlusion = surface.z * unpremultiply;
    // The mesh's own shadows inside it (decision 060).
    point.sun_shade = 1.0 - saturate(surface.w * unpremultiply);
    point.roughness = max(slot_surface.x * surface.x * unpremultiply, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), albedo_color, metalness);
    point.diffuse_color = albedo_color * (1.0 - metalness);
    point.geometric = geometric;
    point.normal = geometric;
    if slot_surface.w > 0.0 {
        point.normal = rippled(geometric, world, slot_surface.w);
    }
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    // Each frame measures depth along its own view, so blended frames
    // rebuild the surface up to a few tenths of the radius off the mesh
    // that casts its shadow: the sun is looked up a radius toward it, from
    // outside the mesh's own shadow, with a flat receiver plane
    // (`world_dx` and `world_dy` stay zero). Placed objects' impostors
    // cast cards through their centers instead (decision 060): a quarter
    // of the radius clears the sunlit half of their own, and neighbors'
    // cards still shade them.
    let reach = select(1.0, 0.25, (record.flags & IMPOSTOR_HEMISPHERE) != 0u);
    point.shadow_offset = lighting.sun.xyz * (reach * length(input.depth_axis.xyz));
    return widened(point, dpdx(point.normal), dpdy(point.normal));
}

fn impostor_output(input: ImpostorTemporalOutput) -> ImpostorOutput {
    var output: ImpostorOutput;
    output.clip_position = input.clip_position;
    output.frames_ab = input.frames_ab;
    output.frame_c = input.frame_c;
    output.world = input.world;
    output.normal_x = input.normal_x;
    output.normal_y = input.normal_y;
    output.normal_z = input.normal_z;
    output.depth_axis = input.depth_axis;
    output.material = input.material;
    output.batch = input.batch;
    return output;
}

fn shade_impostor(input: ImpostorOutput) -> Shaded {
    return finish(impostor_point(input), vec3<f32>(0.0));
}

fn shade_impostor_lit(input: ImpostorOutput) -> Shaded {
    let point = impostor_point(input);
    return finish(point, local_light(point, input.clip_position.xy));
}

@fragment
fn fs_impostor(input: ImpostorOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_impostor(input).color), 1.0);
}

@fragment
fn fs_impostor_lit(input: ImpostorOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_impostor_lit(input).color), 1.0);
}

@fragment
fn fs_impostor_post(input: ImpostorOutput) -> SceneTargets {
    return scene_targets(shade_impostor(input));
}

@fragment
fn fs_impostor_post_lit(input: ImpostorOutput) -> SceneTargets {
    return scene_targets(shade_impostor_lit(input));
}

@fragment
fn fs_impostor_temporal(input: ImpostorTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_impostor(impostor_output(input)), input.clip_position,
        input.previous);
}

@fragment
fn fs_impostor_temporal_lit(input: ImpostorTemporalOutput) -> TemporalTargets {
    return temporal_targets(shade_impostor_lit(impostor_output(input)), input.clip_position,
        input.previous);
}
