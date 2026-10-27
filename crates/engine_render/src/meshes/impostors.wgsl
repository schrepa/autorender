// Octahedral impostors: a draw group's mesh baked from a grid of
// directions (its record's frames a side) over the whole sphere, or
// over the upper hemisphere for placed objects, into one
// layer of four texture arrays, drawn for instances too small for its
// texels to show as a card facing the viewer that blends the three frames
// around the view, each read where the line of sight meets the surface
// the frames' depths hold. Texels hold premultiplied coverage: albedo and
// coverage; emissive color and emission mask; normal and depth; roughness
// and metalness factors, occlusion and the sky the surface sees past its
// own mesh. Normals are plain vectors, which filter and average
// across levels as octahedral ones, folded across the lower hemisphere's
// seams, cannot.
//
// Frames, records and their directions: `impostor_frames.wgsl`.

// The flag of records whose material is divided into rooms
// (`far_rooms`), above the masked one.
const IMPOSTOR_ROOMS: u32 = 0x40000u;
// The flag of records whose meshes shade as foliage, above the frames.
const IMPOSTOR_FOLIAGE: u32 = 0x1000000u;

// One record per draw group; groups without an impostor never draw one.
@group(1) @binding(11) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(1) @binding(12) var impostor_albedo: texture_2d_array<f32>;
@group(1) @binding(13) var impostor_emissive: texture_2d_array<f32>;
@group(1) @binding(14) var impostor_geometry: texture_2d_array<f32>;
@group(1) @binding(15) var impostor_surface: texture_2d_array<f32>;
@group(1) @binding(16) var impostor_sampler: sampler;

// Baking: every frame renders the mesh orthographically into its cell, the
// sphere filling the cell. A pass draws a tile of the grid's frames into
// the cells of its own target, instance `i` the tile's `i`th frame along
// its rows, which bakes copy into the layer; the frames' nearest depths
// draw the whole grid as one tile.
struct ImpostorBake {
    // Mesh-space center and radius of the frames' sphere.
    sphere: vec4<f32>,
    // The record's flags.
    flags: vec4<u32>,
    // The tile: the grid's column and row of its first frame, its frames
    // along each row, and the cells along each side of the pass's target.
    tile: vec4<u32>,
}
@group(2) @binding(15) var<uniform> impostor_bake: ImpostorBake;
// Each frame's nearest depth toward its viewer, at four times the layer's
// texels, drawn before the colors.
@group(2) @binding(16) var impostor_shade: texture_depth_2d;

fn bake_hemisphere() -> bool {
    return (impostor_bake.flags.x & IMPOSTOR_HEMISPHERE) != 0u;
}

// The share of the sky above the horizon that mesh-space `position`, with
// unit normal `normal`, sees past its own mesh, weighted by how squarely
// its surface faces each direction: the frames' nearest depths are a depth
// map from every frame's direction. Cards shade the sun by
// it, as the mesh's own shadows would darken its insides.
fn impostor_self_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let size = textureDimensions(impostor_shade).x;
    let frames = impostor_frames(impostor_bake.flags.x);
    // A texel of the depths, in sphere radii.
    let texel = 2.0 * f32(frames) / f32(size);
    // Moved a texel and a half along its normal, a surface clears its own
    // depth.
    let p = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w
        + normal * (1.5 * texel);
    let hemisphere = bake_hemisphere();
    var seen = 0.0;
    var total = 0.0;
    // Eight directions a side at most: finer grids look up every other
    // frame, or fewer.
    let stride = (frames + 7u) / 8u;
    for (var frame = 0u; frame < frames * frames; frame++) {
        if (frame % frames) % stride != 0u || (frame / frames) % stride != 0u {
            continue;
        }
        let d = impostor_frame_direction(frame, frames, hemisphere);
        let facing = dot(normal, d);
        if d.y < 0.0 || facing <= 0.0 {
            continue;
        }
        let basis = impostor_basis(d);
        let image = vec2<f32>(dot(p, basis.right), -dot(p, basis.up));
        let cell = vec2<f32>(f32(frame % frames), f32(frame / frames));
        let atlas = (cell + 0.5 + image * 0.5) / f32(frames);
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

fn bake_vertex(position: vec3<f32>, encoded_normal: vec2<f32>, uv: vec2<f32>, instance: u32)
        -> BakeOutput {
    let frames = impostor_frames(impostor_bake.flags.x);
    let tile = impostor_bake.tile;
    // The instance's cell in the pass's target, from the top left, and its
    // frame in the grid.
    let cell = vec2<u32>(instance % tile.z, instance / tile.z);
    let frame = (tile.y + cell.y) * frames + tile.x + cell.x;
    let d = impostor_frame_direction(frame, frames, bake_hemisphere());
    let basis = impostor_basis(d);
    let local = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w;
    let image = vec2<f32>(dot(local, basis.right), -dot(local, basis.up));
    let depth = clamp(dot(local, d), -1.0, 1.0);
    let drawn = (vec2<f32>(cell) + 0.5 + image * 0.5) / f32(tile.w);
    var output: BakeOutput;
    // Nearer the frame's viewer is greater, as reversed depth keeps it.
    output.clip_position = vec4<f32>(drawn.x * 2.0 - 1.0, 1.0 - drawn.y * 2.0,
        0.5 + 0.5 * depth, 1.0);
    output.normal_u = vec4<f32>(octahedral_normal(encoded_normal), uv.x);
    output.local_v = vec4<f32>(position, uv.y);
    output.depth = depth;
    return output;
}

@vertex
fn vs_impostor_bake(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @builtin(instance_index) instance: u32) -> BakeOutput {
    return bake_vertex(position, normal, uv, instance);
}

@vertex
fn vs_impostor_bake_extended(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @location(9) color: vec4<f32>, @location(10) uv1: vec2<f32>,
        @builtin(instance_index) instance: u32) -> BakeExtendedOutput {
    let vertex = bake_vertex(position, normal, uv, instance);
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
    // Bakes sample their materials' own mips, as filtering at a frame's
    // texels reads them; world-mapped maps bake through the mesh's own
    // coordinates, which repeat about as often as its maker meant it to
    // stand, filtered at the frame's texels as fragments filter them at
    // their pixels, cards taking over where a texel spans a pixel.
    let texels = material_texels(uv, uv1, 0.0);
    let local = input.local_v.xyz;
    let local_dx = dpdx(local);
    let local_dy = dpdy(local);
    let geometric = geometric_normal(input.normal_u.xyz, front);
    var normal = geometric;
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        normal = mapped_normal(geometric, local_dx, local_dy, texels.normal_dx,
            texels.normal_dy, texels.normal, material_maps.params.x);
    }
    var albedo = texels.base.rgb * color.rgb;
    if material_flag(VARIANT_TOP_LAYER, TOP_LAYER) {
        // The top layer settles on the faces turned up as fragments settle
        // it, its noise's breakup averaged over the texel.
        let up = stable_normalize(normal).y;
        let amount = (top_layer_amount(up, 0.25, texels) + top_layer_amount(up, 0.5, texels)
            + top_layer_amount(up, 0.75, texels)) / 3.0;
        albedo = mix(albedo, material_maps.top.rgb, amount);
        normal = mix(normal, geometric, 0.5 * amount);
    }
    var emissive = vec3<f32>(0.0);
    var occlusion = 1.0;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        emissive = texels.emissive;
        occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
    }
    bake_mask(texels, color);
    var targets: BakeTargets;
    targets.albedo = vec4<f32>(albedo, 1.0);
    // Texture alpha masks emission.
    targets.emissive = vec4<f32>(emissive, texels.base.a);
    targets.geometry = vec4<f32>(stable_normalize(normal) * 0.5 + 0.5, 0.5 + 0.5 * input.depth);
    targets.surface = vec4<f32>(texels.factors, occlusion,
        impostor_self_visibility(local, geometric));
    return targets;
}

// The frames' nearest depths alone.
@fragment
fn fs_impostor_shade(input: BakeOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, uv, 0.0), vec4<f32>(1.0));
}

@fragment
fn fs_impostor_shade_extended(input: BakeExtendedOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, input.uv1, 0.0), input.color);
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

// A card point as fragments read it. Its line of sight is measured in
// sphere radii toward the viewer from the plane through the center facing
// the viewer, which it crosses where each blended frame shows at its layer
// coordinates here, at the frame's own depth (its crossing); along it, each
// frame's image moves by the frame's slope per radius.
struct ImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Layer coordinates, in the three blended frames, of where the line of
    // sight crosses the plane through the center.
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    // The card point in world space.
    @location(2) world: vec3<f32>,
    // Columns of the normal transform (cofactors, signed by the pose's
    // orientation); w: the first two frames' blend weights.
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    // World offset per radius toward the viewer along the line of sight,
    // and how far toward the viewer the card stands, in radii.
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    // Linear color and opacity as four 16-bit fractions, the packed surface,
    // and the three frames (eight bits each).
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    // Each frame's depth toward its own viewer, in radii, where the line of
    // sight crosses the plane.
    @location(10) crossings: vec3<f32>,
    // How each frame's image of the line of sight moves per radius toward
    // the viewer, in the frame's image coordinates (-1 to 1 across it).
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
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
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
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
    // Reversed depth: the near plane is 1, the far plane 0.
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
    let frames = impostor_frames(record.flags);
    let grid = (impostor_grid(dir, hemisphere) * 0.5 + 0.5) * f32(frames - 1u);
    let cell = min(floor(grid), vec2<f32>(f32(frames - 2u)));
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
    // Where the card point's line of sight crosses the plane through the
    // center facing the viewer, from the center.
    let across = basis.right * (corner.x * outline.x) + basis.up * (corner.y * outline.y);
    let local = center + dir * support + across;
    let world = model * vec4<f32>(local, 1.0);
    // Each frame's image of the line of sight: where that crossing shows,
    // how it moves per radius toward the viewer, and its depth toward the
    // frame's own viewer, which fragments follow to the surface the frames'
    // depths hold.
    var uvs: array<vec2<f32>, 3>;
    var slopes: array<vec2<f32>, 3>;
    var crossings = vec3<f32>(0.0);
    var packed_frames = 0u;
    for (var k = 0u; k < 3u; k++) {
        let frame = corners[k].y * frames + corners[k].x;
        let d = impostor_frame_direction(frame, frames, hemisphere);
        let frame_basis = impostor_basis(d);
        let image = vec2<f32>(dot(across, frame_basis.right), -dot(across, frame_basis.up))
            / radius;
        uvs[k] = (vec2<f32>(corners[k]) + 0.5 + image * 0.5) / f32(frames);
        slopes[k] = vec2<f32>(dot(dir, frame_basis.right), -dot(dir, frame_basis.up));
        crossings[k] = dot(across, d) / radius;
        packed_frames |= frame << (8u * k);
    }
    let color = colors[slot];
    // Cards sway in the wind as their meshes do.
    var swayed = world.xyz;
    let sway = slot_sway(color.flags);
    if sway > 0.0 {
        swayed += slot_wind(model, world.xyz, sway, camera.position.w);
    }
    var output: ImpostorOutput;
    output.clip_position = camera.view_projection * vec4<f32>(swayed, 1.0);
    output.frames_ab = vec4<f32>(uvs[0], uvs[1]);
    output.frame_c = uvs[2];
    output.world = swayed;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    // The frames fill the word's low 24 bits: the origin key's top byte
    // fits above them.
    output.material = vec4<u32>(color.color, color.surface,
        packed_frames | (origin_key(model[3].xyz) & 0xff000000u));
    output.batch = input.batch;
    output.crossings = crossings;
    output.slopes_ab = vec4<f32>(slopes[0], slopes[1]);
    output.slope_c = slopes[2];
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
    // Only moving slots read their previous poses.
    var pose = poses[slot];
    if (input.slot & MOVED_SLOT) != 0u {
        pose = previous_poses[slot];
    }
    var world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    // Last frame's sway, at its time.
    let sway = slot_sway(colors[slot].flags);
    if sway > 0.0 {
        let model = pose_matrix(pose);
        world = vec4<f32>(world.xyz + slot_wind(model, world.xyz, sway, camera.wind.w), 1.0);
    }
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
    output.crossings = vertex.crossings;
    output.slopes_ab = vertex.slopes_ab;
    output.slope_c = vertex.slope_c;
    return output;
}

// How squarely a frame faces the line of sight whose image in it moves by
// `slope` per radius toward the viewer: the cosine between the two, kept
// from vanishing where views from below see placed objects' frames from
// the horizon.
fn impostor_facing(slope: vec2<f32>) -> f32 {
    return sqrt(max(1.0 - dot(slope, slope), 0.01));
}

// How far toward the viewer, in radii from the center, the line of sight
// meets the surface a frame's texel holds, times the texel's coverage:
// `sampled` its premultiplied depth (half of one more than the depth, times
// coverage), `covered` its coverage, `crossing` the frame's depth where the
// line of sight crosses the plane through the center, and `facing` how
// squarely the frame faces it.
fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return (2.0 * sampled - covered * (1.0 + crossing)) / facing;
}

// The surface an impostor fragment shows: the three frames blended by
// weight and coverage, unpremultiplied, on the slot's color and surface.
// Each frame is read where the card point's line of sight meets the
// surface the frames' depths hold, as their depths where it crosses the
// plane through the center place it (parallax), so the three show the
// same point as the view turns; discards fragments the blend leaves under
// half covered, scaling coverage up on coarser levels as masked materials
// do.
fn impostor_point(input: ImpostorOutput) -> SurfacePoint {
    let record = impostor_records[impostor_index(input.batch)];
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let shown = vec3<u32>(packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u);
    let frames = impostor_frames(record.flags);
    let masked = (record.flags & IMPOSTOR_MASKED) != 0u;
    // The level whose texels match the pixels the display shows (its
    // render size's share of them: the mip bias), and half a texel of the
    // coarser level sampled, which keeps each frame's samples in its cell.
    // The atlas is its frames' texels across and has their levels down to
    // two texels a frame (`AtlasShape`), which the record holds.
    let texels = record.flags & (IMPOSTOR_HEMISPHERE - 1u);
    let size = f32(frames * texels);
    let footprint = max(length(dpdx(input.frames_ab.xy)), length(dpdy(input.frames_ab.xy)))
        * size * material_detail();
    let coarsest = f32(countTrailingZeros(texels) - 1u);
    let level = clamp(log2(max(footprint, 1.0e-6)), 0.0, coarsest); // filter: the atlas's level whose texels match the pixel
    let inset = 0.5 * exp2(ceil(level)) / size;
    let span = 1.0 / f32(frames);
    let scale = 1.0 + MASK_MIP_SCALE * level;
    let crossed = array<vec2<f32>, 3>(input.frames_ab.xy, input.frames_ab.zw, input.frame_c);
    let slopes = array<vec2<f32>, 3>(input.slopes_ab.xy, input.slopes_ab.zw, input.slope_c);
    var cells: array<vec2<f32>, 3>;
    var facings = vec3<f32>(0.0);
    var uvs: array<vec2<f32>, 3>;
    var covered = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        cells[k] = vec2<f32>(f32(shown[k] % frames), f32(shown[k] / frames)) * span;
        facings[k] = impostor_facing(slopes[k]);
        uvs[k] = clamp(crossed[k], cells[k] + inset, cells[k] + span - inset);
        covered[k] = textureSampleLevel(impostor_albedo, impostor_sampler, uvs[k],
            record.layer, level).a;
    }
    // Most of a card around its mesh is empty: those fragments stop before
    // reading the rest.
    let probed = dot(weights, covered);
    if probed * scale < 1.0 / 255.0 {
        discard;
    }
    // Where the line of sight meets the surfaces the frames hold there, by
    // weight and coverage: each frame is then read where it shows that
    // point.
    var met = 0.0;
    for (var k = 0u; k < 3u; k++) {
        let sampled = textureSampleLevel(impostor_geometry, impostor_sampler, uvs[k],
            record.layer, level).w;
        met += weights[k] * impostor_met(sampled, covered[k], input.crossings[k], facings[k]);
    }
    let along = clamp(met / probed, -1.0, 1.0);
    var albedo = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        uvs[k] = clamp(crossed[k] + slopes[k] * (0.5 * span * along), cells[k] + inset,
            cells[k] + span - inset);
        let texel = textureSampleLevel(impostor_albedo, impostor_sampler, uvs[k], record.layer,
            level);
        covered[k] = texel.a;
        albedo += weights[k] * texel;
    }
    let coverage = albedo.a;
    // Masked materials' cards dissolve as their meshes do with hashed
    // cutoffs (`alpha_cutoff`).
    var cutoff = 0.5;
    if masked {
        fragment_pixel = input.clip_position.xy;
        cutoff = alpha_cutoff(cutoff);
    }
    if coverage * scale < cutoff {
        discard;
    }
    // Fading, a card covers the pixels its fade shows, what
    // it switches with the rest.
    if (input.batch & IMPOSTOR_FADING) != 0u {
        let shows = fade_noise(input.clip_position.xy) < impostor_fade(input.batch);
        if shows == ((input.batch & IMPOSTOR_FADING_OUT) != 0u) {
            discard;
        }
    }
    var emissive = vec4<f32>(0.0);
    var geometry = vec4<f32>(0.0);
    var surface = vec4<f32>(0.0);
    var sight = 0.0;
    for (var k = 0u; k < 3u; k++) {
        emissive += weights[k] * textureSampleLevel(impostor_emissive, impostor_sampler, uvs[k],
            record.layer, level);
        let sampled = textureSampleLevel(impostor_geometry, impostor_sampler, uvs[k],
            record.layer, level);
        geometry += weights[k] * sampled;
        sight += weights[k] * impostor_met(sampled.w, covered[k], input.crossings[k], facings[k]);
        surface += weights[k] * textureSampleLevel(impostor_surface, impostor_sampler, uvs[k],
            record.layer, level);
    }
    let unpremultiply = 1.0 / max(coverage, 1.0e-4);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let geometric = stable_normalize(input.normal_x.xyz * mesh_normal.x
        + input.normal_y.xyz * mesh_normal.y + input.normal_z * mesh_normal.z);
    // The surface behind the card point: where its line of sight meets the
    // surfaces the frames show, in radii toward the viewer from the center.
    let depth = clamp(sight * unpremultiply, -1.0, 1.0);
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    // Roughness, metalness, emission over 16 and ripples.
    let slot_surface = unpack4x8unorm(input.material.z);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    var albedo_color = albedo.rgb * unpremultiply * color.rgb;
    // Foliage's leaves, tinted by their plant as their meshes are.
    var leaf = 0.0;
    if (record.flags & IMPOSTOR_FOLIAGE) != 0u {
        leaf = leafiness(albedo_color);
        albedo_color *= plant_tint(packed, leaf);
    }
    let metalness = slot_surface.y * surface.y * unpremultiply;
    var point: SurfacePoint;
    point.albedo = albedo_color;
    point.leaf = leaf;
    point.crown = leaf;
    // Cards of what shades as foliage are sheets, as its meshes are.
    point.body = select(BODY_SOLID, BODY_THIN, (record.flags & IMPOSTOR_FOLIAGE) != 0u);
    point.opacity = color.a;
    point.emission = slot_surface.z * 16.0 * emissive.a * unpremultiply;
    point.emissive = record.emissive * emissive.rgb * unpremultiply;
    point.occlusion = surface.z * unpremultiply;
    // The mesh's own shadows inside it.
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
    // The frames' depths rebuild the surface near, not on, the mesh that
    // casts its shadow: the sun is looked up a radius toward it, from
    // outside the mesh's own shadow, with a flat receiver plane
    // (`world_dx` and `world_dy` stay zero). Placed objects' impostors
    // cast cards through their centers instead: a quarter
    // of the radius clears the sunlit half of their own, and neighbors'
    // cards still shade them.
    let reach = select(1.0, 0.25, (record.flags & IMPOSTOR_HEMISPHERE) != 0u);
    point.shadow_offset = lighting.sun.xyz * (reach * length(input.depth_axis.xyz));
    if (record.flags & IMPOSTOR_ROOMS) != 0u {
        // As on meshes, the window share is what emits by texture alpha.
        let window = select(0.0, saturate(emissive.a * unpremultiply), slot_surface.z > 0.0);
        point = far_rooms(point, window);
    }
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
    output.crossings = input.crossings;
    output.slopes_ab = input.slopes_ab;
    output.slope_c = input.slope_c;
    return output;
}

// A card fragment's shading and the depth of the surface it shows: cards
// stand in front of their meshes' bounds, and the ground and their
// neighbors hide what they would hide of the meshes, so objects sunk into
// slopes show only what stands out of them.
struct ImpostorShaded {
    shaded: Shaded,
    depth: f32,
}

// Cards fading over their steady meshes keep their card's
// depth, in front of the mesh.
fn impostor_shaded(input: ImpostorOutput, point: SurfacePoint, light: vec3<f32>)
        -> ImpostorShaded {
    if impostor_over(input.batch) {
        return ImpostorShaded(finish(point, light), input.clip_position.z);
    }
    let clip = camera.view_projection * vec4<f32>(point.world, 1.0);
    return ImpostorShaded(finish(point, light), saturate(clip.z / clip.w));
}

fn shade_impostor(input: ImpostorOutput) -> ImpostorShaded {
    return impostor_shaded(input, impostor_point(input), vec3<f32>(0.0));
}

fn shade_impostor_lit(input: ImpostorOutput) -> ImpostorShaded {
    let point = impostor_point(input);
    return impostor_shaded(input, point, local_light(point, input.clip_position.xy));
}

struct ImpostorColor {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

struct ImpostorSceneTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
    @location(2) surface: vec4<u32>,
    @builtin(frag_depth) depth: f32,
}

struct ImpostorTemporalTargets {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @location(2) glow: vec2<f32>,
    @location(3) surface: vec4<u32>,
    @builtin(frag_depth) depth: f32,
}

fn impostor_color(shaded: ImpostorShaded, input: ImpostorOutput) -> ImpostorColor {
    return ImpostorColor(vec4<f32>(encoded(shaded.shaded.color, input.clip_position.xy), 1.0),
        shaded.depth);
}

fn impostor_scene_targets(shaded: ImpostorShaded, input: ImpostorOutput)
        -> ImpostorSceneTargets {
    let targets = scene_targets(shaded.shaded, input.clip_position.xy);
    return ImpostorSceneTargets(targets.color, targets.glow, targets.surface, shaded.depth);
}

fn impostor_temporal_targets(shaded: ImpostorShaded, input: ImpostorTemporalOutput)
        -> ImpostorTemporalTargets {
    let targets = temporal_targets(shaded.shaded, input.clip_position, input.previous);
    return ImpostorTemporalTargets(targets.color, targets.motion, targets.glow, targets.surface,
        shaded.depth);
}

@fragment
fn fs_impostor(input: ImpostorOutput) -> ImpostorColor {
    return impostor_color(shade_impostor(input), input);
}

@fragment
fn fs_impostor_lit(input: ImpostorOutput) -> ImpostorColor {
    return impostor_color(shade_impostor_lit(input), input);
}

@fragment
fn fs_impostor_post(input: ImpostorOutput) -> ImpostorSceneTargets {
    return impostor_scene_targets(shade_impostor(input), input);
}

@fragment
fn fs_impostor_post_lit(input: ImpostorOutput) -> ImpostorSceneTargets {
    return impostor_scene_targets(shade_impostor_lit(input), input);
}

@fragment
fn fs_impostor_temporal(input: ImpostorTemporalOutput) -> ImpostorTemporalTargets {
    return impostor_temporal_targets(shade_impostor(impostor_output(input)), input);
}

@fragment
fn fs_impostor_temporal_lit(input: ImpostorTemporalOutput) -> ImpostorTemporalTargets {
    return impostor_temporal_targets(shade_impostor_lit(impostor_output(input)), input);
}
