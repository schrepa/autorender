import package::engine::impostors::{impostor_albedo, impostor_emissive, impostor_geometry,
    impostor_sampler, impostor_surface};
import package::engine::materials::MASK_MIP_SCALE;
import package::engine::math::maximum_abs;
import package::engine::mesh::impostor_frames::{IMPOSTOR_CUTOFF, IMPOSTOR_FADING,
    IMPOSTOR_FADING_OUT, IMPOSTOR_FOLIAGE, IMPOSTOR_HEMISPHERE, IMPOSTOR_MASKED, ImpostorRecord,
    impostor_basis, impostor_facing, impostor_fade, impostor_frame_direction, impostor_frames,
    impostor_grid, impostor_index, impostor_met, impostor_over};
import package::engine::mesh::vertex::origin_key;
import package::engine::shading::{BODY_SOLID, BODY_THIN, SurfacePoint};
import package::engine::view::{camera, fade_noise, material_detail, stable_normalize};
import package::face::debug::DEBUG_FADING;
import package::face::foliage::foliage;
import package::face::lighting::{finish, local_light, widened};
import package::face::placed_tiles::placed_impostor;
import package::face::sun::sun_direction;
import package::face::wind::slot_sway;
import package::generated::targets::{DepthViewTargets, depth_view_targets};
import package::meshes::debug::point_targets;
import package::meshes::forward::drawn_pixel;
import package::meshes::shade::{alpha_cutoff, fragment_pixel, leafiness, plant_tint, rippled};
import package::meshes::slots::{instance_colors, instance_poses, instance_previous_poses};
import package::meshes::vertex::{posed, slot_wind};
import package::place::instances::{AffinePose, CardDebugTargets, CardShaded, CardVertex,
    MOVED_SLOT, SlotColor};

@group(0) @binding(0) var<storage, read> impostor_records: array<ImpostorRecord>;

struct CardInput {
    @builtin(vertex_index) vertex: u32,
    @location(2) slot: u32,
    @location(3) batch: u32,
    @location(14) draw: vec4<u32>,
}

struct CardOutput {
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
    @location(9) previous: vec3<f32>,
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
    @location(13) @interpolate(flat) object: vec2<u32>,
}

const IMPOSTOR_CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

fn toward_viewer(point: vec3<f32>) -> vec3<f32> {
    let projection = camera.view_projection;
    let w_row = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    if length(w_row) > 0.5 {
        return stable_normalize(camera.position.xyz - point);
    }
    let near = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let far = camera.inverse_view_projection * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return stable_normalize(near.xyz / near.w - far.xyz / far.w);
}

struct Card {
    local: vec3<f32>,
    output: CardVertex,
}

fn posed_card(vertex: u32, batch: u32, pose: AffinePose, color: SlotColor,
        record: ImpostorRecord) -> Card {
    let model = posed(pose);
    let center = record.sphere.xyz;
    let radius = record.sphere.w;
    let inverse_scale = 1.0 / max(maximum_abs(model[0].xyz),
        max(maximum_abs(model[1].xyz), maximum_abs(model[2].xyz)));
    let x = model[0].xyz * inverse_scale;
    let y = model[1].xyz * inverse_scale;
    let z = model[2].xyz * inverse_scale;
    let yz = cross(y, z);
    let zx = cross(z, x);
    let xy = cross(x, y);
    let orientation = sign(dot(x, yz));
    let toward = toward_viewer((model * vec4<f32>(center, 1.0)).xyz);
    let dir = stable_normalize(vec3<f32>(dot(yz, toward), dot(zx, toward), dot(xy, toward))
        * orientation);
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
    let basis = impostor_basis(dir);
    let half = record.extent;
    let support = dot(abs(dir), half);
    let outline = vec2<f32>(dot(abs(basis.right), half), dot(abs(basis.up), half));
    let corner = IMPOSTOR_CARD[vertex];
    let across = basis.right * (corner.x * outline.x) + basis.up * (corner.y * outline.y);
    let local = center + dir * support + across;
    let world = model * vec4<f32>(local, 1.0);
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
    var swayed = world.xyz;
    let sway = slot_sway(color.flags);
    if sway > 0.0 {
        swayed += slot_wind(model, world.xyz, sway, camera.position.w);
    }
    var output: CardVertex;
    output.clip_position = camera.view_projection * vec4<f32>(swayed, 1.0);
    output.frames_ab = vec4<f32>(uvs[0], uvs[1]);
    output.frame_c = uvs[2];
    output.world = swayed;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    output.material = vec4<u32>(color.color, color.surface,
        packed_frames | (origin_key(model[3].xyz) & 0xff000000u));
    output.batch = batch;
    output.crossings = crossings;
    output.slopes_ab = vec4<f32>(slopes[0], slopes[1]);
    output.slope_c = slopes[2];
    return Card(local, output);
}

fn card_previous(local: vec3<f32>, pose: AffinePose, color: SlotColor) -> vec3<f32> {
    let point = vec4<f32>(local, 1.0);
    var world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    let sway = slot_sway(color.flags);
    if sway > 0.0 {
        let model = posed(pose);
        world = vec4<f32>(world.xyz + slot_wind(model, world.xyz, sway, camera.wind.w), 1.0);
    }
    return (camera.previous_view_projection * world).xyw;
}

fn placed_card(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor) -> CardVertex {
    let made = posed_card(vertex, batch, pose, color, placed_impostor(impostor_index(batch)));
    var output = made.output;
    output.previous = card_previous(made.local, previous, color);
    return output;
}

@vertex
fn vs_card(input: CardInput) -> CardOutput {
    let slot = input.slot & ~MOVED_SLOT;
    let color = instance_colors[slot];
    let made = posed_card(input.vertex, input.batch, instance_poses[slot], color,
        impostor_records[impostor_index(input.batch)]);
    var pose = instance_poses[slot];
    if (input.slot & MOVED_SLOT) != 0u {
        pose = instance_previous_poses[slot];
    }
    let card = made.output;
    return CardOutput(card.clip_position, card.frames_ab, card.frame_c, card.world,
        card.normal_x, card.normal_y, card.normal_z, card.depth_axis, card.material, card.batch,
        card_previous(made.local, pose, color), card.crossings, card.slopes_ab, card.slope_c,
        vec2<u32>(input.draw.x, slot));
}

fn card_vertex(input: CardOutput) -> CardVertex {
    return CardVertex(input.clip_position, input.frames_ab, input.frame_c, input.world,
        input.normal_x, input.normal_y, input.normal_z, input.depth_axis, input.material,
        input.batch, input.previous, input.crossings, input.slopes_ab, input.slope_c);
}

fn card_point(input: CardVertex, record: ImpostorRecord) -> SurfacePoint {
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let shown = vec3<u32>(packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u);
    let frames = impostor_frames(record.flags);
    let masked = (record.flags & IMPOSTOR_MASKED) != 0u;
    let texels = record.flags & (IMPOSTOR_HEMISPHERE - 1u);
    let size = f32(frames * texels);
    let footprint = max(length(dpdx(input.frames_ab.xy)), length(dpdy(input.frames_ab.xy)))
        * size * material_detail();
    let coarsest = f32(countTrailingZeros(texels) - 1u);
    let level = clamp(log2(max(footprint, 1.0e-6)), 0.0, coarsest);
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
    let probed = dot(weights, covered);
    if probed * scale < 1.0 / 255.0 {
        discard;
    }
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
    var cutoff = IMPOSTOR_CUTOFF;
    if masked {
        fragment_pixel = input.clip_position.xy;
        cutoff = alpha_cutoff(cutoff);
    }
    if coverage * scale < cutoff {
        discard;
    }
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
    let depth = clamp(sight * unpremultiply, -1.0, 1.0);
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    let slot_surface = unpack4x8unorm(input.material.z);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    var albedo_color = albedo.rgb * unpremultiply * color.rgb;
    var leaf = 0.0;
    if (record.flags & IMPOSTOR_FOLIAGE) != 0u {
        leaf = leafiness(albedo_color);
        albedo_color *= plant_tint(packed, leaf);
    }
    let metalness = slot_surface.y * surface.y * unpremultiply;
    var point: SurfacePoint;
    point.albedo = albedo_color;
    point.transmission = foliage().x * leaf;
    point.shadow_softness = leaf;
    point.opacity = color.a;
    point.emission = slot_surface.z * 16.0 * emissive.a * unpremultiply;
    point.emissive = record.emissive * emissive.rgb * unpremultiply;
    point.occlusion = surface.z * unpremultiply;
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
    let reach = select(1.0, 0.25, (record.flags & IMPOSTOR_HEMISPHERE) != 0u);
    point.shadow_offset = sun_direction() * (reach * length(input.depth_axis.xyz));
    return widened(point, dpdx(point.normal), dpdy(point.normal));
}

fn mesh_card_point(input: CardVertex) -> SurfacePoint {
    return card_point(input, impostor_records[impostor_index(input.batch)]);
}

fn placed_card_point(input: CardVertex) -> SurfacePoint {
    return card_point(input, placed_impostor(impostor_index(input.batch)));
}

fn card_shaded(input: CardVertex, point: SurfacePoint, light: vec3<f32>) -> CardShaded {
    if impostor_over(input.batch) {
        return CardShaded(finish(point, light), input.clip_position.z);
    }
    let clip = camera.view_projection * vec4<f32>(point.world, 1.0);
    return CardShaded(finish(point, light), saturate(clip.z / clip.w));
}

fn card_body(record: ImpostorRecord) -> u32 {
    return select(BODY_SOLID, BODY_THIN, (record.flags & IMPOSTOR_FOLIAGE) != 0u);
}

fn card_lit(input: CardVertex, point: SurfacePoint, body: u32) -> CardShaded {
    var lit = card_shaded(input, point, local_light(point, input.clip_position.xy));
    lit.shaded.body = body;
    return lit;
}

fn mesh_card_lit(input: CardVertex) -> CardShaded {
    return card_lit(input, mesh_card_point(input),
        card_body(impostor_records[impostor_index(input.batch)]));
}

fn placed_card_lit(input: CardVertex) -> CardShaded {
    return card_lit(input, placed_card_point(input),
        card_body(placed_impostor_records[impostor_index(input.batch)]));
}

fn card_debug(input: CardVertex, point: SurfacePoint, record: u32, object: u32)
        -> CardDebugTargets {
    let shaded = card_shaded(input, point, vec3<f32>(0.0));
    let fading = select(0u, DEBUG_FADING, (input.batch & IMPOSTOR_FADING) != 0u);
    let targets = point_targets(record | fading, object, impostor_index(input.batch), point);
    return CardDebugTargets(targets.ids, targets.count, targets.terrain, shaded.depth);
}

@fragment
fn fs_card(input: CardOutput) -> DepthViewTargets {
    let card = card_vertex(input);
    let shaded = mesh_card_lit(card);
    return depth_view_targets(shaded.shaded, drawn_pixel(input.clip_position, input.previous),
        shaded.depth);
}

@fragment
fn fs_debug_card(input: CardOutput) -> CardDebugTargets {
    let card = card_vertex(input);
    return card_debug(card, mesh_card_point(card), input.object.x, input.object.y);
}
