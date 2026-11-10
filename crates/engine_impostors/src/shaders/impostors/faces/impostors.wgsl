import package::engine::materials::{MASK_MIP_SCALE, SURFACE_EMISSION};
import package::engine::math::maximum_abs;
import package::engine::mesh::declarations::{AffinePose as PoseRows, FADE_LEVEL_MAX,
    fade_hidden, fade_word_level, fade_word_on, fade_word_out, faded_entry_fade,
    faded_slot_slot};
import package::engine::mesh::vertex::pose_matrix;
import package::engine::shading::{SurfacePoint, ViewPixel, derived};
import package::engine::slots::shade;
import package::engine::surfaces::SURFACE_SOLID;
import package::engine::view::{camera, material_detail, stable_normalize, view_pixel};
import package::face::debug::debug_targets;
import package::face::debug_modes::point_source;
import package::face::instances::{AffinePose, DEBUG_FADING, SlotColor};
import package::face::lighting::SURFACE_THIN;
import package::generated::impostors::{card_frames_first, card_frames_second,
    card_frames_third, card_frames_with_first, card_frames_with_origin, card_frames_with_second,
    card_frames_with_third, impostor_flags_frames as flags_frames,
    impostor_flags_hemisphere as flags_hemisphere, impostor_flags_texels as flags_texels,
    impostor_flags_thin as flags_thin};
import package::generated::pieces::impostors::piece;
import package::place::impostors::{CardDebugTargets, CardShaded, CardVertex, ImpostorBasis,
    ImpostorRecord};

@group(0) @binding(0) var atlas_albedo: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_emissive: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_geometry: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_surface: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_sampler: sampler;

const IMPOSTOR_CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

fn impostor_frames(flags: u32) -> u32 {
    return flags_frames(flags);
}

fn impostor_flags_texels(flags: u32) -> u32 {
    return flags_texels(flags);
}

fn impostor_flags_hemisphere(flags: u32) -> bool {
    return flags_hemisphere(flags);
}

fn impostor_flags_thin(flags: u32) -> bool {
    return flags_thin(flags);
}

fn impostor_cutoff() -> f32 {
    return piece().cutoff.x;
}

fn octahedral(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    return select(n.xz, (vec2<f32>(1.0) - abs(n.zx)) * sign, n.y < 0.0);
}

fn octahedral_direction(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    let fold = max(-d.y, 0.0);
    d.x += select(fold, -fold, d.x >= 0.0);
    d.z += select(fold, -fold, d.z >= 0.0);
    return normalize(d);
}

fn hemi_octahedral(d: vec3<f32>) -> vec2<f32> {
    let above = vec3<f32>(d.x, max(d.y, 0.0), d.z);
    let n = above / max(abs(above.x) + above.y + abs(above.z), 1.0e-6);
    return vec2<f32>(n.x + n.z, n.x - n.z);
}

fn hemi_direction(p: vec2<f32>) -> vec3<f32> {
    let x = 0.5 * (p.x + p.y);
    let z = 0.5 * (p.x - p.y);
    return normalize(vec3<f32>(x, 1.0 - abs(x) - abs(z), z));
}

fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    if hemisphere {
        return hemi_octahedral(d);
    }
    return octahedral(d);
}

fn impostor_frame_direction(frame: u32, frames: u32, hemisphere: bool) -> vec3<f32> {
    let cell = vec2<f32>(f32(frame % frames), f32(frame / frames));
    let p = cell / f32(frames - 1u) * 2.0 - 1.0;
    if hemisphere {
        return hemi_direction(p);
    }
    return octahedral_direction(p);
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var hint = vec3<f32>(0.0, 1.0, 0.0);
    if abs(d.y) > 0.999 {
        hint = vec3<f32>(0.0, 0.0, -sign(d.y));
    }
    let right = normalize(cross(hint, d));
    return ImpostorBasis(right, cross(d, right));
}

fn impostor_facing(slope: vec2<f32>) -> f32 {
    return sqrt(max(1.0 - dot(slope, slope), 0.01));
}

fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return (2.0 * sampled - covered * (1.0 + crossing)) / facing;
}

fn impostor_albedo(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_albedo, atlas_sampler, at, layer, level);
}

fn impostor_emissive(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_emissive, atlas_sampler, at, layer, level);
}

fn impostor_geometry(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_geometry, atlas_sampler, at, layer, level);
}

fn impostor_surface(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_surface, atlas_sampler, at, layer, level);
}

fn impostor_size() -> u32 {
    return textureDimensions(atlas_albedo).x;
}

fn impostor_levels() -> u32 {
    return textureNumLevels(atlas_albedo);
}

fn origin_key(origin: vec3<f32>) -> u32 {
    let cell = bitcast<vec2<u32>>(vec2<i32>(floor(origin.xz * 4.0)));
    var h = (cell.x * 0x8da6b343u) ^ (cell.y * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return (h ^ (h >> 16u)) & 0xffff0000u;
}

fn posed(pose: AffinePose) -> mat4x4<f32> {
    return pose_matrix(PoseRows(pose.row_0, pose.row_1, pose.row_2));
}

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
    let hemisphere = impostor_flags_hemisphere(record.flags);
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
    var shown = vec3<u32>(0u);
    for (var k = 0u; k < 3u; k++) {
        let frame = corners[k].y * frames + corners[k].x;
        let d = impostor_frame_direction(frame, frames, hemisphere);
        let frame_basis = impostor_basis(d);
        let image = vec2<f32>(dot(across, frame_basis.right), -dot(across, frame_basis.up))
            / radius;
        uvs[k] = (vec2<f32>(corners[k]) + 0.5 + image * 0.5) / f32(frames);
        slopes[k] = vec2<f32>(dot(dir, frame_basis.right), -dot(dir, frame_basis.up));
        crossings[k] = dot(across, d) / radius;
        shown[k] = frame;
    }
    let packed_frames = card_frames_with_origin(card_frames_with_third(card_frames_with_second(
        card_frames_with_first(0u, shown.x), shown.y), shown.z),
        origin_key(model[3].xyz) >> 24u);
    var output: CardVertex;
    output.clip_position = camera.view_projection * vec4<f32>(world.xyz, 1.0);
    output.frames_ab = vec4<f32>(uvs[0], uvs[1]);
    output.frame_c = uvs[2];
    output.world = world.xyz;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    output.material = vec4<u32>(color.color, color.surface, packed_frames);
    output.batch = batch;
    output.crossings = crossings;
    output.slopes_ab = vec4<f32>(slopes[0], slopes[1]);
    output.slope_c = slopes[2];
    return Card(local, output);
}

fn card_previous(local: vec3<f32>, pose: AffinePose, color: SlotColor) -> vec3<f32> {
    let point = vec4<f32>(local, 1.0);
    let world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    return (camera.previous_view_projection * world).xyw;
}

fn card_point(input: CardVertex, record: ImpostorRecord, fade: u32) -> SurfacePoint {
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let shown = vec3<u32>(card_frames_first(packed), card_frames_second(packed),
        card_frames_third(packed));
    let frames = impostor_frames(record.flags);
    let texels = impostor_flags_texels(record.flags);
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
        covered[k] = impostor_albedo(uvs[k], record.layer, level).a;
    }
    let probed = dot(weights, covered);
    if probed * scale < 1.0 / 255.0 {
        discard;
    }
    var met = 0.0;
    for (var k = 0u; k < 3u; k++) {
        let sampled = impostor_geometry(uvs[k], record.layer, level).w;
        met += weights[k] * impostor_met(sampled, covered[k], input.crossings[k], facings[k]);
    }
    let along = clamp(met / probed, -1.0, 1.0);
    var albedo = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        uvs[k] = clamp(crossed[k] + slopes[k] * (0.5 * span * along), cells[k] + inset,
            cells[k] + span - inset);
        let texel = impostor_albedo(uvs[k], record.layer, level);
        covered[k] = texel.a;
        albedo += weights[k] * texel;
    }
    let coverage = albedo.a;
    if coverage * scale < impostor_cutoff() {
        discard;
    }
    if fade_hidden(fade, input.clip_position.xy) {
        discard;
    }
    var emissive = vec4<f32>(0.0);
    var geometry = vec4<f32>(0.0);
    var surface = vec4<f32>(0.0);
    var sight = 0.0;
    for (var k = 0u; k < 3u; k++) {
        emissive += weights[k] * impostor_emissive(uvs[k], record.layer, level);
        let sampled = impostor_geometry(uvs[k], record.layer, level);
        geometry += weights[k] * sampled;
        sight += weights[k] * impostor_met(sampled.w, covered[k], input.crossings[k], facings[k]);
        surface += weights[k] * impostor_surface(uvs[k], record.layer, level);
    }
    let unpremultiply = 1.0 / max(coverage, 1.0e-4);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let geometric = stable_normalize(input.normal_x.xyz * mesh_normal.x
        + input.normal_y.xyz * mesh_normal.y + input.normal_z * mesh_normal.z);
    let depth = clamp(sight * unpremultiply, -1.0, 1.0);
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    let slot_surface = unpack4x8unorm(input.material.z);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    let albedo_color = albedo.rgb * unpremultiply * color.rgb;
    var point: SurfacePoint;
    point.base_color = albedo_color;
    point.opacity = color.a;
    point.emissive = albedo_color * (slot_surface.z * SURFACE_EMISSION * emissive.a * unpremultiply)
        + record.emissive * emissive.rgb * unpremultiply;
    point.occlusion = surface.z * unpremultiply;
    point.roughness = slot_surface.x * surface.x * unpremultiply;
    point.metalness = slot_surface.y * surface.y * unpremultiply;
    point.kind = select(SURFACE_SOLID, SURFACE_THIN, impostor_flags_thin(record.flags));
    point.geometric = geometric;
    point.normal = geometric;
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.world = world;
    let reach = select(1.0, 0.25, impostor_flags_hemisphere(record.flags));
    point.inputs.thickness = reach * length(input.depth_axis.xyz);
    point.inputs.sun_blocked = 1.0 - saturate(surface.w * unpremultiply);
    point.normal_dx = dpdx(geometric);
    point.normal_dy = dpdy(geometric);
    return derived(point);
}

fn card_fading(fade: u32) -> bool {
    return fade_word_on(fade)
        && (fade_word_level(fade) != FADE_LEVEL_MAX || fade_word_out(fade));
}

fn card_depth(input: CardVertex, point: SurfacePoint, over: bool) -> f32 {
    if over {
        return input.clip_position.z;
    }
    let clip = camera.view_projection * vec4<f32>(point.world, 1.0);
    return saturate(clip.z / clip.w);
}

fn card_lit(input: CardVertex, point: SurfacePoint, over: bool) -> CardShaded {
    let shaded = shade(point, input.clip_position.xy);
    return CardShaded(point, shaded, card_depth(input, point, over));
}

fn card_debug(input: CardVertex, point: SurfacePoint, ids: vec2<u32>, index: u32, fade: u32,
        over: bool) -> CardDebugTargets {
    let shaded = card_lit(input, point, over);
    let record = ids.x | select(0u, DEBUG_FADING, card_fading(fade));
    let targets = debug_targets(record, ids.y, index, point.normal, point_source(0u, point));
    return CardDebugTargets(targets.ids, targets.count, targets.words, shaded.depth);
}

fn placed_card_point(input: CardVertex, impostor: ImpostorRecord) -> SurfacePoint {
    return card_point(input, impostor, faded_entry_fade(input.batch));
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor, impostor: ImpostorRecord) -> CardVertex {
    let made = posed_card(vertex, batch, pose, color, impostor);
    var output = made.output;
    output.previous = card_previous(made.local, previous, color);
    return output;
}

fn placed_card_shade(input: CardVertex, impostor: ImpostorRecord) -> CardShaded {
    return card_lit(input, placed_card_point(input, impostor), false);
}

fn placed_card_depth(input: CardVertex, impostor: ImpostorRecord) -> f32 {
    return card_depth(input, placed_card_point(input, impostor), false);
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    return view_pixel(input.clip_position, vec4<f32>(input.previous.xy, 0.0, input.previous.z));
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32, impostor: ImpostorRecord)
        -> CardDebugTargets {
    return card_debug(input, placed_card_point(input, impostor), vec2<u32>(record, object),
        faded_slot_slot(input.batch), faded_entry_fade(input.batch), false);
}
