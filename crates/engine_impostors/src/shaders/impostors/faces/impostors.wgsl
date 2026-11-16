import package::engine::materials::{MASK_MIP_SCALE, SURFACE_EMISSION};
import package::engine::math::maximum_abs;
import package::engine::geometry::declarations::{AffinePose as PoseRows, FADE_LEVEL_MAX,
    fade_hidden, fade_word_level, fade_word_on, fade_word_out, faded_entry_fade,
    faded_slot_slot};
import package::engine::geometry::vertex::pose_matrix;
import package::engine::shading::{SurfacePoint, ViewPixel};
import package::engine::slots::{derived, set_direct_blocked, set_metalness, set_occlusion,
    set_roughness, set_tangled, set_thickness, set_transmission, shade, standard_point};
import package::engine::surfaces::SURFACE_SOLID;
import package::engine::view::{camera, material_detail, stable_normalize, view_pixel};
import package::face::debug::debug_targets;
import package::face::debug_modes::point_source;
import package::face::instances::{AffinePose, DEBUG_FADING, SlotColor};
import package::face::lighting::SURFACE_THIN;
import package::face::sun::sun_direction;
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

fn card_corner(vertex: u32) -> vec2<f32> {
    let right = vertex == 1u || vertex == 2u || vertex == 4u;
    let up = vertex == 2u || vertex == 4u || vertex == 5u;
    return select(vec2<f32>(-1.0), vec2<f32>(1.0), vec2<bool>(right, up));
}

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
    let p = frame_cell(frame, frames) / f32(frames - 1u) * 2.0 - 1.0;
    if hemisphere {
        return hemi_direction(p);
    }
    return octahedral_direction(p);
}

struct NearFrames {
    shown: vec3<u32>,
    weights: vec3<f32>,
}

fn near_frames(d: vec3<f32>, frames: u32, hemisphere: bool) -> NearFrames {
    let grid = (impostor_grid(d, hemisphere) * 0.5 + 0.5) * f32(frames - 1u);
    let cell = min(floor(grid), vec2<f32>(f32(frames - 2u)));
    let f = grid - cell;
    let base = vec2<u32>(cell);
    let first = base.y * frames + base.x;
    if f.x + f.y > 1.0 {
        return NearFrames(vec3<u32>(first + frames + 1u, first + frames, first + 1u),
            vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y));
    }
    return NearFrames(vec3<u32>(first, first + 1u, first + frames),
        vec3<f32>(1.0 - f.x - f.y, f.x, f.y));
}

fn frame_cell(frame: u32, frames: u32) -> vec2<f32> {
    return vec2<f32>(f32(frame % frames), f32(frame / frames));
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

struct CardFrame {
    uv: vec2<f32>,
    slope: vec2<f32>,
    crossing: f32,
}

fn card_frame(shown: u32, frames: u32, hemisphere: bool, dir: vec3<f32>, across: vec3<f32>,
        radius: f32) -> CardFrame {
    let d = impostor_frame_direction(shown, frames, hemisphere);
    let frame_basis = impostor_basis(d);
    let image = vec2<f32>(dot(across, frame_basis.right), -dot(across, frame_basis.up)) / radius;
    return CardFrame((frame_cell(shown, frames) + 0.5 + image * 0.5) / f32(frames),
        vec2<f32>(dot(dir, frame_basis.right), -dot(dir, frame_basis.up)),
        dot(across, d) / radius);
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
    let near = near_frames(dir, frames, hemisphere);
    let shown = near.shown;
    let weights = near.weights;
    let basis = impostor_basis(dir);
    let half = record.extent;
    let support = dot(abs(dir), half);
    let outline = vec2<f32>(dot(abs(basis.right), half), dot(abs(basis.up), half));
    let corner = card_corner(vertex);
    let across = basis.right * (corner.x * outline.x) + basis.up * (corner.y * outline.y);
    let local = center + dir * support + across;
    let world = model * vec4<f32>(local, 1.0);
    let a = card_frame(shown.x, frames, hemisphere, dir, across, radius);
    let b = card_frame(shown.y, frames, hemisphere, dir, across, radius);
    let c = card_frame(shown.z, frames, hemisphere, dir, across, radius);
    let packed_frames = card_frames_with_origin(card_frames_with_third(card_frames_with_second(
        card_frames_with_first(0u, shown.x), shown.y), shown.z),
        origin_key(model[3].xyz) >> 24u);
    var output: CardVertex;
    output.clip_position = camera.view_projection * vec4<f32>(world.xyz, 1.0);
    output.frames_ab = vec4<f32>(a.uv, b.uv);
    output.frame_c = c.uv;
    output.world = world.xyz;
    output.normal_x = vec4<f32>(yz * orientation, weights.x);
    output.normal_y = vec4<f32>(zx * orientation, weights.y);
    output.normal_z = xy * orientation;
    let axis = model[0].xyz * dir.x + model[1].xyz * dir.y + model[2].xyz * dir.z;
    output.depth_axis = vec4<f32>(axis * radius, support / radius);
    output.material = vec4<u32>(color.color, color.surface, packed_frames);
    output.batch = batch;
    output.crossings = vec3<f32>(a.crossing, b.crossing, c.crossing);
    output.slopes_ab = vec4<f32>(a.slope, b.slope);
    output.slope_c = c.slope;
    return Card(local, output);
}

fn card_previous(local: vec3<f32>, pose: AffinePose, color: SlotColor) -> vec3<f32> {
    let point = vec4<f32>(local, 1.0);
    let world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
        dot(pose.row_2, point), 1.0);
    return (camera.previous_view_projection * world).xyw;
}

fn card_level(input: CardVertex, frames: u32, texels: u32) -> f32 {
    let footprint = max(length(dpdx(input.frames_ab.xy)), length(dpdy(input.frames_ab.xy)))
        * f32(frames * texels) * material_detail();
    let coarsest = f32(countTrailingZeros(texels) - 1u);
    return clamp(log2(max(footprint, 1.0e-6)), 0.0, coarsest);
}

struct CardTap {
    crossed: vec2<f32>,
    slope: vec2<f32>,
    cell: vec2<f32>,
    facing: f32,
    crossing: f32,
}

fn card_tap(crossed: vec2<f32>, slope: vec2<f32>, shown: u32, frames: u32, span: f32,
        crossing: f32) -> CardTap {
    return CardTap(crossed, slope, frame_cell(shown, frames) * span, impostor_facing(slope),
        crossing);
}

fn tap_uv(tap: CardTap, at: vec2<f32>, inset: f32, span: f32) -> vec2<f32> {
    return clamp(at, tap.cell + inset, tap.cell + span - inset);
}

fn tap_met(tap: CardTap, sampled: f32, covered: f32) -> f32 {
    return impostor_met(sampled, covered, tap.crossing, tap.facing);
}

struct CardSample {
    emissive: vec3<f32>,
    geometry: vec4<f32>,
    surface: vec4<f32>,
    sight: f32,
}

fn card_sample(held: CardSample, tap: CardTap, uv: vec2<f32>, weight: f32, covered: f32,
        layer: u32, level: f32, emits: bool) -> CardSample {
    var next = held;
    if emits {
        next.emissive += weight * impostor_emissive(uv, layer, level).rgb;
    }
    let sampled = impostor_geometry(uv, layer, level);
    next.geometry += weight * sampled;
    next.sight += weight * tap_met(tap, sampled.w, covered);
    next.surface += weight * impostor_surface(uv, layer, level);
    return next;
}

fn card_point(input: CardVertex, record: ImpostorRecord, fade: u32) -> SurfacePoint {
    let weights = vec3<f32>(input.normal_x.w, input.normal_y.w,
        1.0 - input.normal_x.w - input.normal_y.w);
    let packed = input.material.w;
    let frames = impostor_frames(record.flags);
    let texels = impostor_flags_texels(record.flags);
    let size = f32(frames * texels);
    let level = card_level(input, frames, texels);
    let inset = 0.5 * exp2(ceil(level)) / size;
    let span = 1.0 / f32(frames);
    let scale = 1.0 + MASK_MIP_SCALE * level;
    let layer = record.layer;
    let a = card_tap(input.frames_ab.xy, input.slopes_ab.xy, card_frames_first(packed), frames,
        span, input.crossings.x);
    let b = card_tap(input.frames_ab.zw, input.slopes_ab.zw, card_frames_second(packed), frames,
        span, input.crossings.y);
    let c = card_tap(input.frame_c, input.slope_c, card_frames_third(packed), frames, span,
        input.crossings.z);
    let probe_a = tap_uv(a, a.crossed, inset, span);
    let probe_b = tap_uv(b, b.crossed, inset, span);
    let probe_c = tap_uv(c, c.crossed, inset, span);
    let covered = vec3<f32>(impostor_albedo(probe_a, layer, level).a,
        impostor_albedo(probe_b, layer, level).a, impostor_albedo(probe_c, layer, level).a);
    let probed = dot(weights, covered);
    if probed * scale < 1.0 / 255.0 {
        discard;
    }
    var met = 0.0;
    met += weights.x * tap_met(a, impostor_geometry(probe_a, layer, level).w, covered.x);
    met += weights.y * tap_met(b, impostor_geometry(probe_b, layer, level).w, covered.y);
    met += weights.z * tap_met(c, impostor_geometry(probe_c, layer, level).w, covered.z);
    let along = clamp(met / probed, -1.0, 1.0);
    let uv_a = tap_uv(a, a.crossed + a.slope * (0.5 * span * along), inset, span);
    let uv_b = tap_uv(b, b.crossed + b.slope * (0.5 * span * along), inset, span);
    let uv_c = tap_uv(c, c.crossed + c.slope * (0.5 * span * along), inset, span);
    let texel_a = impostor_albedo(uv_a, layer, level);
    let texel_b = impostor_albedo(uv_b, layer, level);
    let texel_c = impostor_albedo(uv_c, layer, level);
    var albedo = vec4<f32>(0.0);
    albedo += weights.x * texel_a;
    albedo += weights.y * texel_b;
    albedo += weights.z * texel_c;
    let coverage = albedo.a;
    if coverage * scale < impostor_cutoff() {
        discard;
    }
    if fade_hidden(fade, input.clip_position.xy) {
        discard;
    }
    let slot_surface = unpack4x8unorm(input.material.z);
    let emits = any(record.emissive != vec3<f32>(0.0));
    var sampled = CardSample(vec3<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), 0.0);
    sampled = card_sample(sampled, a, uv_a, weights.x, texel_a.a, layer, level, emits);
    sampled = card_sample(sampled, b, uv_b, weights.y, texel_b.a, layer, level, emits);
    sampled = card_sample(sampled, c, uv_c, weights.z, texel_c.a, layer, level, emits);
    let emissive = sampled.emissive;
    let geometry = sampled.geometry;
    let surface = sampled.surface;
    let sight = sampled.sight;
    let unpremultiply = 1.0 / max(coverage, 1.0e-4);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let geometric = stable_normalize(input.normal_x.xyz * mesh_normal.x
        + input.normal_y.xyz * mesh_normal.y + input.normal_z * mesh_normal.z);
    let depth = clamp(sight * unpremultiply, -1.0, 1.0);
    let world = input.world - input.depth_axis.xyz * (input.depth_axis.w - depth);
    let color = vec4<f32>(unpack2x16unorm(input.material.x), unpack2x16unorm(input.material.y));
    let albedo_color = albedo.rgb * unpremultiply * color.rgb;
    var point = standard_point();
    point.base_color = albedo_color;
    point.opacity = color.a;
    point.emissive = albedo_color * (slot_surface.z * SURFACE_EMISSION * surface.w * unpremultiply)
        + record.emissive * emissive * unpremultiply;
    let thin = impostor_flags_thin(record.flags);
    point.kind = select(SURFACE_SOLID, SURFACE_THIN, thin);
    point.geometric = geometric;
    point.normal = geometric;
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.world = world;
    let reach = select(1.0, 0.25, impostor_flags_hemisphere(record.flags));
    point = set_thickness(point, reach * length(input.depth_axis.xyz));
    point = set_tangled(point, select(0.0, 1.0, thin));
    point.normal_dx = dpdx(geometric);
    point.normal_dy = dpdy(geometric);
    point = set_roughness(point, slot_surface.x * surface.x * unpremultiply);
    point = set_metalness(point, slot_surface.y * surface.y * unpremultiply);
    point = set_occlusion(point, surface.z * unpremultiply);
    point = set_transmission(point, 0.0);
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

fn unposed(input: CardVertex, way: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(input.normal_x.xyz, way), dot(input.normal_y.xyz, way),
        dot(input.normal_z, way));
}

fn sun_through(below: f32, softness: f32, grain: f32) -> f32 {
    if softness > 0.0 {
        return exp2(-below / softness);
    }
    return saturate(1.5 - below / grain); // filter: a step no sharper than the depth one texel of the level read spans, and a point within half a texel of the first surface counts as on it
}

struct CardSun {
    frames: u32,
    weights: vec3<f32>,
    turned: f32,
}

fn card_sun_frames(input: CardVertex, record: ImpostorRecord) -> CardSun {
    let toward = unposed(input, sun_direction());
    let turned = length(toward);
    let near = near_frames(toward / turned, impostor_frames(record.flags),
        impostor_flags_hemisphere(record.flags));
    let packed = card_frames_with_third(card_frames_with_second(card_frames_with_first(0u,
        near.shown.x), near.shown.y), near.shown.z);
    return CardSun(packed, near.weights, turned);
}

struct SunLook {
    at: vec3<f32>,
    frames: u32,
    hemisphere: bool,
    span: f32,
    inset: f32,
    layer: u32,
    level: f32,
    meters: f32,
    softness: f32,
    grain: f32,
}

fn sun_reaching(look: SunLook, frame: u32) -> f32 {
    let d = impostor_frame_direction(frame, look.frames, look.hemisphere);
    let basis = impostor_basis(d);
    let cell = frame_cell(frame, look.frames) * look.span;
    let across = vec2<f32>(dot(look.at, basis.right), -dot(look.at, basis.up));
    let uv = clamp(cell + (0.5 + 0.5 * across) * look.span, cell + look.inset,
        cell + look.span - look.inset);
    let top = 2.0 * impostor_geometry(uv, look.layer, look.level).w - 1.0;
    let below = max(top - dot(look.at, d), 0.0) * look.meters;
    return sun_through(below, look.softness, look.grain);
}

fn card_sun(input: CardVertex, record: ImpostorRecord, point: SurfacePoint, sun: CardSun)
        -> f32 {
    let frames = impostor_frames(record.flags);
    let texels = impostor_flags_texels(record.flags);
    let hemisphere = impostor_flags_hemisphere(record.flags);
    let level = card_level(input, frames, texels);
    let inset = 0.5 * exp2(ceil(level)) / f32(frames * texels);
    let span = 1.0 / f32(frames);
    let texel = 2.0 * exp2(level) / f32(texels);
    let axis = unposed(input, input.depth_axis.xyz);
    let reach = length(axis);
    let meters = reach / sun.turned;
    let shown = card_frames_first(input.material.w);
    let seen_from = impostor_frame_direction(shown, frames, hemisphere);
    let held = impostor_basis(seen_from);
    let image = (input.frames_ab.xy * f32(frames) - frame_cell(shown, frames) - 0.5) * 2.0;
    let at = held.right * image.x - held.up * image.y + seen_from * input.crossings.x
        + (axis * input.depth_axis.w + unposed(input, point.world - input.world)) / reach;
    let look = SunLook(at, frames, hemisphere, span, inset, record.layer, level, meters,
        point.direct_softness, texel * meters);
    var reaching = 0.0;
    reaching += sun.weights.x * sun_reaching(look, card_frames_first(sun.frames));
    reaching += sun.weights.y * sun_reaching(look, card_frames_second(sun.frames));
    reaching += sun.weights.z * sun_reaching(look, card_frames_third(sun.frames));
    return reaching;
}

fn card_lit(input: CardVertex, record: ImpostorRecord, point: SurfacePoint, over: bool,
        sun: CardSun) -> CardShaded {
    let open = (1.0 - point.direct_blocked) * card_sun(input, record, point, sun);
    let lit = set_direct_blocked(point, 1.0 - open);
    let shaded = shade(lit, input.clip_position.xy);
    return CardShaded(lit, shaded, card_depth(input, lit, over));
}

fn card_debug(input: CardVertex, point: SurfacePoint, ids: vec2<u32>, index: u32, fade: u32,
        over: bool) -> CardDebugTargets {
    let record = ids.x | select(0u, DEBUG_FADING, card_fading(fade));
    let targets = debug_targets(record, ids.y, index, point.normal, point_source(0u, point));
    return CardDebugTargets(targets.ids, targets.count, targets.words,
        card_depth(input, point, over));
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

fn placed_card_lit(input: CardVertex, impostor: ImpostorRecord, point: SurfacePoint)
        -> CardShaded {
    return card_lit(input, impostor, point, false, card_sun_frames(input, impostor));
}

fn placed_card_depth(input: CardVertex, impostor: ImpostorRecord) -> f32 {
    return card_depth(input, placed_card_point(input, impostor), false);
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    return view_pixel(input.clip_position, vec4<f32>(input.previous.xy, 0.0, input.previous.z));
}

fn placed_card_debug(input: CardVertex, point: SurfacePoint, record: u32, object: u32)
        -> CardDebugTargets {
    return card_debug(input, point, vec2<u32>(record, object), faded_slot_slot(input.batch),
        faded_entry_fade(input.batch), false);
}
