import package::cover::faces::cover::{cover_card_look, cover_cell, cover_shape};
import package::cover::laws::{BLADE_HALF, BLADE_SIZE, BLADE_UPPER, CARD_CELL, CARD_JOINTS,
    CARD_MIRRORED, CARD_TINT_RANGE, COVER_DRAW_VERTICES, FLAT_JOINTS, FORM_BLUNT, FORM_FLAT,
    STRAIGHT_JOINTS, blade_taper};
import package::engine::math::PI;
import package::engine::view::{camera, detail_across};
import package::generated::cover::{CoverBlade, CoverCard, CoverFrame, ShapeRows,
    cover_facing_angle, cover_facing_tilt_x, cover_facing_tilt_y, cover_lean_wind,
    cover_lean_wind_before, cover_lean_x, cover_lean_y};
import package::generated::kinds::KIND_COVER;

@group(2) @binding(0) var<storage, read> cover_blades: array<CoverBlade>;
@group(2) @binding(1) var<uniform> cover_frame: CoverFrame;
@group(2) @binding(2) var<storage, read> cover_cards: array<CoverCard>;

struct BladeVertex {
    blade: CoverBlade,
    size: vec2<f32>,
    along: f32,
    across: f32,
    flat: bool,
}

const FLAT_SPANS: f32 = f32(FLAT_JOINTS - 1u);
const STRAIGHT_SPANS: f32 = f32(STRAIGHT_JOINTS - 1u);

fn blade_vertex(instance: u32, vertex: u32) -> BladeVertex {
    var at: BladeVertex;
    let within = vertex % COVER_DRAW_VERTICES;
    let joint = f32(within / 2u);
    at.across = f32(within & 1u) * 2.0 - 1.0;
    at.flat = vertex >= COVER_DRAW_VERTICES;
    if at.flat {
        at.blade = cover_blades[cover_frame.blades.x - 1u - instance];
        at.along = joint / FLAT_SPANS;
    } else {
        at.blade = cover_blades[instance];
        let piece = select(1.0, 0.5, (at.blade.size & BLADE_HALF) != 0u);
        let start = select(0.0, 0.5, (at.blade.size & BLADE_UPPER) != 0u);
        at.along = start + piece * joint / STRAIGHT_SPANS;
    }
    at.size = unpack2x16float(at.blade.size & BLADE_SIZE);
    return at;
}

struct CoverVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

const COVER_WHITE: u32 = 0xffffffffu;
const COVER_UNGROUNDED: f32 = -1.0;
const CARD_LAST: u32 = 2u * CARD_JOINTS - 1u;
const CARD_SPANS: f32 = f32(CARD_JOINTS - 1u);

struct CoverPoint {
    world: vec3<f32>,
    normal: vec3<f32>,
    across: f32,
}

fn shown_form(at: BladeVertex) -> u32 {
    let form = at.blade.root_color >> 24u;
    return (form >> 2u) | select(0u, FORM_BLUNT, (form & 1u) != 0u)
        | select(0u, FORM_FLAT, at.flat);
}

struct CoverSpine {
    at: vec3<f32>,
    tangent: vec3<f32>,
}

fn cover_side(side: vec3<f32>, root: vec3<f32>, width: f32) -> vec3<f32> {
    let ray = normalize(root - camera.position.xyz);
    let needed = detail_across(root) * camera.pixel_error;
    let seen = width * length(cross(side, ray));
    let facing = vec3<f32>(-ray.z, 0.0, ray.x);
    if seen >= needed || dot(facing, facing) <= 0.0 {
        return side;
    }
    let turned = normalize(select(-facing, facing, dot(side, facing) >= 0.0));
    return normalize(mix(side, turned, 1.0 - seen / needed));
}

fn cover_turn(packed: u32) -> f32 {
    return f32(cover_facing_angle(packed)) / 65535.0 * (2.0 * PI);
}

fn cover_leaning(packed: u32, facing: vec3<f32>, curl: f32, wind: f32) -> vec3<f32> {
    let blown = vec3<f32>(cover_frame.wind.x, 0.0, cover_frame.wind.y);
    let toward = select(facing, normalize(blown), length(blown) > 1.0e-4);
    let own = facing * curl + vec3<f32>(f32(cover_lean_x(packed)), 0.0,
        f32(cover_lean_y(packed))) / 127.5 - vec3<f32>(1.0, 0.0, 1.0);
    return toward * wind + own * (1.0 - wind);
}

fn cover_spine(root: vec3<f32>, grows: vec3<f32>, height: f32, reach: vec3<f32>,
        arc: vec3<f32>, t: f32) -> CoverSpine {
    let bend = length(reach);
    let tip = root + grows * (height * (1.0 - arc.y * bend * bend)) + reach * (height * arc.z);
    let middle = root + grows * (height * arc.x);
    var spine: CoverSpine;
    spine.at = (1.0 - t) * (1.0 - t) * root + 2.0 * t * (1.0 - t) * middle + t * t * tip;
    spine.tangent = 2.0 * (1.0 - t) * (middle - root) + 2.0 * t * (tip - middle);
    return spine;
}

fn cover_point(at: BladeVertex, shape: ShapeRows, wind: f32) -> CoverPoint {
    let blade = at.blade;
    let size = at.size;
    let form = blade.root_color >> 24u;
    let bent = f32((form >> 1u) & 1u);
    let turn = cover_turn(blade.facing);
    let facing = vec3<f32>(cos(turn), 0.0, sin(turn));
    let side = cover_side(vec3<f32>(-facing.z, 0.0, facing.x), blade.root, size.y);
    let curl = mix(shape.tips.z, shape.tips.w, fract(turn * 3.1));
    let lean = cover_leaning(blade.lean, facing, curl, wind);
    let leaned = length(lean);
    let bend = min(leaned + shape.bends.y * bent, mix(shape.bends.x, shape.bends.z, bent));
    let direction = select(facing, lean / max(leaned, 1.0e-4), leaned > 1.0e-4);
    let arc = vec3<f32>(mix(shape.arc.x, shape.bends.w, bent), shape.arc.yz);
    let t = at.along;
    let spine = cover_spine(blade.root, vec3<f32>(0.0, 1.0, 0.0), size.x, direction * bend, arc,
        t);
    var point: CoverPoint;
    point.across = at.across * blade_taper(shape.arc.w, shape.tips.xy, shown_form(at), t);
    point.world = spine.at + side * (0.5 * point.across * size.y);
    point.normal = normalize(cross(side, spine.tangent));
    return point;
}

fn cover_color(blade: CoverBlade, along: f32, reach: f32) -> vec3<f32> {
    let root = unpack4x8unorm(blade.root_color).rgb;
    let tip = unpack4x8unorm(blade.tip_color).rgb;
    return mix(root * root, tip * tip, smoothstep(0.0, reach, along));
}

fn cover_varyings(instance: u32, vertex: u32) -> CoverVaryings {
    let shape = cover_shape();
    let at = blade_vertex(instance, vertex);
    let wind = cover_lean_wind(at.blade.lean);
    let before = cover_lean_wind_before(at.blade.lean);
    let point = cover_point(at, shape, f32(wind) / 127.5);
    var previous = point.world;
    if before != wind {
        previous = cover_point(at, shape, f32(before) / 127.5).world;
    }
    var output: CoverVaryings;
    output.clip_position = camera.view_projection * vec4<f32>(point.world, 1.0);
    output.normal_u = vec4<f32>(point.normal, at.along);
    output.world_v = vec4<f32>(point.world, point.across);
    output.material = vec4<u32>(COVER_WHITE, COVER_WHITE,
        (at.blade.tip_color >> 24u) | (shown_form(at) << 24u), 0u);
    output.previous = (camera.previous_view_projection * vec4<f32>(previous, 1.0)).xyw;
    output.color = vec4<f32>(cover_color(at.blade, at.along, shape.steps.z), COVER_UNGROUNDED);
    if cover_frame.wind.z != 0.0 {
        output.color.w = at.size.x;
        output.uv1 = vec2<f32>(f32(cover_facing_tilt_x(at.blade.facing)),
            f32(cover_facing_tilt_y(at.blade.facing))) / 255.0 * 2.0 - 1.0;
    }
    output.object = vec4<u32>(0u, 0u, 0u, KIND_COVER);
    return output;
}

@vertex
fn vs_cover(@builtin(instance_index) instance: u32, @builtin(vertex_index) vertex: u32)
        -> CoverVaryings {
    return cover_varyings(instance, vertex);
}

fn cover_depth(instance: u32, vertex: u32) -> CoverVaryings {
    let shape = cover_shape();
    let at = blade_vertex(instance, vertex);
    let wind = cover_lean_wind(at.blade.lean);
    let point = cover_point(at, shape, f32(wind) / 127.5);
    var output: CoverVaryings;
    output.clip_position = camera.view_projection * vec4<f32>(point.world, 1.0);
    return output;
}

struct CastVertex {
    instance: u32,
    vertex: u32,
}

fn cast_vertex(instance: u32, vertex: u32, locals: u32) -> CastVertex {
    let slot = vertex / COVER_DRAW_VERTICES;
    let local = slot % locals;
    return CastVertex(slot / locals + instance,
        local * COVER_DRAW_VERTICES + vertex % COVER_DRAW_VERTICES);
}

@vertex
fn vs_cover_cast(@builtin(instance_index) instance: u32, @builtin(vertex_index) vertex: u32)
        -> CoverVaryings {
    let placed = cast_vertex(instance, vertex, 2u);
    return cover_depth(placed.instance, placed.vertex);
}

struct CardCorner {
    along: f32,
    across: f32,
    uv: vec2<f32>,
}

fn card_side(side: vec3<f32>, root: vec3<f32>, turned: f32) -> vec3<f32> {
    let ray = root - camera.position.xyz;
    let level = vec3<f32>(-ray.z, 0.0, ray.x);
    let away = dot(level, level);
    if away <= 0.0 || turned <= 0.0 {
        return side;
    }
    let broad = level * inverseSqrt(away);
    let shown = abs(dot(side, broad));
    let toward = select(-broad, broad, dot(side, broad) >= 0.0);
    return normalize(mix(toward, side, saturate(shown / turned)));
}

fn card_corner(card: CoverCard, vertex: u32) -> CardCorner {
    let within = min(vertex, CARD_LAST);
    let cell = cover_cell(card.cell & CARD_CELL);
    let right = f32(within & 1u);
    let mirrored = (card.cell & CARD_MIRRORED) != 0u;
    let across = mix(-cell.span.x, cell.span.y, select(right, 1.0 - right, mirrored));
    var corner: CardCorner;
    corner.along = mix(-cell.span.z, 1.0, f32(within / 2u) / CARD_SPANS);
    corner.across = across * select(1.0, -1.0, mirrored);
    let sideways = vec2<f32>(-cell.axis.w / cell.span.w, cell.axis.z * cell.span.w);
    corner.uv = cell.axis.xy + corner.along * cell.axis.zw + across * sideways;
    return corner;
}

fn card_point(card: CoverCard, corner: CardCorner, shape: ShapeRows, form: vec4<f32>,
        wind: f32) -> CoverPoint {
    let size = unpack2x16float(card.size);
    let turn = cover_turn(card.facing);
    let facing = vec3<f32>(cos(turn), 0.0, sin(turn));
    let side = card_side(vec3<f32>(-facing.z, 0.0, facing.x), card.root, form.x);
    let rise = f32(cover_facing_tilt_x(card.facing)) / 255.0 * (0.5 * PI);
    let grows = facing * cos(rise) + vec3<f32>(0.0, sin(rise), 0.0);
    let lean = cover_leaning(card.lean, facing, 0.0, wind);
    let reach = lean * (size.y * min(1.0, shape.bends.x / max(length(lean), 1.0e-4)));
    var spine = cover_spine(card.root, grows, size.x, reach, shape.arc.xyz, corner.along);
    if corner.along < 0.0 {
        spine.at = card.root + grows * (size.x * corner.along);
        spine.tangent = grows;
    }
    var point: CoverPoint;
    point.world = spine.at + side * (corner.across * size.x);
    let face = normalize(cross(side, spine.tangent));
    point.normal = normalize(face + side * (form.y * sign(corner.across)));
    return point;
}

fn card_varyings(instance: u32, vertex: u32, moved: bool) -> CoverVaryings {
    let shape = cover_shape();
    let form = cover_card_look().form;
    let bin = vertex / COVER_DRAW_VERTICES;
    let card = cover_cards[bin * cover_frame.blades.y + instance];
    let corner = card_corner(card, vertex % COVER_DRAW_VERTICES);
    let wind = cover_lean_wind(card.lean);
    let before = cover_lean_wind_before(card.lean);
    let point = card_point(card, corner, shape, form, f32(wind) / 127.5);
    let tint = unpack4x8unorm(card.tint).rgb;
    var output: CoverVaryings;
    output.clip_position = camera.view_projection * vec4<f32>(point.world, 1.0);
    output.normal_u = vec4<f32>(point.normal, corner.uv.x);
    output.world_v = vec4<f32>(point.world, corner.uv.y);
    output.material = vec4<u32>(COVER_WHITE, COVER_WHITE, card.tint >> 24u, 0u);
    if moved {
        var previous = point.world;
        if before != wind {
            previous = card_point(card, corner, shape, form, f32(before) / 127.5).world;
        }
        output.previous = (camera.previous_view_projection * vec4<f32>(previous, 1.0)).xyw;
    }
    output.color = vec4<f32>(tint * tint / CARD_TINT_RANGE, saturate(corner.along));
    output.object = vec4<u32>(0u, 0u, 0u, KIND_COVER);
    return output;
}

@vertex
fn vs_card(@builtin(instance_index) instance: u32, @builtin(vertex_index) vertex: u32)
        -> CoverVaryings {
    return card_varyings(instance, vertex, true);
}

@vertex
fn vs_card_cast(@builtin(instance_index) instance: u32, @builtin(vertex_index) vertex: u32)
        -> CoverVaryings {
    let placed = cast_vertex(instance, vertex, max(cover_frame.blades.z, 1u));
    return card_varyings(placed.instance, placed.vertex, false);
}
