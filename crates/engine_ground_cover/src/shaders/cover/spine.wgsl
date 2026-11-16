import package::cover::laws::BLADE_SIZE;
import package::engine::math::PI;
import package::engine::view::{camera, detail_across};
import package::generated::cover::{CoverBlade, ShapeRows, cover_facing_angle, cover_lean_wind,
    cover_lean_wind_before, cover_lean_x, cover_lean_y};

struct CoverSpine {
    at: vec3<f32>,
    tangent: vec3<f32>,
}

struct SpineEnds {
    tip: vec3<f32>,
    middle: vec3<f32>,
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

fn cover_leaning(packed: u32, facing: vec3<f32>, curl: f32, wind: f32, blown: vec3<f32>)
        -> vec3<f32> {
    let toward = select(facing, normalize(blown), length(blown) > 1.0e-4);
    let own = facing * curl + vec3<f32>(f32(cover_lean_x(packed)), 0.0,
        f32(cover_lean_y(packed))) / 127.5 - vec3<f32>(1.0, 0.0, 1.0);
    return toward * wind + own * (1.0 - wind);
}

fn spine_middle(root: vec3<f32>, grows: vec3<f32>, height: f32, arc: f32) -> vec3<f32> {
    return root + grows * (height * arc);
}

fn spine_ends(root: vec3<f32>, grows: vec3<f32>, height: f32, reach: vec3<f32>,
        arc: vec3<f32>) -> SpineEnds {
    let bend = length(reach);
    return SpineEnds(root + grows * (height * (1.0 - arc.y * bend * bend))
        + reach * (height * arc.z), spine_middle(root, grows, height, arc.x));
}

fn spine_at(root: vec3<f32>, ends: SpineEnds, t: f32) -> CoverSpine {
    var spine: CoverSpine;
    spine.at = (1.0 - t) * (1.0 - t) * root + 2.0 * t * (1.0 - t) * ends.middle
        + t * t * ends.tip;
    spine.tangent = 2.0 * (1.0 - t) * (ends.middle - root) + 2.0 * t * (ends.tip - ends.middle);
    return spine;
}

fn cover_spine(root: vec3<f32>, grows: vec3<f32>, height: f32, reach: vec3<f32>,
        arc: vec3<f32>, t: f32) -> CoverSpine {
    return spine_at(root, spine_ends(root, grows, height, reach, arc), t);
}

fn blade_ends(blade: CoverBlade, shape: ShapeRows, wind: f32, blown: vec3<f32>) -> SpineEnds {
    let size = unpack2x16float(blade.size & BLADE_SIZE);
    let form = blade.root_color >> 24u;
    let bent = f32((form >> 1u) & 1u);
    let turn = cover_turn(blade.facing);
    let facing = vec3<f32>(cos(turn), 0.0, sin(turn));
    let curl = mix(shape.tips.z, shape.tips.w, fract(turn * 3.1));
    let lean = cover_leaning(blade.lean, facing, curl, wind, blown);
    let leaned = length(lean);
    let bend = min(leaned + shape.bends.y * bent, mix(shape.bends.x, shape.bends.z, bent));
    let direction = select(facing, lean / max(leaned, 1.0e-4), leaned > 1.0e-4);
    let arc = vec3<f32>(blade_arc(blade, shape), shape.arc.yz);
    return spine_ends(blade.root, vec3<f32>(0.0, 1.0, 0.0), size.x, direction * bend, arc);
}

fn blade_arc(blade: CoverBlade, shape: ShapeRows) -> f32 {
    let bent = f32(((blade.root_color >> 24u) >> 1u) & 1u);
    return mix(shape.arc.x, shape.bends.w, bent);
}

fn blade_middle(blade: CoverBlade, shape: ShapeRows) -> vec3<f32> {
    let size = unpack2x16float(blade.size & BLADE_SIZE);
    return spine_middle(blade.root, vec3<f32>(0.0, 1.0, 0.0), size.x, blade_arc(blade, shape));
}

fn blade_side(blade: CoverBlade) -> vec3<f32> {
    let size = unpack2x16float(blade.size & BLADE_SIZE);
    let turn = cover_turn(blade.facing);
    let facing = vec3<f32>(cos(turn), 0.0, sin(turn));
    return cover_side(vec3<f32>(-facing.z, 0.0, facing.x), blade.root, size.y);
}

fn shaped_blade(blade: CoverBlade, shape: ShapeRows, blown: vec3<f32>) -> CoverBlade {
    let wind = cover_lean_wind(blade.lean);
    let before = cover_lean_wind_before(blade.lean);
    let ends = blade_ends(blade, shape, f32(wind) / 127.5, blown);
    let side = blade_side(blade);
    var shaped = blade;
    shaped.tip = ends.tip;
    shaped.before = ends.tip;
    if before != wind {
        shaped.before = blade_ends(blade, shape, f32(before) / 127.5, blown).tip;
    }
    shaped.side_x = side.x;
    shaped.side_z = side.z;
    return shaped;
}

fn view_held(header: CoverBlade) -> bool {
    return all(header.root == camera.position.xyz) && all(header.tip == camera.detail.xyz)
        && all(header.before == vec3<f32>(camera.pixel_angle, camera.texel, camera.pixel_error));
}

fn view_header() -> CoverBlade {
    var header: CoverBlade;
    header.root = camera.position.xyz;
    header.tip = camera.detail.xyz;
    header.before = vec3<f32>(camera.pixel_angle, camera.texel, camera.pixel_error);
    return header;
}
