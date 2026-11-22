use engine_core::gpu_struct;

gpu_struct! {
    pub struct GroundVertex {
        pub pos_tag: [f32; 4],
    }
}

pub const NORMAL_POS_Y: u32 = 0;
pub const NORMAL_POS_Z: u32 = 1;
pub const NORMAL_NEG_Z: u32 = 2;
pub const NORMAL_POS_X: u32 = 3;
pub const NORMAL_NEG_X: u32 = 4;

pub const STYLE_DESERT: u32 = 0;
pub const STYLE_AVENUE: u32 = 1;
pub const STYLE_ARCADE: u32 = 2;
pub const STYLE_TERRACE_TRAV: u32 = 3;
pub const STYLE_TERRACE_SAND: u32 = 4;
pub const STYLE_PLAZA_RADIAL: u32 = 5;
pub const STYLE_STAIR: u32 = 6;

fn encode_tag(mat_id: u32, style: u32, normal_id: u32) -> f32 {
    let bits = 0x8000_0000_u32
        | ((mat_id & 0xF) << 8)
        | ((style & 0xF) << 4)
        | (normal_id & 0x7);
    f32::from_bits(bits)
}

fn push_quad(
    out: &mut Vec<GroundVertex>,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    tag: f32,
) {
    let v = |p: [f32; 3]| GroundVertex {
        pos_tag: [p[0], p[1], p[2], tag],
    };
    out.push(v(p0));
    out.push(v(p1));
    out.push(v(p2));
    out.push(v(p0));
    out.push(v(p2));
    out.push(v(p3));
}

#[derive(Clone, Copy)]
struct Surface {
    mat: u32,
    style: u32,
}

fn push_top_rect(
    out: &mut Vec<GroundVertex>,
    x_span: [f32; 2],
    z_span: [f32; 2],
    y: f32,
    surf: Surface,
) {
    let tag = encode_tag(surf.mat, surf.style, NORMAL_POS_Y);
    push_quad(
        out,
        [x_span[0], y, z_span[0]],
        [x_span[0], y, z_span[1]],
        [x_span[1], y, z_span[1]],
        [x_span[1], y, z_span[0]],
        tag,
    );
}

fn push_ring_frame(
    out: &mut Vec<GroundVertex>,
    center: [f32; 2],
    outer_half: f32,
    inner_half: f32,
    y: f32,
    surf: Surface,
) {
    let ox0 = center[0] - outer_half;
    let ox1 = center[0] + outer_half;
    let oz0 = center[1] - outer_half;
    let oz1 = center[1] + outer_half;
    let ix0 = center[0] - inner_half;
    let ix1 = center[0] + inner_half;
    let iz0 = center[1] - inner_half;
    let iz1 = center[1] + inner_half;
    push_top_rect(out, [ox0, ox1], [oz0, iz0], y, surf);
    push_top_rect(out, [ox0, ox1], [iz1, oz1], y, surf);
    push_top_rect(out, [ox0, ix0], [iz0, iz1], y, surf);
    push_top_rect(out, [ix1, ox1], [iz0, iz1], y, surf);
}

fn push_pit_walls(
    out: &mut Vec<GroundVertex>,
    center: [f32; 2],
    half: f32,
    y_top: f32,
    y_bot: f32,
    surf: Surface,
) {
    let x0 = center[0] - half;
    let x1 = center[0] + half;
    let z0 = center[1] - half;
    let z1 = center[1] + half;
    let t_pz = encode_tag(surf.mat, surf.style, NORMAL_POS_Z);
    push_quad(out, [x0, y_top, z0], [x0, y_bot, z0], [x1, y_bot, z0], [x1, y_top, z0], t_pz);
    let t_nz = encode_tag(surf.mat, surf.style, NORMAL_NEG_Z);
    push_quad(out, [x1, y_top, z1], [x1, y_bot, z1], [x0, y_bot, z1], [x0, y_top, z1], t_nz);
    let t_px = encode_tag(surf.mat, surf.style, NORMAL_POS_X);
    push_quad(out, [x0, y_top, z1], [x0, y_bot, z1], [x0, y_bot, z0], [x0, y_top, z0], t_px);
    let t_nx = encode_tag(surf.mat, surf.style, NORMAL_NEG_X);
    push_quad(out, [x1, y_top, z0], [x1, y_bot, z0], [x1, y_bot, z1], [x1, y_top, z1], t_nx);
}

fn push_grand_stair(out: &mut Vec<GroundVertex>) {
    let steps = 12_u32;
    let z_start = -150.0_f32;
    let z_end = -175.0_f32;
    let dz = (z_end - z_start) / (steps as f32);
    let dy = 3.6 / (steps as f32);
    let x0 = -12.0_f32;
    let x1 = 12.0_f32;
    let surf = Surface { mat: 0, style: STYLE_STAIR };
    let riser_tag = encode_tag(0, STYLE_STAIR, NORMAL_NEG_Z);
    for s in 0..steps {
        let y_hi = -(s as f32) * dy;
        let y_lo = y_hi - dy;
        let za = z_start + (s as f32) * dz;
        let zb = za + dz;
        push_quad(out, [x1, y_hi, za], [x1, y_lo, za], [x0, y_lo, za], [x0, y_hi, za], riser_tag);
        push_top_rect(out, [x0, x1], [zb, za], y_lo, surf);
    }
}

pub fn build_ground_mesh() -> Vec<GroundVertex> {
    let mut out = Vec::with_capacity(256);
    let desert = Surface { mat: 3, style: STYLE_DESERT };
    let avenue = Surface { mat: 0, style: STYLE_AVENUE };
    let arcade = Surface { mat: 3, style: STYLE_ARCADE };
    let t_trav = Surface { mat: 0, style: STYLE_TERRACE_TRAV };
    let t_sand = Surface { mat: 3, style: STYLE_TERRACE_SAND };
    let plaza = Surface { mat: 0, style: STYLE_PLAZA_RADIAL };

    push_top_rect(&mut out, [-4500.0, 4500.0], [-4500.0, -265.0], 0.0, desert);
    push_top_rect(&mut out, [-4500.0, 4500.0], [-145.0, 4500.0], 0.0, desert);
    push_top_rect(&mut out, [-4500.0, -60.0], [-265.0, -145.0], 0.0, desert);
    push_top_rect(&mut out, [60.0, 4500.0], [-265.0, -145.0], 0.0, desert);

    push_top_rect(&mut out, [-25.0, 25.0], [-150.0, 160.0], 0.02, avenue);
    push_top_rect(&mut out, [52.0, 68.0], [-120.0, 40.0], 0.02, arcade);

    push_ring_frame(&mut out, [0.0, -205.0], 60.0, 50.0, 0.0, t_trav);
    push_pit_walls(&mut out, [0.0, -205.0], 50.0, 0.0, -1.2, t_trav);

    push_ring_frame(&mut out, [0.0, -205.0], 50.0, 40.0, -1.2, t_trav);
    push_pit_walls(&mut out, [0.0, -205.0], 40.0, -1.2, -2.4, t_sand);

    push_ring_frame(&mut out, [0.0, -205.0], 40.0, 30.0, -2.4, t_sand);
    push_pit_walls(&mut out, [0.0, -205.0], 30.0, -2.4, -3.6, t_trav);

    push_top_rect(&mut out, [-30.0, 30.0], [-235.0, -175.0], -3.6, plaza);

    push_grand_stair(&mut out);

    out
}
