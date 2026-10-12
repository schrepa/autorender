//! The city's buildings, their roofs and its industry.

use super::*;

pub(super) fn tower(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.5, 0.5),
        facade(30.0),
        ROOF,
        false,
    );
}

pub(super) fn house(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.5, 0.5),
        facade(3.0),
        ROOF,
        false,
    );
}

pub(super) fn midrise(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.5, 0.5),
        facade(12.0),
        ROOF,
        false,
    );
}

pub(super) fn round(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.5, 0.5)],
        12,
        0.0,
        facade(30.0),
        ROOF,
        false,
    );
}

pub(super) fn hex(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.5, 0.5)],
        6,
        0.0,
        facade(24.0),
        ROOF,
        false,
    );
}

pub(super) fn octa(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.5, 0.5)],
        8,
        PI / 8.0,
        facade(30.0),
        ROOF,
        false,
    );
}

pub(super) fn crowned(b: &mut Builder) {
    b.lathe(
        &[
            (0.42, -0.5),
            (0.42, 0.28),
            (0.5, 0.3),
            (0.5, 0.34),
            (0.3, 0.36),
            (0.3, 0.42),
            (0.0, 0.5),
        ],
        12,
        0.0,
        facade(30.0),
        ROOF,
        false,
    );
}

pub(super) fn roof(b: &mut Builder) {
    b.lathe(
        &[(0.707, -0.5), (0.0, 0.5)],
        4,
        FRAC_PI_4,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn spire(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.0, 0.5)],
        8,
        0.0,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn obelisk(b: &mut Builder) {
    b.lathe(
        &[(0.42, -0.5), (0.3, 0.3), (0.0, 0.5)],
        4,
        FRAC_PI_4,
        Uv::FULL,
        ROOF,
        false,
    );
}

pub(super) fn tapered(b: &mut Builder) {
    b.lathe(
        &[(0.707, -0.5), (0.42, 0.5)],
        4,
        FRAC_PI_4,
        facade(30.0),
        ROOF,
        false,
    );
}

pub(super) fn dome(b: &mut Builder) {
    b.lathe(
        &[
            (0.5, -0.5),
            (0.462, -0.117),
            (0.354, 0.207),
            (0.191, 0.424),
            (0.0, 0.5),
        ],
        12,
        0.0,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn setback(b: &mut Builder, shape: Shape) {
    use Shape::*;
    let tiers: &[(f32, f32, f32)] = match shape {
        Setback => &[(0.5, -0.5, 0.0), (0.36, 0.0, 0.3), (0.22, 0.3, 0.5)],
        Setback2 => &[(0.5, -0.5, 0.2), (0.3, 0.2, 0.5)],
        Setback4 => &[
            (0.5, -0.5, -0.05),
            (0.38, -0.05, 0.2),
            (0.26, 0.2, 0.38),
            (0.14, 0.38, 0.5),
        ],
        _ => &[
            (0.5, -0.5, -0.3),
            (0.42, -0.3, -0.1),
            (0.34, -0.1, 0.1),
            (0.26, 0.1, 0.3),
            (0.18, 0.3, 0.5),
        ],
    };
    for &(half, y0, y1) in tiers {
        let floors = math::max((y1 - y0) * FLOORS, 1.0);
        b.cuboid(
            v(-half, y0, -half),
            v(half, y1, half),
            facade(floors),
            ROOF,
            false,
        );
    }
}

pub(super) fn twisted(b: &mut Builder, shape: Shape) {
    use Shape::*;
    let (slabs, twist) = if shape == Twisted {
        (8, FRAC_PI_4)
    } else {
        (12, FRAC_PI_2)
    };
    for slab in 0..slabs {
        let (y0, y1) = (
            slab as f32 / slabs as f32 - 0.5,
            (slab + 1) as f32 / slabs as f32 - 0.5,
        );
        let band = facade(30.0).band(
            1.0 - (slab + 1) as f32 / slabs as f32,
            1.0 - slab as f32 / slabs as f32,
        );
        let start = b.mark();
        b.cuboid(v(-0.35, y0, -0.35), v(0.35, y1, 0.35), band, ROOF, false);
        b.place(
            start,
            Mat4::from_rotation_y(twist * slab as f32 / (slabs - 1) as f32),
        );
    }
}

pub(super) fn ell(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.5, 0.0),
        facade(12.0),
        ROOF,
        false,
    );
    b.cuboid(
        v(-0.5, -0.5, 0.0),
        v(0.0, 0.2, 0.5),
        facade(7.0),
        ROOF,
        false,
    );
}

pub(super) fn tee(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(0.5, 0.0, -0.1),
        facade(5.0),
        ROOF,
        false,
    );
    b.cuboid(
        v(-0.2, -0.5, -0.1),
        v(0.2, 0.5, 0.5),
        facade(20.0),
        ROOF,
        false,
    );
}

pub(super) fn court(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.5),
        v(-0.2, 0.5, 0.5),
        facade(12.0),
        ROOF,
        false,
    );
    b.cuboid(
        v(0.2, -0.5, -0.5),
        v(0.5, 0.5, 0.5),
        facade(12.0),
        ROOF,
        false,
    );
    b.cuboid(
        v(-0.2, -0.5, -0.5),
        v(0.2, 0.3, -0.2),
        facade(9.0),
        ROOF,
        false,
    );
}

pub(super) fn pagoda(b: &mut Builder) {
    b.lathe(
        &[
            (0.354, -0.5),
            (0.354, -0.25),
            (0.707, -0.25),
            (0.283, -0.12),
            (0.283, 0.02),
            (0.566, 0.02),
            (0.212, 0.13),
            (0.212, 0.24),
            (0.424, 0.24),
            (0.1, 0.36),
            (0.0, 0.5),
        ],
        4,
        FRAC_PI_4,
        facade(8.0),
        ROOF,
        false,
    );
}

pub(super) fn water_tower(b: &mut Builder) {
    for (x, z) in [(0.25, 0.25), (-0.25, 0.25), (-0.25, -0.25), (0.25, -0.25)] {
        b.cuboid(
            v(x - 0.04, -0.5, z - 0.04),
            v(x + 0.04, 0.0, z + 0.04),
            BARK,
            BARK,
            false,
        );
    }
    b.lathe(
        &[(0.4, 0.0), (0.4, 0.3), (0.0, 0.5)],
        10,
        0.0,
        Uv::FULL,
        Uv::FULL,
        true,
    );
}

pub(super) fn tank(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.5, 0.2), (0.35, 0.4), (0.0, 0.5)],
        12,
        0.0,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn chimney(b: &mut Builder) {
    b.lathe(
        &[(0.5, -0.5), (0.35, 0.4), (0.5, 0.42), (0.5, 0.5)],
        8,
        0.0,
        Uv::FULL,
        ROOF,
        false,
    );
}

// Roofs along X: one ridge, or steep teeth repeating along Z.
pub(super) fn wedge(b: &mut Builder) {
    b.ridge(-0.5, 0.5, [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)], Uv::FULL);
}

pub(super) fn sawtooth(b: &mut Builder) {
    for tooth in 0..4 {
        let (z0, z1) = (-0.5 + tooth as f32 * 0.25, -0.25 + tooth as f32 * 0.25);
        b.ridge(-0.5, 0.5, [(z0, -0.5), (z1, -0.5), (z0, 0.5)], Uv::FULL);
    }
}
