//! The city's street furniture and landmarks.

use super::*;

pub(super) fn tv_tower(b: &mut Builder) {
    b.lathe(
        &[
            (0.12, -0.5),
            (0.08, 0.1),
            (0.3, 0.13),
            (0.4, 0.18),
            (0.32, 0.25),
            (0.06, 0.28),
            (0.03, 0.45),
            (0.0, 0.5),
        ],
        12,
        0.0,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn lamp(b: &mut Builder) {
    b.lathe(&[(0.06, -0.5), (0.04, 0.42)], 6, 0.0, BODY, BODY, false);
    box_(b, v(0.0, 0.38, -0.03), v(0.4, 0.42, 0.03), BODY, BODY);
    box_(b, v(0.3, 0.33, -0.08), v(0.5, 0.39, 0.08), WINDOWS, WINDOWS);
}

pub(super) fn antenna(b: &mut Builder) {
    b.lathe(
        &[(0.25, -0.5), (0.05, 0.35)],
        4,
        FRAC_PI_4,
        Uv::FULL,
        Uv::FULL,
        false,
    );
    box_(
        b,
        v(-0.3, -0.1, -0.02),
        v(0.3, -0.06, 0.02),
        Uv::FULL,
        Uv::FULL,
    );
    box_(
        b,
        v(-0.02, 0.1, -0.22),
        v(0.02, 0.14, 0.22),
        Uv::FULL,
        Uv::FULL,
    );
    b.lathe(
        &[(0.0, 0.33), (0.1, 0.42), (0.0, 0.5)],
        6,
        0.0,
        WINDOWS,
        WINDOWS,
        false,
    );
}

pub(super) fn beacon(b: &mut Builder) {
    b.lathe(&[(0.08, -0.5), (0.08, -0.05)], 6, 0.0, BODY, BODY, false);
    box_(
        b,
        v(-0.5, -0.05, -0.08),
        v(0.5, 0.05, 0.08),
        WINDOWS,
        WINDOWS,
    );
    box_(
        b,
        v(-0.08, -0.04, -0.5),
        v(0.08, 0.06, 0.5),
        WINDOWS,
        WINDOWS,
    );
    b.lathe(
        &[(0.0, 0.05), (0.2, 0.25), (0.0, 0.5)],
        8,
        0.0,
        WINDOWS,
        WINDOWS,
        false,
    );
}

pub(super) fn billboard(b: &mut Builder) {
    box_(b, v(-0.5, 0.0, -0.04), v(0.5, 0.5, 0.04), Uv::FULL, BODY);
    box_(b, v(-0.33, -0.5, -0.03), v(-0.27, 0.0, 0.03), BODY, BODY);
    box_(b, v(0.27, -0.5, -0.03), v(0.33, 0.0, 0.03), BODY, BODY);
}

pub(super) fn fountain(b: &mut Builder) {
    b.lathe(
        &[
            (0.5, -0.5),
            (0.5, -0.4),
            (0.2, -0.4),
            (0.2, -0.2),
            (0.3, -0.2),
            (0.3, -0.15),
            (0.1, -0.15),
            (0.08, 0.2),
            (0.2, 0.25),
            (0.0, 0.3),
        ],
        10,
        0.0,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn pillar(b: &mut Builder) {
    b.cuboid(
        v(-0.15, -0.5, -0.15),
        v(0.15, 0.35, 0.15),
        Uv::FULL,
        Uv::FULL,
        false,
    );
    box_(
        b,
        v(-0.5, 0.35, -0.25),
        v(0.5, 0.5, 0.25),
        Uv::FULL,
        Uv::FULL,
    );
}

pub(super) fn deck(b: &mut Builder) {
    box_(b, v(-0.5, -0.5, -0.5), v(0.5, 0.0, 0.5), Uv::FULL, Uv::FULL);
    b.cuboid(v(-0.5, 0.0, 0.45), v(0.5, 0.5, 0.5), BODY, BODY, false);
    b.cuboid(v(-0.5, 0.0, -0.5), v(0.5, 0.5, -0.45), BODY, BODY, false);
}

pub(super) fn arch(b: &mut Builder) {
    b.cuboid(
        v(-0.5, -0.5, -0.3),
        v(-0.3, 0.3, 0.3),
        facade(4.0),
        ROOF,
        false,
    );
    b.cuboid(
        v(0.3, -0.5, -0.3),
        v(0.5, 0.3, 0.3),
        facade(4.0),
        ROOF,
        false,
    );
    box_(b, v(-0.5, 0.3, -0.3), v(0.5, 0.5, 0.3), facade(2.0), ROOF);
}

pub(super) fn stadium(b: &mut Builder) {
    // Outer wall, a rim and seating sloping down to the field.
    let sides = 16;
    for index in 0..sides {
        let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
        let at = |radius: f32, y: f32, step: f32| {
            let (sin, cos) = math::sin_cos(angle(step));
            v(radius * cos, y, radius * sin)
        };
        let (sin, cos) = math::sin_cos(angle(0.5));
        let radial = v(cos, 0.0, sin);
        b.quad(
            [
                at(0.5, -0.5, 1.0),
                at(0.5, -0.5, 0.0),
                at(0.5, 0.1, 0.0),
                at(0.5, 0.1, 1.0),
            ],
            facade(4.0),
            radial,
        );
        b.quad(
            [
                at(0.45, 0.1, 1.0),
                at(0.45, 0.1, 0.0),
                at(0.5, 0.1, 0.0),
                at(0.5, 0.1, 1.0),
            ],
            ROOF,
            Vec3::Y,
        );
        b.quad(
            [
                at(0.3, -0.4, 0.0),
                at(0.3, -0.4, 1.0),
                at(0.45, 0.1, 1.0),
                at(0.45, 0.1, 0.0),
            ],
            Uv::FULL,
            Vec3::Y - radial,
        );
        b.face(
            &[v(0.0, -0.4, 0.0), at(0.3, -0.4, 1.0), at(0.3, -0.4, 0.0)],
            &[[0.5, 0.5], [0.5, 0.9], [0.9, 0.9]],
            Vec3::Y,
        );
    }
}
