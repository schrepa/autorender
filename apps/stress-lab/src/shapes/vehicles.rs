//! The city's vehicles and machines.

use super::*;

pub(super) fn car(b: &mut Builder) {
    box_(b, v(-0.5, -0.5, -0.5), v(0.5, 0.0, 0.5), BODY, BODY);
    b.cuboid(
        v(-0.25, 0.0, -0.42),
        v(0.25, 0.45, 0.42),
        WINDOWS,
        BODY,
        false,
    );
}

pub(super) fn bus(b: &mut Builder) {
    box_(b, v(-0.5, -0.5, -0.5), v(0.5, 0.35, 0.5), WINDOWS, BODY);
    b.cuboid(v(-0.48, 0.35, -0.4), v(0.48, 0.5, 0.4), BODY, BODY, false);
}

pub(super) fn wagon(b: &mut Builder) {
    box_(b, v(-0.5, -0.5, -0.5), v(0.5, 0.3, 0.5), WINDOWS, BODY);
    b.cuboid(v(-0.48, 0.3, -0.35), v(0.48, 0.5, 0.35), BODY, BODY, false);
}

pub(super) fn boat(b: &mut Builder) {
    // Hull with a pointed bow toward +x.
    let (y0, y1) = (-0.5, -0.1);
    box_(b, v(-0.5, y0, -0.35), v(0.25, y1, 0.35), BODY, BODY);
    let bow = [v(0.25, y0, -0.35), v(0.25, y0, 0.35), v(0.5, y0, 0.0)];
    let top = [v(0.25, y1, -0.35), v(0.25, y1, 0.35), v(0.5, y1, 0.0)];
    let uv = [[0.0, 1.0], [1.0, 1.0], [0.5, 0.6]];
    b.face(&top, &uv, Vec3::Y);
    b.face(&bow, &uv, Vec3::NEG_Y);
    b.quad([bow[1], bow[2], top[2], top[1]], BODY, v(1.0, 0.0, 1.0));
    b.quad([bow[2], bow[0], top[0], top[2]], BODY, v(1.0, 0.0, -1.0));
    b.cuboid(v(-0.3, y1, -0.22), v(0.1, 0.3, 0.22), WINDOWS, BODY, false);
    b.cuboid(v(-0.2, 0.3, -0.02), v(-0.16, 0.5, 0.02), BODY, BODY, false);
}

pub(super) fn drone(b: &mut Builder) {
    box_(b, v(-0.18, -0.12, -0.18), v(0.18, 0.08, 0.18), BODY, BODY);
    box_(b, v(-0.36, -0.02, -0.03), v(0.36, 0.04, 0.03), BODY, BODY);
    box_(b, v(-0.03, -0.01, -0.36), v(0.03, 0.05, 0.36), BODY, BODY);
    for (x, z) in [(0.35, 0.0), (-0.35, 0.0), (0.0, 0.35), (0.0, -0.35)] {
        let start = b.mark();
        b.lathe(
            &[(0.15, 0.05), (0.15, 0.08)],
            8,
            0.0,
            WINDOWS,
            WINDOWS,
            true,
        );
        b.place(start, Mat4::from_translation(v(x, 0.0, z)));
    }
}

pub(super) fn helicopter(b: &mut Builder) {
    box_(b, v(-0.2, -0.3, -0.18), v(0.25, 0.15, 0.18), WINDOWS, BODY);
    box_(b, v(-0.5, -0.05, -0.04), v(-0.2, 0.05, 0.04), BODY, BODY);
    box_(b, v(-0.5, 0.05, -0.02), v(-0.42, 0.25, 0.02), BODY, BODY);
    box_(b, v(-0.2, -0.5, -0.2), v(0.25, -0.45, -0.16), BODY, BODY);
    box_(b, v(-0.2, -0.5, 0.16), v(0.25, -0.45, 0.2), BODY, BODY);
    box_(b, v(-0.03, 0.15, -0.03), v(0.03, 0.22, 0.03), BODY, BODY);
}

pub(super) fn rotor(b: &mut Builder) {
    // Three flat blades about a shaft along Y.
    b.lathe(&[(0.06, -0.5), (0.06, 0.5)], 8, 0.0, BODY, BODY, true);
    for blade in 0..3 {
        let start = b.mark();
        box_(b, v(0.04, 0.3, -0.07), v(0.49, 0.38, 0.07), BODY, BODY);
        b.place(start, Mat4::from_rotation_y(TAU * blade as f32 / 3.0));
    }
}

pub(super) fn jib(b: &mut Builder) {
    // Arm and counter-arm about the pivot at the origin.
    box_(
        b,
        v(-0.5, 0.3, -0.06),
        v(0.5, 0.45, 0.06),
        Uv::FULL,
        Uv::FULL,
    );
    box_(
        b,
        v(-0.48, 0.05, -0.12),
        v(-0.3, 0.3, 0.12),
        Uv::FULL,
        Uv::FULL,
    );
    box_(b, v(-0.1, -0.5, -0.1), v(0.1, 0.3, 0.1), WINDOWS, BODY);
    box_(b, v(0.3, 0.2, -0.05), v(0.4, 0.3, 0.05), Uv::FULL, Uv::FULL);
    b.lathe(
        &[(0.05, 0.45), (0.0, 0.5)],
        4,
        FRAC_PI_4,
        Uv::FULL,
        Uv::FULL,
        false,
    );
}

pub(super) fn rim(b: &mut Builder) {
    // A wheel in the XY plane: rim segments, spokes and a hub.
    let segments = 16;
    for segment in 0..segments {
        let start = b.mark();
        box_(
            b,
            v(0.42, -0.09, -0.05),
            v(0.48, 0.09, 0.05),
            Uv::FULL,
            Uv::FULL,
        );
        b.place(
            start,
            Mat4::from_rotation_z(TAU * segment as f32 / segments as f32),
        );
    }
    for spoke in 0..8 {
        let start = b.mark();
        box_(
            b,
            v(0.0, -0.01, -0.01),
            v(0.45, 0.01, 0.01),
            Uv::FULL,
            Uv::FULL,
        );
        b.place(start, Mat4::from_rotation_z(TAU * spoke as f32 / 8.0));
    }
    let start = b.mark();
    b.lathe(
        &[(0.05, -0.08), (0.05, 0.08)],
        8,
        0.0,
        Uv::FULL,
        Uv::FULL,
        true,
    );
    b.place(start, Mat4::from_rotation_x(FRAC_PI_2));
}

pub(super) fn cabin(b: &mut Builder) {
    box_(b, v(-0.5, -0.5, -0.5), v(0.5, 0.25, 0.5), WINDOWS, BODY);
    b.lathe(
        &[(0.707, 0.25), (0.0, 0.5)],
        4,
        FRAC_PI_4,
        BODY,
        BODY,
        false,
    );
}
