//! The metro's vehicles and machines.

use super::*;

pub(super) fn bus(b: &mut Builder) {
    b.block(v(-2.7, 0.35, -0.65), v(2.7, 1.9, 0.65), PAINT);
    b.block(v(-2.5, 1.05, -0.66), v(2.4, 1.7, 0.66), WINDOW);
    b.block(v(2.6, 0.9, -0.55), v(2.71, 1.7, 0.55), WINDOW);
    lights(b, 2.7, 0.6, 0.5);
    wheels(b, &[-1.8, 1.8], 0.66, 0.35);
}

pub(super) fn double_decker(b: &mut Builder) {
    b.block(v(-2.5, 0.35, -0.65), v(2.5, 3.1, 0.65), PAINT);
    for y in [1.0, 2.2] {
        b.block(v(-2.3, y, -0.66), v(2.3, y + 0.6, 0.66), WINDOW);
    }
    lights(b, 2.5, 0.6, 0.5);
    wheels(b, &[-1.6, 1.6], 0.66, 0.35);
}

pub(super) fn truck(b: &mut Builder) {
    b.block(v(-2.4, 0.4, -0.6), v(0.9, 2.2, 0.6), PAINT);
    b.block(v(0.95, 0.4, -0.58), v(2.1, 1.7, 0.58), TRIM);
    b.block(v(1.5, 1.1, -0.59), v(2.11, 1.6, 0.59), WINDOW);
    lights(b, 2.1, 0.7, 0.45);
    wheels(b, &[-1.6, -0.9, 1.4], 0.62, 0.38);
}

pub(super) fn mixer(b: &mut Builder) {
    b.block(v(-2.2, 0.4, -0.55), v(0.7, 0.8, 0.55), GRILLE);
    b.block(v(0.8, 0.4, -0.58), v(2.0, 1.7, 0.58), PAINT);
    b.block(v(1.4, 1.1, -0.59), v(2.01, 1.6, 0.59), WINDOW);
    lights(b, 2.0, 0.7, 0.45);
    wheels(b, &[-1.5, -0.8, 1.4], 0.6, 0.38);
}

pub(super) fn mixer_drum(b: &mut Builder) {
    let start = b.mark();
    b.lathe(
        &[(0.3, -1.4), (0.9, -0.6), (0.9, 0.5), (0.4, 1.3)],
        10,
        0.0,
        PAINT,
        TRIM,
        true,
    );
    b.place(start, Mat4::from_rotation_z(FRAC_PI_2 - 0.2));
}

pub(super) fn train(b: &mut Builder, model: Model) {
    let front = if model == Model::Locomotive { 2.2 } else { 2.5 };
    b.block(v(-2.5, 0.3, -0.85), v(front, 2.1, 0.85), TRIM);
    b.block(v(-2.3, 1.2, -0.86), v(front - 0.2, 1.8, 0.86), WINDOW);
    if model == Model::Locomotive {
        // A raked nose: a low block and a slope up to the roof.
        b.cuboid(v(2.2, 0.3, -0.85), v(2.8, 1.3, 0.85), TRIM, TRIM, true);
        b.quad(
            [
                v(2.8, 1.3, 0.85),
                v(2.8, 1.3, -0.85),
                v(2.2, 2.1, -0.85),
                v(2.2, 2.1, 0.85),
            ],
            WINDOW,
            v(1.0, 0.8, 0.0),
        );
        for z in [-0.85, 0.85] {
            b.poly(
                &[v(2.2, 1.3, z), v(2.8, 1.3, z), v(2.2, 2.1, z)],
                TRIM,
                v(0.0, 0.0, z),
            );
        }
        b.block(v(2.79, 0.6, -0.6), v(2.83, 0.8, -0.3), HEADLIGHT);
        b.block(v(2.79, 0.6, 0.3), v(2.83, 0.8, 0.6), HEADLIGHT);
    }
    b.block(v(-2.3, 0.0, -0.7), v(front - 0.2, 0.3, 0.7), GRILLE);
}

pub(super) fn speedboat(b: &mut Builder) {
    hull(b, 2.2, 0.8, 0.6);
    b.block(v(-0.4, 0.6, -0.5), v(0.4, 1.0, 0.5), WINDOW);
    b.block(v(-1.8, 0.6, -0.6), v(-0.6, 0.8, 0.6), INTERIOR);
}

pub(super) fn ferry(b: &mut Builder) {
    hull(b, 6.0, 2.0, 1.0);
    b.block(v(-4.0, 1.0, -1.6), v(3.0, 2.4, 1.6), PAINT);
    b.block(v(-3.8, 1.5, -1.61), v(2.8, 2.1, 1.61), WINDOW);
    b.block(v(-2.0, 2.4, -1.2), v(1.0, 3.3, 1.2), PAINT);
    b.block(v(-1.8, 2.7, -1.21), v(0.8, 3.1, 1.21), WINDOW);
    b.lathe(
        &[(0.35, 3.3), (0.35, 4.3)],
        8,
        0.0,
        SIREN_RED,
        SIREN_RED,
        false,
    );
}

pub(super) fn barge(b: &mut Builder) {
    hull(b, 7.0, 2.2, 0.9);
    for index in 0..3 {
        let x = -5.0 + index as f32 * 3.2;
        b.block(v(x, 0.9, -1.6), v(x + 2.8, 1.8, 1.6), GRILLE);
    }
    b.block(v(4.6, 0.9, -1.4), v(6.2, 2.6, 1.4), PAINT);
    b.block(v(4.7, 1.9, -1.41), v(6.1, 2.4, 1.41), WINDOW);
}

pub(super) fn sailboat(b: &mut Builder) {
    hull(b, 2.6, 0.9, 0.5);
    b.bar(v(0.3, 0.5, 0.0), v(0.3, 5.5, 0.0), 0.05, 6, TRIM);
    b.block(v(-1.2, 0.5, -0.6), v(0.0, 0.9, 0.6), INTERIOR);
}

pub(super) fn sail(b: &mut Builder) {
    let cloth = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.poly(
        &[v(0.0, 0.0, 0.0), v(-2.4, 0.0, 0.0), v(0.0, 4.6, 0.0)],
        cloth,
        Vec3::Z,
    );
}

pub(super) fn drone(b: &mut Builder) {
    b.block(v(-0.35, -0.12, -0.35), v(0.35, 0.12, 0.35), PAINT);
    for (x, z) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        b.bar(v(0.0, 0.05, 0.0), v(x * 0.7, 0.08, z * 0.7), 0.04, 4, TRIM);
        let start = b.mark();
        b.lathe(&[(0.32, 0.0), (0.32, 0.04)], 10, 0.0, GRILLE, WINDOW, true);
        b.place(start, Mat4::from_translation(v(x * 0.7, 0.1, z * 0.7)));
    }
    b.block(v(0.34, -0.05, -0.1), v(0.37, 0.05, 0.1), SIREN_BLUE);
}

pub(super) fn heli(b: &mut Builder) {
    b.block(v(-0.9, -0.6, -0.7), v(1.2, 0.6, 0.7), PAINT);
    b.block(v(0.6, -0.3, -0.71), v(1.21, 0.5, 0.71), WINDOW);
    b.block(v(-3.2, -0.15, -0.15), v(-0.9, 0.2, 0.15), PAINT);
    b.block(v(-3.2, 0.2, -0.05), v(-2.8, 1.1, 0.05), PAINT);
    for z in [-0.7, 0.7] {
        b.block(v(-0.8, -1.0, z - 0.06), v(1.2, -0.92, z + 0.06), TRIM);
        b.block(v(0.0, -0.92, z - 0.03), v(0.1, -0.6, z + 0.03), TRIM);
    }
    b.bar(v(0.0, 0.6, 0.0), v(0.0, 0.9, 0.0), 0.07, 6, TRIM);
}

pub(super) fn rotor(b: &mut Builder) {
    for blade in 0..4 {
        let start = b.mark();
        b.block(v(0.1, -0.03, -0.12), v(3.6, 0.03, 0.12), GRILLE);
        b.place(start, Mat4::from_rotation_y(FRAC_PI_2 * blade as f32));
    }
    b.lathe(&[(0.2, -0.1), (0.2, 0.1)], 8, 0.0, TRIM, TRIM, true);
}

pub(super) fn blimp(b: &mut Builder) {
    let profile: Vec<(f32, f32)> = (0..=10)
        .map(|index| {
            let angle = PI * index as f32 / 10.0;
            let (sin, cos) = math::sin_cos(angle);
            (2.2 * sin, -7.0 * cos)
        })
        .collect();
    let start = b.mark();
    b.lathe(&profile, 12, 0.0, PAINT, PAINT, false);
    b.place(start, Mat4::from_rotation_z(-FRAC_PI_2));
    b.block(v(-1.2, -2.8, -0.5), v(1.4, -2.1, 0.5), WINDOW);
    for angle in [0.0, FRAC_PI_2, PI, -FRAC_PI_2] {
        let start = b.mark();
        b.block(v(-7.4, 0.0, -0.06), v(-5.0, 2.2, 0.06), PAINT);
        b.place(start, Mat4::from_rotation_x(angle));
    }
}

pub(super) fn excavator_body(b: &mut Builder) {
    b.block(v(-1.6, 0.0, -1.2), v(1.6, 0.6, -0.6), TIRE);
    b.block(v(-1.6, 0.0, 0.6), v(1.6, 0.6, 1.2), TIRE);
    b.block(v(-1.3, 0.6, -1.0), v(1.3, 1.6, 1.0), PAINT);
    b.block(v(0.3, 1.6, 0.1), v(1.3, 2.6, 1.0), PAINT);
    b.block(v(0.35, 1.8, 0.09), v(1.25, 2.5, 1.01), WINDOW);
}

pub(super) fn excavator_boom(b: &mut Builder) {
    // From its pivot at the origin up and out along +X.
    b.block(v(0.0, -0.25, -0.2), v(3.6, 0.25, 0.2), PAINT);
    b.block(v(3.2, -0.25, -0.2), v(3.8, 1.6, 0.2), PAINT);
}

pub(super) fn excavator_stick(b: &mut Builder) {
    b.block(v(-0.18, -2.6, -0.16), v(0.18, 0.0, 0.16), PAINT);
    b.block(v(-0.5, -3.1, -0.4), v(0.4, -2.5, 0.4), GRILLE);
}

pub(super) fn crane_mast(b: &mut Builder) {
    lattice_column(b, 1.1, 40.0);
}

pub(super) fn crane_jib(b: &mut Builder) {
    // A lattice arm along +X, a counter-arm, a cab and weights.
    let lattice = |length: f32| Map::Rect(0.0, 0.0, length / 2.0, 1.0);
    let (low, high) = (-0.45, 0.45);
    for (y0, y1, z) in [(low, high, 0.45), (low, high, -0.45)] {
        b.quad(
            [
                v(-7.0, y0, z),
                v(22.0, y0, z),
                v(22.0, y1, z),
                v(-7.0, y1, z),
            ],
            lattice(29.0),
            v(0.0, 0.0, z),
        );
    }
    b.quad(
        [
            v(-7.0, high, 0.45),
            v(22.0, high, 0.45),
            v(22.0, high, -0.45),
            v(-7.0, high, -0.45),
        ],
        lattice(29.0),
        Vec3::Y,
    );
    b.block(
        v(-7.0, -1.8, -0.9),
        v(-4.5, -0.45, 0.9),
        Map::Rect(0.02, 0.02, 0.1, 0.1),
    );
    b.block(
        v(-1.0, -1.6, -0.8),
        v(0.8, -0.45, 0.8),
        Map::Rect(0.02, 0.02, 0.1, 0.1),
    );
    b.lathe(
        &[(0.6, high), (0.0, 4.0)],
        4,
        FRAC_PI_4,
        lattice(2.0),
        lattice(1.0),
        false,
    );
}

pub(super) fn ferris_rim(b: &mut Builder) {
    // A wheel in the XY plane: two rims, spokes and a hub.
    let segments = 24;
    for z in [-0.6, 0.6] {
        for segment in 0..segments {
            let start = b.mark();
            b.block(v(6.2, -0.8, z - 0.08), v(6.5, 0.8, z + 0.08), CHROME);
            b.place(
                start,
                Mat4::from_rotation_z(TAU * segment as f32 / segments as f32),
            );
        }
    }
    for spoke in 0..12 {
        let start = b.mark();
        b.block(v(0.0, -0.05, -0.05), v(6.3, 0.05, 0.05), STEEL);
        b.place(start, Mat4::from_rotation_z(TAU * spoke as f32 / 12.0));
        // Lights along the rim.
        let start = b.mark();
        b.block(v(6.45, -0.12, 0.62), v(6.6, 0.12, 0.72), WARM);
        b.place(
            start,
            Mat4::from_rotation_z(TAU * (spoke as f32 + 0.5) / 12.0),
        );
    }
    let start = b.mark();
    b.lathe(&[(0.5, -0.8), (0.5, 0.8)], 10, 0.0, CHROME, CHROME, true);
    b.place(start, Mat4::from_rotation_x(FRAC_PI_2));
}

pub(super) fn ferris_cabin(b: &mut Builder) {
    b.block(v(-0.6, -0.9, -0.5), v(0.6, 0.2, 0.5), PAINTED);
    b.block(v(-0.55, -0.4, -0.51), v(0.55, 0.1, 0.51), COOL);
    b.lathe(
        &[(0.75, 0.2), (0.0, 0.6)],
        4,
        FRAC_PI_4,
        CHROME,
        CHROME,
        true,
    );
}

pub(super) fn ferris_leg(b: &mut Builder) {
    b.block(v(-0.25, 0.0, -0.25), v(0.25, 8.0, 0.25), STEEL);
}

pub(super) fn carousel_base(b: &mut Builder) {
    b.lathe(&[(3.2, 0.0), (3.2, 0.4)], 16, 0.0, TIMBER, TIMBER, false);
    b.bar(v(0.0, 0.4, 0.0), v(0.0, 3.2, 0.0), 0.2, 8, CHROME);
}

pub(super) fn carousel_top(b: &mut Builder) {
    // Canopy, poles and horses, turning together.
    b.lathe(
        &[(3.3, 2.8), (3.3, 3.1), (0.0, 4.2)],
        16,
        0.0,
        Map::Rect(0.0, 0.0, 8.0, 1.0),
        CHROME,
        true,
    );
    for index in 0..8 {
        let (sin, cos) = math::sin_cos(TAU * index as f32 / 8.0);
        let at = v(cos * 2.4, 0.4, sin * 2.4);
        b.bar(at, at + Vec3::Y * 2.4, 0.04, 4, CHROME);
        let start = b.mark();
        b.block(v(-0.45, 0.9, -0.15), v(0.45, 1.3, 0.15), PAINTED);
        b.block(v(0.3, 1.2, -0.1), v(0.55, 1.65, 0.1), PAINTED);
        b.place(
            start,
            Mat4::from_translation(at - Vec3::Y * 0.4)
                * Mat4::from_rotation_y(-TAU * index as f32 / 8.0 + FRAC_PI_2),
        );
    }
}

pub(super) fn turbine_tower(b: &mut Builder) {
    b.lathe(&[(0.7, 0.0), (0.35, 24.0)], 10, 0.0, STEEL, STEEL, false);
    b.block(v(-0.6, 24.0, -0.5), v(1.8, 25.0, 0.5), STEEL);
}

pub(super) fn turbine_rotor(b: &mut Builder) {
    // A nose along +X and three blades in the YZ plane.
    let start = b.mark();
    b.lathe(
        &[(0.0, -0.6), (0.4, -0.3), (0.4, 0.3)],
        8,
        0.0,
        STEEL,
        STEEL,
        false,
    );
    b.place(start, Mat4::from_rotation_z(FRAC_PI_2));
    for blade in 0..3 {
        let start = b.mark();
        b.block(v(0.05, 0.0, -0.25), v(0.18, 10.0, 0.25), STEEL);
        b.place(start, Mat4::from_rotation_x(TAU * blade as f32 / 3.0));
    }
}

pub(super) fn beacon(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 1.6, 0.0), 0.08, 6, STEEL);
    b.block(v(-1.2, 1.5, -0.12), v(1.2, 1.7, 0.12), WARM);
    b.block(v(-0.12, 1.5, -1.2), v(0.12, 1.7, 1.2), WARM);
    b.lathe(
        &[(0.0, 1.6), (0.35, 1.9), (0.0, 2.3)],
        8,
        0.0,
        RED,
        RED,
        false,
    );
}

/// Wheels at `xs` along each side `z` from the center, of `radius`.
fn wheels(b: &mut Builder, xs: &[f32], z: f32, radius: f32) {
    for &x in xs {
        for side in [-z, z] {
            let start = b.mark();
            b.lathe(&[(radius, -0.1), (radius, 0.1)], 8, 0.0, TIRE, TIRE, true);
            b.place(
                start,
                Mat4::from_translation(v(x, radius, side)) * Mat4::from_rotation_x(FRAC_PI_2),
            );
        }
    }
}

/// Headlights at the front `x` and taillights at the back, at `y`, `z` apart.
fn lights(b: &mut Builder, x: f32, y: f32, z: f32) {
    for side in [-z, z] {
        b.block(
            v(x - 0.02, y - 0.08, side - 0.12),
            v(x + 0.02, y + 0.08, side + 0.12),
            HEADLIGHT,
        );
        b.block(
            v(-x - 0.02, y - 0.08, side - 0.12),
            v(-x + 0.02, y + 0.08, side + 0.12),
            TAILLIGHT,
        );
    }
}

pub(super) fn car(b: &mut Builder, kind: Car) {
    // Body length, width, body height, cabin extent and height.
    let (length, width, body, cabin, roof) = match kind {
        Car::Sedan | Car::Taxi | Car::Police => (2.1, 0.95, 0.55, (-0.55, 0.45), 0.95),
        Car::Hatch => (1.8, 0.9, 0.55, (-0.75, 0.35), 0.98),
        Car::Suv => (2.2, 1.0, 0.7, (-0.9, 0.5), 1.2),
        Car::Van => (2.4, 1.0, 0.75, (-1.15, 0.8), 1.3),
        Car::Pickup => (2.4, 1.0, 0.65, (0.0, 0.75), 1.15),
        Car::Sports => (2.1, 0.95, 0.45, (-0.35, 0.35), 0.78),
    };
    let (half, side) = (length * 0.5, width * 0.5);
    b.block(v(-half, 0.22, -side), v(half, body, side), PAINT);
    // The glass stops below the roof panel's top and the grille behind the
    // headlights' fronts: faces sharing a plane fight in the depth buffer.
    b.block(
        v(cabin.0, body, -side * 0.88),
        v(cabin.1, roof - 0.02, side * 0.88),
        WINDOW,
    );
    b.block(
        v(cabin.0 + 0.08, roof - 0.04, -side * 0.84),
        v(cabin.1 - 0.06, roof, side * 0.84),
        PAINT,
    );
    b.block(
        v(half - 0.02, 0.25, -side * 0.6),
        v(half + 0.02, 0.4, side * 0.6),
        GRILLE,
    );
    lights(b, half + 0.01, body - 0.12, side * 0.65);
    wheels(b, &[-half * 0.65, half * 0.65], side * 0.95, 0.22);
    match kind {
        Car::Taxi => b.block(v(-0.15, roof, -0.12), v(0.15, roof + 0.12, 0.12), TAXI_SIGN),
        Car::Police => {
            b.block(v(-0.1, roof, -0.3), v(0.1, roof + 0.08, 0.0), SIREN_BLUE);
            b.block(v(-0.1, roof, 0.0), v(0.1, roof + 0.08, 0.3), SIREN_RED);
        }
        Car::Sports => b.block(
            v(-half, body, -side * 0.9),
            v(-half + 0.25, body + 0.12, side * 0.9),
            TRIM,
        ),
        Car::Pickup => b.block(v(-half, body, -side), v(-0.05, body + 0.2, side), TRIM),
        _ => {}
    }
}

/// A hull `length` long and `width` wide with a pointed bow toward +X, its
/// deck `height` up.
fn hull(b: &mut Builder, length: f32, width: f32, height: f32) {
    let (half, side) = (length * 0.5, width * 0.5);
    let bow = half * 0.6;
    b.cuboid(
        v(-half, 0.0, -side),
        v(bow, height, side),
        PAINT,
        TRIM,
        false,
    );
    let low = [v(bow, 0.0, -side), v(bow, 0.0, side), v(half, 0.0, 0.0)];
    let top = [
        v(bow, height, -side),
        v(bow, height, side),
        v(half, height, 0.0),
    ];
    b.poly(&top, TRIM, Vec3::Y);
    b.quad([low[1], low[2], top[2], top[1]], PAINT, v(1.0, 0.0, 1.0));
    b.quad([low[2], low[0], top[0], top[2]], PAINT, v(1.0, 0.0, -1.0));
    b.block(
        v(-half, height * 0.1, -side - 0.02),
        v(bow, height * 0.25, side + 0.02),
        SIREN_RED,
    );
}

/// A square lattice column `width` wide and `height` tall.
fn lattice_column(b: &mut Builder, width: f32, height: f32) {
    let half = width * 0.5;
    let map = Map::Rect(0.0, 0.0, 1.0, height / width);
    for (a, c, outward) in [
        (v(-half, 0.0, half), v(half, 0.0, half), Vec3::Z),
        (v(half, 0.0, -half), v(-half, 0.0, -half), Vec3::NEG_Z),
        (v(half, 0.0, half), v(half, 0.0, -half), Vec3::X),
        (v(-half, 0.0, -half), v(-half, 0.0, half), Vec3::NEG_X),
    ] {
        b.quad(
            [a, c, c + Vec3::Y * height, a + Vec3::Y * height],
            map,
            outward,
        );
    }
}
