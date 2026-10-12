//! The metro's building details: rooftop machinery, signs, balconies,
//! fire escapes and doors.

use super::*;

pub(super) fn ac_unit(b: &mut Builder) {
    b.cuboid(v(-0.6, 0.0, -0.45), v(0.6, 0.8, 0.45), STEEL, STEEL, false);
    let start = b.mark();
    b.lathe(&[(0.3, 0.8), (0.3, 0.85)], 10, 0.0, DARK, DARK, false);
    b.place(start, Mat4::from_translation(v(0.2, 0.0, 0.0)));
}

pub(super) fn water_tank(b: &mut Builder) {
    let wood = Map::Tile(2.0, 2.0);
    for (x, z) in [(0.6, 0.6), (-0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)] {
        b.cuboid(
            v(x - 0.08, 0.0, z - 0.08),
            v(x + 0.08, 1.4, z + 0.08),
            wood,
            wood,
            false,
        );
    }
    b.lathe(
        &[(0.95, 1.4), (0.95, 3.0), (1.05, 3.0), (0.0, 3.8)],
        10,
        0.0,
        wood,
        wood,
        true,
    );
}

pub(super) fn antenna(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 9.0, 0.0), 0.12, 6, STEEL);
    for (height, width) in [(3.0, 1.4), (5.5, 1.0), (7.5, 0.6)] {
        b.block(
            v(-width, height, -0.04),
            v(width, height + 0.08, 0.04),
            STEEL,
        );
    }
    b.lathe(
        &[(0.0, 9.0), (0.18, 9.2), (0.0, 9.4)],
        6,
        0.0,
        RED,
        RED,
        false,
    );
}

pub(super) fn dish_post(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 1.2, 0.0), 0.08, 6, STEEL);
    b.block(v(-0.3, 0.0, -0.3), v(0.3, 0.12, 0.3), DARK);
}

pub(super) fn dish_head(b: &mut Builder) {
    // A dish facing +X around its pivot.
    let start = b.mark();
    b.lathe(
        &[(0.0, -0.25), (0.45, -0.18), (0.8, 0.0), (0.82, 0.03)],
        12,
        0.0,
        CHROME,
        CHROME,
        false,
    );
    b.bar(v(0.0, -0.2, 0.0), v(0.0, 0.5, 0.0), 0.03, 4, STEEL);
    b.place(start, Mat4::from_rotation_z(-FRAC_PI_2));
}

pub(super) fn solar(b: &mut Builder) {
    let start = b.mark();
    b.cuboid(
        v(-1.5, -0.04, -0.8),
        v(1.5, 0.04, 0.8),
        Map::Rect(0.0, 0.0, 1.0, 0.03),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        true,
    );
    b.place(
        start,
        Mat4::from_translation(v(0.0, 0.9, 0.0)) * Mat4::from_rotation_x(-0.45),
    );
    for x in [-1.3, 1.3] {
        b.cuboid(
            v(x - 0.05, 0.0, -0.4),
            v(x + 0.05, 0.75, -0.3),
            Map::Rect(0.0, 0.0, 0.02, 0.02),
            Map::Rect(0.0, 0.0, 0.02, 0.02),
            false,
        );
        b.cuboid(
            v(x - 0.05, 0.0, 0.3),
            v(x + 0.05, 1.1, 0.4),
            Map::Rect(0.0, 0.0, 0.02, 0.02),
            Map::Rect(0.0, 0.0, 0.02, 0.02),
            false,
        );
    }
}

pub(super) fn railing(b: &mut Builder, size: u8) {
    // A see-through fence around a roof: four upright sheets.
    let half = [TOWER_WIDTH, MID_WIDTH, HOUSE_WIDTH][size as usize] * 0.5 - 0.15;
    let height = 1.0;
    for (a, b2, outward) in [
        (v(-half, 0.0, half), v(half, 0.0, half), Vec3::Z),
        (v(half, 0.0, -half), v(-half, 0.0, -half), Vec3::NEG_Z),
        (v(half, 0.0, half), v(half, 0.0, -half), Vec3::X),
        (v(-half, 0.0, -half), v(-half, 0.0, half), Vec3::NEG_X),
    ] {
        let repeats = (a - b2).length() / 2.0;
        b.quad(
            [a, b2, b2 + Vec3::Y * height, a + Vec3::Y * height],
            Map::Rect(0.0, 0.0, repeats, 0.5),
            outward,
        );
    }
}

pub(super) fn billboard(b: &mut Builder) {
    // Posts, then the face along X showing the whole picture.
    let frame = Map::patch(0.002, 0.002);
    for x in [-1.3, 1.3] {
        b.cuboid(
            v(x - 0.1, 0.0, -0.1),
            v(x + 0.1, 2.0, 0.1),
            frame,
            frame,
            false,
        );
    }
    b.cuboid(v(-2.2, 2.0, -0.12), v(2.2, 4.2, 0.12), frame, frame, true);
    b.quad(
        [
            v(-2.1, 2.1, 0.121),
            v(2.1, 2.1, 0.121),
            v(2.1, 4.1, 0.121),
            v(-2.1, 4.1, 0.121),
        ],
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        Vec3::Z,
    );
    b.quad(
        [
            v(2.1, 2.1, -0.121),
            v(-2.1, 2.1, -0.121),
            v(-2.1, 4.1, -0.121),
            v(2.1, 4.1, -0.121),
        ],
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        Vec3::NEG_Z,
    );
}

pub(super) fn trivision(b: &mut Builder) {
    // A three-sided prism along X turning about its axis.
    for face in 0..3 {
        let angle = |step: f32| TAU * (face as f32 + step) / 3.0;
        let at = |step: f32, x: f32| {
            let (sin, cos) = math::sin_cos(angle(step));
            v(x, cos * 0.9, sin * 0.9)
        };
        let (sin, cos) = math::sin_cos(angle(0.5));
        b.quad(
            [at(1.0, -2.0), at(1.0, 2.0), at(0.0, 2.0), at(0.0, -2.0)],
            Map::Rect(0.0, 0.0, 1.0, 1.0),
            v(0.0, cos, sin),
        );
    }
    for x in [-2.0, 2.0] {
        let corners: Vec<Vec3> = (0..3)
            .map(|index| {
                let (sin, cos) = math::sin_cos(TAU * index as f32 / 3.0);
                v(x, cos * 0.9, sin * 0.9)
            })
            .collect();
        b.poly(&corners, Map::patch(0.002, 0.002), v(x, 0.0, 0.0));
    }
}

pub(super) fn neon_blade(b: &mut Builder) {
    let face = Map::Rect(0.02, 0.02, 0.48, 0.98);
    b.cuboid(
        v(-0.12, 0.0, -0.7),
        v(0.12, 3.2, 0.7),
        face,
        Map::patch(0.01, 0.01),
        true,
    );
}

pub(super) fn awning(b: &mut Builder) {
    // A sloping canvas over a shop front, open below.
    let canvas = Map::Rect(0.0, 0.0, 3.0, 1.0);
    b.quad(
        [
            v(-1.6, 0.0, 1.0),
            v(1.6, 0.0, 1.0),
            v(1.6, 0.6, 0.0),
            v(-1.6, 0.6, 0.0),
        ],
        canvas,
        v(0.0, 1.0, 0.6),
    );
    b.quad(
        [
            v(-1.6, -0.25, 1.0),
            v(1.6, -0.25, 1.0),
            v(1.6, 0.0, 1.0),
            v(-1.6, 0.0, 1.0),
        ],
        canvas,
        Vec3::Z,
    );
    for x in [-1.6, 1.6] {
        b.poly(
            &[v(x, 0.0, 0.0), v(x, 0.0, 1.0), v(x, 0.6, 0.0)],
            canvas,
            v(x, 0.0, 0.0),
        );
    }
}

pub(super) fn balcony(b: &mut Builder) {
    let concrete = Map::Tile(2.0, 2.0);
    b.block(v(-1.2, 0.0, 0.0), v(1.2, 0.15, 1.0), concrete);
}

pub(super) fn balcony_rail(b: &mut Builder) {
    let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.quad(
        [
            v(-1.2, 0.0, 1.0),
            v(1.2, 0.0, 1.0),
            v(1.2, 0.8, 1.0),
            v(-1.2, 0.8, 1.0),
        ],
        glass,
        Vec3::Z,
    );
    b.quad(
        [
            v(1.2, 0.0, 1.0),
            v(1.2, 0.0, 0.0),
            v(1.2, 0.8, 0.0),
            v(1.2, 0.8, 1.0),
        ],
        glass,
        Vec3::X,
    );
    b.quad(
        [
            v(-1.2, 0.0, 0.0),
            v(-1.2, 0.0, 1.0),
            v(-1.2, 0.8, 1.0),
            v(-1.2, 0.8, 0.0),
        ],
        glass,
        Vec3::NEG_X,
    );
}

pub(super) fn fire_escape(b: &mut Builder) {
    // Landings every two floors joined by stair flights, all
    // see-through lattice.
    let lattice = Map::Rect(0.0, 0.0, 2.0, 1.0);
    for level in 0..3 {
        let y = level as f32 * FLOOR * 2.0 + FLOOR;
        b.quad(
            [
                v(-1.5, y, 0.9),
                v(1.5, y, 0.9),
                v(1.5, y, 0.0),
                v(-1.5, y, 0.0),
            ],
            lattice,
            Vec3::Y,
        );
        b.quad(
            [
                v(-1.5, y, 0.9),
                v(1.5, y, 0.9),
                v(1.5, y + 0.8, 0.9),
                v(-1.5, y + 0.8, 0.9),
            ],
            lattice,
            Vec3::Z,
        );
        if level < 2 {
            let (y0, y1) = (y, y + FLOOR * 2.0);
            b.quad(
                [
                    v(-1.3, y0, 0.45),
                    v(1.3, y1, 0.45),
                    v(1.3, y1, 0.0),
                    v(-1.3, y0, 0.0),
                ],
                lattice,
                Vec3::Y,
            );
        }
    }
}

pub(super) fn gondola(b: &mut Builder) {
    b.block(v(-1.2, 0.0, 0.0), v(1.2, 0.12, 0.7), HAZARD);
    b.block(v(-1.2, 0.12, 0.6), v(1.2, 0.6, 0.7), STEEL);
    for x in [-1.1, 1.1] {
        b.bar(v(x, 0.6, 0.35), v(x, 2.5, 0.2), 0.02, 4, STEEL);
    }
}

pub(super) fn flagpole(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 7.0, 0.0), 0.07, 6, CHROME);
    b.lathe(
        &[(0.0, 7.0), (0.14, 7.1), (0.0, 7.25)],
        6,
        0.0,
        CHROME,
        CHROME,
        false,
    );
}

pub(super) fn flag(b: &mut Builder) {
    // Hangs from its pole edge at the origin toward +X.
    let cloth = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.quad(
        [
            v(0.0, -0.5, 0.0),
            v(1.6, -0.5, 0.0),
            v(1.6, 0.5, 0.0),
            v(0.0, 0.5, 0.0),
        ],
        cloth,
        Vec3::Z,
    );
}

pub(super) fn glass_box(b: &mut Builder, kind: u8) {
    let (half, height) = match kind {
        0 => (v(2.4, 0.0, 2.4), 3.2),
        1 => (v(4.0, 0.0, 2.0), 2.4),
        _ => (v(1.2, 0.0, 5.0), 1.6),
    };
    let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.cuboid(
        v(-half.x, 0.0, -half.z),
        v(half.x, height, half.z),
        glass,
        glass,
        false,
    );
}

pub(super) fn door_canopy(b: &mut Builder) {
    b.block(v(-1.2, 1.4, 0.0), v(1.2, 1.55, 1.3), CHROME);
    b.lathe(&[(0.75, 0.0), (0.75, 0.05)], 12, 0.0, DARK, DARK, false);
}

pub(super) fn door_wings(b: &mut Builder) {
    // Four glass wings about a post, turning about Y.
    let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
    for wing in 0..4 {
        let start = b.mark();
        b.quad(
            [
                v(0.0, 0.0, 0.0),
                v(0.7, 0.0, 0.0),
                v(0.7, 1.35, 0.0),
                v(0.0, 1.35, 0.0),
            ],
            glass,
            Vec3::Z,
        );
        b.place(start, Mat4::from_rotation_y(FRAC_PI_2 * wing as f32));
    }
}
