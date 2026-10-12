//! The metro's infrastructure and landmarks.

use super::*;

pub(super) fn rail_deck(b: &mut Builder) {
    b.cuboid(
        v(-12.0, 0.0, -1.6),
        v(12.0, 0.9, 1.6),
        Map::patch(0.2, 0.05),
        Map::Rect(0.0, 0.0, 3.0, 1.0),
        true,
    );
}

pub(super) fn rail_pillar(b: &mut Builder) {
    let concrete = Map::Tile(4.0, 4.0);
    b.cuboid(
        v(-0.6, 0.0, -0.6),
        v(0.6, 8.8, 0.6),
        concrete,
        concrete,
        false,
    );
    b.block(v(-0.8, 8.8, -1.8), v(0.8, 9.5, 1.8), concrete);
}

pub(super) fn station_frame(b: &mut Builder) {
    for x in [-9.0, -3.0, 3.0, 9.0] {
        for z in [-2.6, 2.6] {
            b.bar(v(x, 0.0, z), v(x, 3.6, z * 0.9), 0.1, 6, CHROME);
        }
        b.bar(v(x, 3.6, -2.4), v(x, 3.6, 2.4), 0.08, 6, CHROME);
    }
    b.block(v(-10.0, 0.0, -2.8), v(10.0, 0.3, 2.8), STONE);
}

pub(super) fn station_roof(b: &mut Builder) {
    // A shallow glass vault along X.
    let steps = 6;
    let arc = |index: u32| {
        let angle = PI * (0.15 + 0.7 * index as f32 / steps as f32);
        let (sin, cos) = math::sin_cos(angle);
        (2.9 * cos, 1.0 * sin)
    };
    for index in 0..steps {
        let ((z0, y0), (z1, y1)) = (arc(index), arc(index + 1));
        b.quad(
            [
                v(-10.0, y0, z0),
                v(10.0, y0, z0),
                v(10.0, y1, z1),
                v(-10.0, y1, z1),
            ],
            Map::Rect(0.0, 0.0, 1.0, 1.0),
            v(0.0, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
        );
    }
}

pub(super) fn bridge_deck(b: &mut Builder) {
    b.cuboid(
        v(-10.0, -0.6, -2.25),
        v(10.0, 0.15, 2.25),
        Map::patch(0.2, 0.3),
        Map::Rect(0.0, 0.0, 2.5, 1.0),
        true,
    );
}

pub(super) fn bridge_arch(b: &mut Builder) {
    // A chrome arch over the deck with hangers.
    let steps = 16;
    let point = |index: u32| {
        let t = index as f32 / steps as f32;
        let (sin, cos) = math::sin_cos(PI * t);
        v(-10.0 * cos, 7.0 * sin, 0.0)
    };
    for side in [-2.3, 2.3] {
        for index in 0..steps {
            b.bar(
                point(index) + Vec3::Z * side,
                point(index + 1) + Vec3::Z * side,
                0.22,
                6,
                Map::Tile(2.0, 2.0),
            );
        }
        for index in 1..steps {
            let top = point(index) + Vec3::Z * side;
            b.bar(v(top.x, 0.0, side), top, 0.03, 4, Map::Tile(2.0, 2.0));
        }
    }
    for index in [4, 8, 12] {
        b.bar(
            point(index) + Vec3::Z * -2.3,
            point(index) + Vec3::Z * 2.3,
            0.12,
            6,
            Map::Tile(2.0, 2.0),
        );
    }
}

pub(super) fn bridge_leaf(b: &mut Builder) {
    // Hinged at the origin, reaching along +X.
    b.cuboid(
        v(0.0, -0.5, -2.25),
        v(10.0, 0.15, 2.25),
        Map::patch(0.2, 0.3),
        Map::Rect(0.0, 0.0, 1.25, 1.0),
        true,
    );
}

pub(super) fn dock(b: &mut Builder) {
    let wood = Map::Tile(2.0, 2.0);
    b.block(v(-6.0, 0.0, -1.5), v(6.0, 0.25, 1.5), wood);
    for x in [-5.5, -2.0, 2.0, 5.5] {
        for z in [-1.3, 1.3] {
            b.bar(v(x, -1.6, z), v(x, 0.6, z), 0.15, 6, wood);
        }
    }
}

pub(super) fn stadium(b: &mut Builder) {
    // An outer wall and seating stepping down to the pitch.
    let sides = 20;
    let seats = |rows: f32| Map::Rect(0.0, 0.0, 6.0, rows);
    for index in 0..sides {
        let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
        let at = |radius: f32, y: f32, step: f32| {
            let (sin, cos) = math::sin_cos(angle(step));
            v(radius * cos * 1.2, y, radius * sin)
        };
        let (sin, cos) = math::sin_cos(angle(0.5));
        let radial = v(cos, 0.0, sin);
        b.quad(
            [
                at(9.5, 0.0, 1.0),
                at(9.5, 0.0, 0.0),
                at(9.5, 7.0, 0.0),
                at(9.5, 7.0, 1.0),
            ],
            Map::Rect(0.0, 0.96, 6.0, 1.0),
            radial,
        );
        b.quad(
            [
                at(9.0, 7.0, 1.0),
                at(9.0, 7.0, 0.0),
                at(9.5, 7.0, 0.0),
                at(9.5, 7.0, 1.0),
            ],
            Map::Rect(0.0, 0.96, 6.0, 1.0),
            Vec3::Y,
        );
        b.quad(
            [
                at(6.0, 0.8, 0.0),
                at(6.0, 0.8, 1.0),
                at(9.0, 7.0, 1.0),
                at(9.0, 7.0, 0.0),
            ],
            seats(3.0),
            Vec3::Y - radial,
        );
    }
}

pub(super) fn stadium_roof(b: &mut Builder) {
    let sides = 20;
    for index in 0..sides {
        let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
        let at = |radius: f32, y: f32, step: f32| {
            let (sin, cos) = math::sin_cos(angle(step));
            v(radius * cos * 1.2, y, radius * sin)
        };
        b.quad(
            [
                at(7.2, 0.6, 0.0),
                at(7.2, 0.6, 1.0),
                at(9.7, 0.0, 1.0),
                at(9.7, 0.0, 0.0),
            ],
            Map::Tile(2.0, 2.0),
            Vec3::Y,
        );
        b.quad(
            [
                at(7.2, 0.4, 1.0),
                at(7.2, 0.4, 0.0),
                at(9.7, -0.2, 0.0),
                at(9.7, -0.2, 1.0),
            ],
            Map::Tile(2.0, 2.0),
            Vec3::NEG_Y,
        );
    }
}

pub(super) fn pitch(b: &mut Builder) {
    b.cuboid(
        v(-7.2, 0.0, -4.8),
        v(7.2, 0.1, 4.8),
        Map::patch(0.1, 0.5),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        false,
    );
}

pub(super) fn floodlight(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 12.0, 0.0), 0.15, 6, STEEL);
    b.block(v(-1.0, 11.6, -0.2), v(1.0, 12.8, 0.2), STEEL);
    b.block(v(-0.9, 11.7, -0.25), v(0.9, 12.7, -0.2), COOL);
}

pub(super) fn tv_tower(b: &mut Builder) {
    b.lathe(
        &[
            (4.0, 0.0),
            (2.2, 6.0),
            (1.4, 40.0),
            (1.2, 80.0),
            (1.0, 118.0),
            (3.0, 122.0),
            (3.0, 126.0),
            (1.0, 128.0),
            (0.7, 150.0),
            (0.3, 175.0),
            (0.0, 190.0),
        ],
        12,
        0.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
}

pub(super) fn obs_pod(b: &mut Builder) {
    let profile: Vec<(f32, f32)> = (0..=8)
        .map(|index| {
            let angle = PI * index as f32 / 8.0;
            let (sin, cos) = math::sin_cos(angle);
            (5.5 * sin, -3.0 * cos)
        })
        .collect();
    b.lathe(
        &profile,
        16,
        0.0,
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        false,
    );
}

pub(super) fn tower_ring(b: &mut Builder) {
    b.lathe(
        &[(6.5, 0.0), (6.5, 1.2), (5.8, 1.6), (4.0, 1.6)],
        20,
        0.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        true,
    );
}
