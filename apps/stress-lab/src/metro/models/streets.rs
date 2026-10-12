//! The metro's streets and their furniture.

use super::*;

pub(super) fn road(b: &mut Builder) {
    b.cuboid(
        v(-10.0, 0.0, -2.0),
        v(10.0, 0.15, 2.0),
        Map::patch(0.2, 0.3),
        Map::Rect(0.0, 0.0, 2.5, 1.0),
        false,
    );
}

pub(super) fn road_cross(b: &mut Builder) {
    b.cuboid(
        v(-2.0, 0.0, -2.0),
        v(2.0, 0.15, 2.0),
        Map::patch(0.5, 0.5),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        false,
    );
}

pub(super) fn sidewalk(b: &mut Builder) {
    // A raised ring 1.6 wide around the block.
    let pavers = Map::Tile(2.0, 2.0);
    let (outer, inner, height) = (10.0, 8.4, 0.3);
    for (min, max) in [
        (v(-outer, 0.0, -outer), v(outer, height, -inner)),
        (v(-outer, 0.0, inner), v(outer, height, outer)),
        (v(-outer, 0.0, -inner), v(-inner, height, inner)),
        (v(inner, 0.0, -inner), v(outer, height, inner)),
    ] {
        b.cuboid(min, max, pavers, pavers, false);
    }
}

pub(super) fn ground(b: &mut Builder) {
    let tile = Map::Tile(4.0, 4.0);
    b.cuboid(v(-8.4, 0.0, -8.4), v(8.4, 0.2, 8.4), tile, tile, false);
}

pub(super) fn lamp(b: &mut Builder) {
    b.block(v(-0.2, 0.0, -0.2), v(0.2, 0.3, 0.2), DARK);
    b.bar(v(0.0, 0.3, 0.0), v(0.0, 3.6, 0.0), 0.07, 6, CHROME);
    b.bar(v(0.0, 3.5, 0.0), v(0.9, 3.7, 0.0), 0.05, 6, CHROME);
    b.block(v(0.6, 3.55, -0.2), v(1.2, 3.72, 0.2), CHROME);
    b.quad(
        [
            v(0.62, 3.549, 0.18),
            v(1.18, 3.549, 0.18),
            v(1.18, 3.549, -0.18),
            v(0.62, 3.549, -0.18),
        ],
        WARM,
        Vec3::NEG_Y,
    );
}

pub(super) fn traffic_light(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 3.4, 0.0), 0.08, 6, DARK);
    b.bar(v(0.0, 3.2, 0.0), v(2.6, 3.2, 0.0), 0.06, 6, DARK);
    b.block(v(2.3, 2.3, -0.18), v(2.7, 3.3, 0.18), DARK);
    for (index, lens) in [RED, AMBER, GREEN].into_iter().enumerate() {
        let y = 3.1 - index as f32 * 0.3;
        b.block(v(2.38, y - 0.1, 0.18), v(2.62, y + 0.1, 0.22), lens);
        b.block(v(2.38, y - 0.1, -0.22), v(2.62, y + 0.1, -0.18), lens);
    }
}

pub(super) fn shelter_frame(b: &mut Builder) {
    for (x, z) in [(-1.8, -0.6), (1.8, -0.6), (-1.8, 0.6), (1.8, 0.6)] {
        b.bar(v(x, 0.0, z), v(x, 2.2, z), 0.05, 6, CHROME);
    }
    b.block(v(-2.0, 2.2, -0.8), v(2.0, 2.32, 0.8), CHROME);
    b.block(v(-1.5, 0.35, -0.5), v(1.5, 0.45, -0.2), TIMBER);
    // An ad panel at one end, lit from inside.
    b.block(v(1.75, 0.3, -0.55), v(1.85, 2.1, 0.55), COOL);
}

pub(super) fn shelter_glass(b: &mut Builder) {
    let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.quad(
        [
            v(1.8, 0.2, -0.6),
            v(-1.8, 0.2, -0.6),
            v(-1.8, 2.1, -0.6),
            v(1.8, 2.1, -0.6),
        ],
        glass,
        Vec3::NEG_Z,
    );
    b.quad(
        [
            v(-1.8, 0.2, -0.6),
            v(-1.8, 0.2, 0.6),
            v(-1.8, 2.1, 0.6),
            v(-1.8, 2.1, -0.6),
        ],
        glass,
        Vec3::NEG_X,
    );
}

pub(super) fn bench(b: &mut Builder) {
    for index in 0..3 {
        let z = -0.2 + index as f32 * 0.14;
        b.block(v(-0.9, 0.42, z), v(0.9, 0.48, z + 0.1), TIMBER);
    }
    b.block(v(-0.9, 0.55, 0.22), v(0.9, 0.9, 0.28), TIMBER);
    for x in [-0.75, 0.75] {
        b.block(v(x - 0.04, 0.0, -0.22), v(x + 0.04, 0.42, 0.28), DARK);
    }
}

pub(super) fn bin(b: &mut Builder) {
    b.lathe(
        &[(0.25, 0.0), (0.3, 0.8), (0.32, 0.85), (0.2, 0.9)],
        8,
        0.0,
        DARK,
        STEEL,
        false,
    );
}

pub(super) fn hydrant(b: &mut Builder) {
    b.lathe(
        &[
            (0.18, 0.0),
            (0.14, 0.1),
            (0.14, 0.45),
            (0.18, 0.5),
            (0.1, 0.6),
            (0.0, 0.66),
        ],
        8,
        0.0,
        PAINTED,
        PAINTED,
        false,
    );
    b.block(v(-0.25, 0.3, -0.05), v(0.25, 0.4, 0.05), PAINTED);
}

pub(super) fn bollard(b: &mut Builder) {
    b.lathe(
        &[(0.14, 0.0), (0.14, 0.8), (0.0, 0.9)],
        8,
        0.0,
        CHROME,
        CHROME,
        false,
    );
    b.lathe(&[(0.145, 0.62), (0.145, 0.7)], 8, 0.0, WARM, WARM, false);
}

pub(super) fn planter(b: &mut Builder) {
    b.cuboid(
        v(-0.9, 0.0, -0.9),
        v(0.9, 0.7, 0.9),
        STONE,
        Map::patch(0.62, 0.62),
        false,
    );
}

pub(super) fn flowers(b: &mut Builder) {
    // Clumps of blossoms in vertex colors over a matte material.
    let colors = [
        [0.95, 0.2, 0.35],
        [1.0, 0.85, 0.2],
        [0.7, 0.35, 0.95],
        [1.0, 0.55, 0.2],
        [0.95, 0.95, 0.95],
    ];
    b.paint([0.15, 0.35, 0.1]);
    b.block(
        v(-0.8, 0.0, -0.8),
        v(0.8, 0.12, 0.8),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
    );
    for index in 0..14 {
        let random = crate::scene::hash(index * 977 + 13);
        let (x, z) = (
            (random % 150) as f32 / 100.0 - 0.75,
            ((random >> 8) % 150) as f32 / 100.0 - 0.75,
        );
        b.paint(colors[(random >> 16) as usize % colors.len()]);
        let size = 0.08 + ((random >> 20) % 5) as f32 * 0.015;
        b.block(
            v(x - size, 0.1, z - size),
            v(x + size, 0.2 + size, z + size),
            Map::Rect(0.0, 0.0, 1.0, 1.0),
        );
    }
}

pub(super) fn kiosk_body(b: &mut Builder) {
    b.block(v(-1.0, 0.0, -0.8), v(1.0, 2.0, 0.8), DARK);
    b.block(v(-0.9, 0.9, 0.8), v(0.9, 1.7, 0.84), WARM);
    b.block(v(-1.05, 0.8, 0.8), v(1.05, 0.9, 1.1), STEEL);
}

pub(super) fn kiosk_roof(b: &mut Builder) {
    b.lathe(
        &[(1.6, 0.0), (0.0, 0.8)],
        4,
        FRAC_PI_4,
        Map::Rect(0.0, 0.0, 3.0, 1.0),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        true,
    );
}

pub(super) fn booth_frame(b: &mut Builder) {
    b.block(v(-0.5, 0.0, -0.5), v(0.5, 0.1, 0.5), PAINTED);
    b.block(v(-0.5, 2.1, -0.5), v(0.5, 2.3, 0.5), PAINTED);
    for (x, z) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
        b.block(
            v(x - 0.05, 0.1, z - 0.05),
            v(x + 0.05, 2.1, z + 0.05),
            PAINTED,
        );
    }
    b.block(v(-0.4, 2.12, 0.5), v(0.4, 2.28, 0.52), COOL);
}

pub(super) fn booth_glass(b: &mut Builder) {
    let glass = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.cuboid(
        v(-0.45, 0.1, -0.45),
        v(0.45, 2.1, 0.45),
        glass,
        glass,
        false,
    );
}

pub(super) fn umbrella(b: &mut Builder) {
    b.bar(
        v(0.0, 0.0, 0.0),
        v(0.0, 2.1, 0.0),
        0.03,
        4,
        Map::Rect(0.0, 0.9, 0.1, 1.0),
    );
    b.lathe(
        &[(1.3, 1.8), (0.0, 2.3)],
        8,
        0.0,
        Map::Rect(0.0, 0.0, 4.0, 1.0),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        true,
    );
}

pub(super) fn cafe_table(b: &mut Builder) {
    b.lathe(&[(0.35, 0.72), (0.35, 0.76)], 10, 0.0, STEEL, STEEL, true);
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 0.72, 0.0), 0.04, 6, STEEL);
    for (x, z) in [(0.6, 0.0), (-0.6, 0.0)] {
        b.block(v(x - 0.2, 0.42, z - 0.2), v(x + 0.2, 0.46, z + 0.2), STEEL);
        b.bar(v(x, 0.0, z), v(x, 0.42, z), 0.03, 4, STEEL);
    }
}

pub(super) fn food_cart(b: &mut Builder) {
    b.block(v(-1.0, 0.35, -0.6), v(1.0, 1.3, 0.6), PAINT);
    b.block(v(-0.9, 1.3, -0.5), v(0.9, 1.35, 0.5), TRIM);
    for (x, z) in [(-0.6, 0.6), (0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)] {
        let start = b.mark();
        b.lathe(&[(0.3, -0.06), (0.3, 0.06)], 10, 0.0, TIRE, TIRE, true);
        b.place(
            start,
            Mat4::from_translation(v(x, 0.3, z)) * Mat4::from_rotation_x(FRAC_PI_2),
        );
    }
}

pub(super) fn plinth(b: &mut Builder) {
    let stone = Map::Tile(2.0, 2.0);
    b.block(v(-1.2, 0.0, -1.2), v(1.2, 0.4, 1.2), stone);
    b.block(v(-0.9, 0.4, -0.9), v(0.9, 2.2, 0.9), stone);
}

pub(super) fn figure(b: &mut Builder) {
    // A blocky standing figure with a raised arm.
    let metal = Map::Tile(1.0, 1.0);
    b.block(v(-0.4, 0.0, -0.25), v(-0.05, 1.3, 0.25), metal);
    b.block(v(0.05, 0.0, -0.25), v(0.4, 1.3, 0.25), metal);
    b.block(v(-0.5, 1.3, -0.3), v(0.5, 2.6, 0.3), metal);
    b.block(v(-0.25, 2.6, -0.25), v(0.25, 3.1, 0.25), metal);
    b.block(v(-0.8, 1.6, -0.15), v(-0.5, 2.5, 0.15), metal);
    b.block(v(0.5, 2.3, -0.15), v(0.8, 3.6, 0.15), metal);
}

pub(super) fn sculpture(b: &mut Builder) {
    // A bean of mirror metal.
    let profile: Vec<(f32, f32)> = (0..=10)
        .map(|index| {
            let angle = PI * index as f32 / 10.0;
            let (sin, cos) = math::sin_cos(angle);
            (1.8 * sin * (1.0 + 0.25 * cos), 1.5 - 1.5 * cos)
        })
        .collect();
    let start = b.mark();
    b.lathe(
        &profile,
        16,
        0.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
    b.place(start, Mat4::from_rotation_z(0.25));
}

pub(super) fn basin(b: &mut Builder) {
    let stone = Map::Tile(2.0, 2.0);
    b.lathe(
        &[
            (3.0, 0.0),
            (3.0, 0.6),
            (2.7, 0.6),
            (2.7, 0.3),
            (0.6, 0.3),
            (0.6, 1.4),
            (1.2, 1.5),
            (1.2, 1.7),
            (0.3, 1.7),
            (0.2, 2.3),
            (0.0, 2.4),
        ],
        16,
        0.0,
        stone,
        stone,
        false,
    );
}

pub(super) fn pool(b: &mut Builder) {
    let water = Map::Tile(8.0, 8.0);
    b.lathe(&[(2.72, 0.0), (2.72, 0.05)], 16, 0.0, water, water, false);
}

pub(super) fn hedge(b: &mut Builder) {
    let leaves = Map::Tile(2.0, 2.0);
    b.cuboid(v(-1.6, 0.0, -0.4), v(1.6, 1.0, 0.4), leaves, leaves, false);
}

pub(super) fn manhole(b: &mut Builder) {
    b.lathe(&[(0.45, 0.0), (0.45, 0.03)], 10, 0.0, DARK, DARK, false);
}

pub(super) fn street_sign(b: &mut Builder) {
    b.bar(v(0.0, 0.0, 0.0), v(0.0, 2.8, 0.0), 0.05, 6, STEEL);
    b.block(v(-0.05, 2.4, -0.6), v(0.05, 2.7, 0.6), SIGN);
    b.block(v(-0.6, 2.05, -0.05), v(0.6, 2.35, 0.05), SIGN);
}
