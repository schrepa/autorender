//! The metro's buildings, their roofs and tops, and its industry.

use super::*;

pub(super) fn house(b: &mut Builder, floors: u8) {
    building_frame(b);
    let height = FLOOR * floors as f32;
    let half = HOUSE_WIDTH * 0.5;
    b.top = Some(height);
    b.cuboid(
        v(-half, 0.0, -half),
        v(half, height, half),
        FACADE,
        WALL,
        false,
    );
}

pub(super) fn townhouse(b: &mut Builder) {
    building_frame(b);
    for (index, floors) in [3.0, 4.0, 3.0].into_iter().enumerate() {
        let height = FLOOR * floors;
        b.top = Some(height);
        let x0 = -3.2 + index as f32 * 2.1334;
        b.cuboid(
            v(x0, 0.0, -1.6),
            v(x0 + 2.1334, height, 1.6),
            FACADE,
            WALL,
            false,
        );
        // A cornice along the front.
        b.cuboid(
            v(x0, height - 0.3, 1.6),
            v(x0 + 2.1334, height - 0.05, 1.8),
            WALL,
            WALL,
            true,
        );
    }
}

pub(super) fn shop(b: &mut Builder) {
    building_frame(b);
    b.top = Some(FLOOR * 2.0);
    b.cuboid(
        v(-2.4, 0.0, -1.6),
        v(2.4, FLOOR * 2.0, 1.6),
        FACADE,
        WALL,
        false,
    );
    // A sign band over the shop window.
    b.cuboid(
        v(-2.4, FLOOR * 0.95, 1.6),
        v(2.4, FLOOR * 1.2, 1.85),
        WALL,
        WALL,
        true,
    );
}

pub(super) fn warehouse(b: &mut Builder) {
    building_frame(b);
    let height = FLOOR * 3.0;
    b.top = Some(height);
    b.cuboid(v(-7.0, 0.0, -3.5), v(7.0, height, 3.5), FACADE, WALL, false);
    // Sawtooth roof lights along X.
    for tooth in 0..4 {
        let z0 = -3.5 + tooth as f32 * 1.75;
        b.ridge(
            -7.0,
            7.0,
            [(z0, height), (z0 + 1.75, height), (z0, height + 1.3)],
            FACADE,
            FACADE,
        );
    }
}

pub(super) fn shed(b: &mut Builder) {
    building_frame(b);
    let height = FLOOR * 2.0;
    b.top = Some(height);
    b.cuboid(v(-5.0, 0.0, -3.0), v(5.0, height, 3.0), FACADE, WALL, false);
    // A barrel vault along X.
    let steps = 8;
    let arc = |index: u32| {
        let angle = PI * index as f32 / steps as f32;
        let (sin, cos) = math::sin_cos(angle);
        (3.0 * cos, height + 1.6 * sin)
    };
    for index in 0..steps {
        let ((z0, y0), (z1, y1)) = (arc(index), arc(index + 1));
        let outward = v(0.0, (y0 + y1) * 0.5 - height, (z0 + z1) * 0.5);
        b.quad(
            [
                v(-5.0, y0, z0),
                v(5.0, y0, z0),
                v(5.0, y1, z1),
                v(-5.0, y1, z1),
            ],
            FACADE,
            outward,
        );
    }
    for x in [-5.0, 5.0] {
        let corners: Vec<Vec3> = (0..=steps)
            .map(|index| {
                let (z, y) = arc(index);
                v(x, y, z)
            })
            .collect();
        b.poly(&corners, FACADE, v(x, 0.0, 0.0));
    }
}

pub(super) fn roof_slab(b: &mut Builder, size: u8) {
    let half = [TOWER_WIDTH, MID_WIDTH, HOUSE_WIDTH][size as usize] * 0.5 - 0.3;
    let roof = Map::Tile(8.0, 8.0);
    b.cuboid(v(-half, 0.0, -half), v(half, 0.35, half), roof, roof, false);
}

pub(super) fn gable(b: &mut Builder) {
    let roof = Map::Tile(4.0, 4.0);
    b.ridge(
        -1.8,
        1.8,
        [(-1.8, 0.0), (1.8, 0.0), (0.0, 1.6)],
        roof,
        Map::Rect(0.0, 0.0, 1.0, 0.5),
    );
}

pub(super) fn hip(b: &mut Builder) {
    b.lathe(
        &[(2.5, 0.0), (0.0, 1.5)],
        4,
        FRAC_PI_4,
        Map::Tile(4.0, 4.0),
        Map::Tile(4.0, 4.0),
        false,
    );
}

pub(super) fn spire(b: &mut Builder) {
    b.lathe(
        &[(1.4, 0.0), (1.0, 2.0), (0.25, 10.0), (0.0, 14.0)],
        8,
        0.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
}

pub(super) fn crown(b: &mut Builder) {
    b.lathe(
        &[
            (3.0, 0.0),
            (3.0, 1.0),
            (2.4, 1.0),
            (2.4, 3.0),
            (1.8, 3.0),
            (1.8, 5.0),
            (1.0, 5.0),
            (0.6, 8.0),
            (0.0, 9.0),
        ],
        8,
        PI / 8.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
}

pub(super) fn dome(b: &mut Builder) {
    let profile: Vec<(f32, f32)> = (0..=6)
        .map(|index| {
            let angle = FRAC_PI_2 * index as f32 / 6.0;
            let (sin, cos) = math::sin_cos(angle);
            (2.5 * cos, 2.0 * sin)
        })
        .collect();
    b.lathe(
        &profile,
        14,
        0.0,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
}

pub(super) fn helipad(b: &mut Builder) {
    b.lathe(
        &[(2.8, 0.0), (2.8, 0.15)],
        16,
        0.0,
        Map::Rect(0.0, 0.0, 1.0, 0.1),
        Map::Rect(0.0, 0.0, 1.0, 1.0),
        false,
    );
}

pub(super) fn tank(b: &mut Builder) {
    b.lathe(
        &[(2.2, 0.0), (2.2, 6.0), (1.6, 6.8), (0.0, 7.2)],
        14,
        0.0,
        Map::Tile(4.0, 4.0),
        Map::Tile(4.0, 4.0),
        false,
    );
    // A ladder up its side.
    b.cuboid(
        v(2.2, 0.0, -0.3),
        v(2.35, 6.0, 0.3),
        Map::Tile(1.0, 1.0),
        Map::Tile(1.0, 1.0),
        false,
    );
}

pub(super) fn silo(b: &mut Builder) {
    b.lathe(
        &[(1.5, 0.0), (1.5, 12.0), (0.9, 13.4), (0.0, 13.8)],
        12,
        0.0,
        Map::Tile(4.0, 4.0),
        Map::Tile(4.0, 4.0),
        false,
    );
}

pub(super) fn stack(b: &mut Builder) {
    b.lathe(
        &[(1.0, 0.0), (0.7, 26.0), (0.85, 26.4), (0.85, 27.0)],
        10,
        0.0,
        Map::Tile(4.0, 4.0),
        Map::Tile(4.0, 4.0),
        false,
    );
}

pub(super) fn container(b: &mut Builder) {
    let side = Map::Rect(0.0, 0.0, 1.0, 1.0);
    b.cuboid(v(-3.05, 0.0, -1.2), v(3.05, 2.6, 1.2), side, side, true);
}

/// Building meshes carry second coordinates spanning them, for grime.
fn building_frame(b: &mut Builder) {
    b.uv1 = Uv1::Span;
}

pub(super) fn tower(b: &mut Builder, style: Style, floors: u32) {
    building_frame(b);
    let height = FLOOR * floors as f32;
    let half = TOWER_WIDTH * 0.5;
    b.top = Some(height);
    // Floors of a fraction of the height, whole floors.
    let at = |fraction: f32| (floors as f32 * fraction).round() * FLOOR;
    match style {
        Style::Box => b.cuboid(
            v(-half, 0.0, -half),
            v(half, height, half),
            FACADE,
            WALL,
            false,
        ),
        Style::Setback => {
            for (from, to, width) in [(0.0, 0.55, 1.0), (0.55, 0.8, 0.75), (0.8, 1.0, 0.5)] {
                let reach = half * width;
                b.cuboid(
                    v(-reach, at(from), -reach),
                    v(reach, at(to), reach),
                    FACADE,
                    WALL,
                    false,
                );
            }
        }
        Style::Round => b.lathe(&[(half, 0.0), (half, height)], 14, 0.0, FACADE, WALL, false),
        Style::Hex => b.lathe(
            &[(half * 1.05, 0.0), (half * 1.05, height)],
            6,
            0.0,
            FACADE,
            WALL,
            false,
        ),
        Style::Tapered => b.lathe(
            &[(half * 1.35, 0.0), (half * 0.85, height)],
            4,
            FRAC_PI_4,
            FACADE,
            WALL,
            false,
        ),
        Style::Deco => b.lathe(
            &[
                (half * 1.1, 0.0),
                (half * 1.1, at(0.72)),
                (half * 0.85, at(0.72)),
                (half * 0.85, at(0.86)),
                (half * 0.6, at(0.86)),
                (half * 0.6, height),
            ],
            8,
            PI / 8.0,
            FACADE,
            WALL,
            false,
        ),
        Style::Twisted => {
            let slabs = floors / 4;
            for slab in 0..slabs {
                let (y0, y1) = (slab as f32 * 4.0 * FLOOR, (slab + 1) as f32 * 4.0 * FLOOR);
                let start = b.mark();
                let reach = half * 0.85;
                b.cuboid(
                    v(-reach, y0, -reach),
                    v(reach, y1, reach),
                    FACADE,
                    WALL,
                    false,
                );
                b.place(
                    start,
                    Mat4::from_rotation_y(
                        FRAC_PI_4 * 1.3 * slab as f32 / math::max(slabs as f32 - 1.0, 1.0),
                    ),
                );
            }
        }
        Style::Fins => {
            let reach = half * 0.9;
            b.cuboid(
                v(-reach, 0.0, -reach),
                v(reach, height, reach),
                FACADE,
                WALL,
                false,
            );
            // Vertical fins on every face, one per bay.
            let count = (TOWER_WIDTH / BAY).floor() as i32;
            for index in 0..=count {
                let along = -reach + index as f32 * (2.0 * reach / count as f32);
                for (min, max) in [
                    (
                        v(along - 0.08, 0.0, reach),
                        v(along + 0.08, height + 0.6, reach + 0.45),
                    ),
                    (
                        v(along - 0.08, 0.0, -reach - 0.45),
                        v(along + 0.08, height + 0.6, -reach),
                    ),
                    (
                        v(reach, 0.0, along - 0.08),
                        v(reach + 0.45, height + 0.6, along + 0.08),
                    ),
                    (
                        v(-reach - 0.45, 0.0, along - 0.08),
                        v(-reach, height + 0.6, along + 0.08),
                    ),
                ] {
                    b.cuboid(min, max, FACADE, WALL, false);
                }
            }
        }
        Style::Slant => {
            // One side rises 8 floors above the other: a sloped roof.
            let low = height - 8.0 * FLOOR;
            let corners = [
                v(-half, 0.0, half),
                v(half, 0.0, half),
                v(half, 0.0, -half),
                v(-half, 0.0, -half),
            ];
            let top_of = |corner: Vec3| if corner.x > 0.0 { height } else { low };
            for index in 0..4 {
                let (a, c) = (corners[index], corners[(index + 1) % 4]);
                let outward = (a + c).normalize_or_zero();
                b.poly(
                    &[a, c, v(c.x, top_of(c), c.z), v(a.x, top_of(a), a.z)],
                    FACADE,
                    outward,
                );
            }
            let roof: Vec<Vec3> = corners
                .iter()
                .rev()
                .map(|corner| v(corner.x, top_of(*corner), corner.z))
                .collect();
            b.poly(&roof, WALL, v(-(height - low), TOWER_WIDTH, 0.0));
        }
    }
}

pub(super) fn midrise(b: &mut Builder, style: Mid, floors: u32) {
    building_frame(b);
    let height = FLOOR * floors as f32;
    let half = MID_WIDTH * 0.5;
    b.top = Some(height);
    match style {
        Mid::Block => b.cuboid(
            v(-half, 0.0, -half),
            v(half, height, half),
            FACADE,
            WALL,
            false,
        ),
        Mid::Ell => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, height, 0.0),
                FACADE,
                WALL,
                false,
            );
            let wing = (floors as f32 * 0.6).round() * FLOOR;
            b.cuboid(v(-half, 0.0, 0.0), v(0.0, wing, half), FACADE, WALL, false);
        }
        Mid::Court => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(-half * 0.4, height, half),
                FACADE,
                WALL,
                false,
            );
            b.cuboid(
                v(half * 0.4, 0.0, -half),
                v(half, height, half),
                FACADE,
                WALL,
                false,
            );
            let low = (floors as f32 * 0.7).round() * FLOOR;
            b.cuboid(
                v(-half * 0.4, 0.0, -half),
                v(half * 0.4, low, -half * 0.4),
                FACADE,
                WALL,
                false,
            );
        }
        Mid::Walkup => {
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, height, half),
                FACADE,
                WALL,
                false,
            );
            // A cornice and a plinth.
            b.cuboid(
                v(-half - 0.2, height - 0.35, -half - 0.2),
                v(half + 0.2, height, half + 0.2),
                WALL,
                WALL,
                true,
            );
            b.cuboid(
                v(-half - 0.1, 0.0, -half - 0.1),
                v(half + 0.1, 0.6, half + 0.1),
                WALL,
                WALL,
                false,
            );
        }
        Mid::Terrace => {
            let low = (floors as f32 * 0.6).round() * FLOOR;
            b.cuboid(
                v(-half, 0.0, -half),
                v(half, low, half),
                FACADE,
                WALL,
                false,
            );
            b.cuboid(
                v(-half, low, -half),
                v(half * 0.3, height, half * 0.4),
                FACADE,
                WALL,
                false,
            );
        }
    }
}
