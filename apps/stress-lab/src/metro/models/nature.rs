//! The metro's trees, bushes and rocks.

use super::*;

pub(super) fn tree(b: &mut Builder, kind: u8) {
    // A trunk and crossed leaf cards around a crown.
    let (height, crown, width) =
        [(4.2, 2.2, 3.4), (5.0, 2.8, 3.0), (3.4, 1.8, 3.8)][kind as usize % 3];
    b.paint([0.55, 0.55, 0.55]);
    b.lathe(
        &[(0.16, 0.0), (0.1, height - crown * 0.4)],
        6,
        0.0,
        TRUNK,
        TRUNK,
        false,
    );
    for layer in 0..3 {
        let t = layer as f32 / 2.0;
        let shade = 0.65 + t * 0.35;
        b.paint([shade, shade, shade]);
        let size = width * (1.0 - t * 0.35);
        b.cards(
            v(0.0, height - crown - 0.2 + t * crown * 0.45, 0.0),
            size,
            crown * (1.0 - t * 0.2),
            t * 0.9 + kind as f32,
            CANOPY,
        );
    }
}

pub(super) fn poplar(b: &mut Builder) {
    b.paint([0.55, 0.55, 0.55]);
    b.lathe(&[(0.12, 0.0), (0.08, 1.5)], 6, 0.0, TRUNK, TRUNK, false);
    for layer in 0..2 {
        let shade = 0.7 + layer as f32 * 0.3;
        b.paint([shade, shade, shade]);
        b.cards(
            v(0.0, 1.0 + layer as f32 * 0.4, 0.0),
            1.6,
            5.2,
            layer as f32 * 0.8,
            CANOPY,
        );
    }
}

pub(super) fn pine(b: &mut Builder) {
    b.paint([0.5, 0.5, 0.5]);
    b.lathe(&[(0.15, 0.0), (0.1, 1.2)], 6, 0.0, TRUNK, TRUNK, false);
    for (index, (radius, low, high)) in [(1.6, 1.0, 3.2), (1.25, 2.3, 4.3), (0.85, 3.4, 5.4)]
        .into_iter()
        .enumerate()
    {
        let shade = 0.6 + index as f32 * 0.2;
        b.paint([shade, shade, shade]);
        b.lathe(
            &[(radius, low), (0.0, high)],
            8,
            index as f32 * 0.4,
            DENSE,
            DENSE,
            true,
        );
    }
}

pub(super) fn palm(b: &mut Builder) {
    b.paint([0.7, 0.7, 0.7]);
    // A leaning trunk of segments, fronds drooping from the top.
    let mut top = Vec3::ZERO;
    for segment in 0..6 {
        let next = top + v(0.12, 1.0, 0.0);
        b.bar(top, next, 0.16 - segment as f32 * 0.012, 6, TRUNK);
        top = next;
    }
    b.paint([1.0, 1.0, 1.0]);
    for frond in 0..7 {
        let yaw = TAU * frond as f32 / 7.0;
        let (sin, cos) = math::sin_cos(yaw);
        let out = v(cos, 0.0, sin);
        let side = v(-sin, 0.0, cos);
        let tip = top + out * 2.4 - Vec3::Y * 0.9;
        let mid = top + out * 1.2 + Vec3::Y * 0.2;
        // The frond's base half toward the trunk, its tip half beyond.
        let (base, end) = (
            Map::Rect(0.0, 0.39, 0.5, 0.78),
            Map::Rect(0.0, 0.0, 0.5, 0.39),
        );
        b.quad(
            [
                top - side * 0.35,
                top + side * 0.35,
                mid + side * 0.5,
                mid - side * 0.5,
            ],
            base,
            Vec3::Y,
        );
        b.quad(
            [
                mid - side * 0.5,
                mid + side * 0.5,
                tip + side * 0.15,
                tip - side * 0.15,
            ],
            end,
            Vec3::Y,
        );
    }
}

pub(super) fn bush(b: &mut Builder) {
    for layer in 0..2 {
        let shade = 0.7 + layer as f32 * 0.3;
        b.paint([shade, shade, shade]);
        b.cards(
            v(0.0, layer as f32 * 0.3, 0.0),
            1.8 - layer as f32 * 0.4,
            1.2,
            layer as f32,
            CANOPY,
        );
    }
}

pub(super) fn rock(b: &mut Builder) {
    b.lathe(
        &[(0.0, 0.0), (1.0, 0.1), (0.8, 0.7), (0.0, 1.0)],
        6,
        0.3,
        Map::Tile(2.0, 2.0),
        Map::Tile(2.0, 2.0),
        false,
    );
}
