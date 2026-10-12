//! The city's trees and bushes.

use super::*;

pub(super) fn tree(b: &mut Builder, shape: Shape) {
    use Shape::*;
    if shape != Bush {
        b.lathe(&[(0.1, -0.5), (0.07, -0.1)], 6, 0.0, BARK, BARK, false);
    }
    match shape {
        Tree => b.lathe(
            &[(0.0, -0.2), (0.45, 0.05), (0.35, 0.3), (0.0, 0.5)],
            7,
            0.0,
            LEAVES,
            LEAVES,
            false,
        ),
        Pine => {
            b.lathe(&[(0.5, -0.3), (0.05, 0.1)], 8, 0.0, LEAVES, LEAVES, true);
            b.lathe(&[(0.4, -0.05), (0.03, 0.3)], 8, 0.3, LEAVES, LEAVES, true);
            b.lathe(&[(0.3, 0.15), (0.0, 0.5)], 8, 0.6, LEAVES, LEAVES, true);
        }
        Poplar => b.lathe(
            &[
                (0.0, -0.3),
                (0.3, -0.1),
                (0.35, 0.2),
                (0.2, 0.4),
                (0.0, 0.5),
            ],
            6,
            0.0,
            LEAVES,
            LEAVES,
            false,
        ),
        _ => b.lathe(
            &[(0.0, -0.5), (0.5, -0.2), (0.4, 0.2), (0.0, 0.5)],
            6,
            0.0,
            LEAVES,
            LEAVES,
            false,
        ),
    }
}
