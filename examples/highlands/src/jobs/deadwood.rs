use std::f32::consts::TAU;

use engine_core::math;
use glam::Vec3;

use super::trees::{Build, Dice, Limb, Stem, Stubs, heading, stubs};
use super::wood::{End, Ring};

const WEATHERED: [f32; 3] = [0.74, 0.72, 0.7];

const ROTTING: [f32; 3] = [0.62, 0.64, 0.52];

const CUT: [f32; 3] = [0.7, 0.66, 0.6];

const STUMP_RINGS: [f32; 5] = [-0.3, 0.0, 0.3, 0.6, 1.0];

pub(crate) fn snag(seed: u32, build: &mut Build) {
    let dice = Dice::new(seed);
    let height = dice.span(1, [5.0, 9.0]);
    let stem = Stem {
        height: 2.2 * height,
        girth: dice.span(2, [0.15, 0.24]),
        taper: 0.8,
        flare: 0.5,
        lean: heading(TAU * dice.roll(3)) * dice.span(4, [0.03, 0.1]),
        bend: Vec3::ZERO,
    };
    let rings: Vec<Ring> = stem
        .rings(16, |_| WEATHERED)
        .into_iter()
        .filter(|ring| ring.at.y <= height)
        .collect();
    build.tube(&rings, [8, 6, 4, 0], [End::Open, End::Broken]);
    stubs(
        build,
        dice,
        &stem,
        &Stubs {
            count: 10,
            heights: [0.3 * height, 0.95 * height],
            lengths: [0.3, 1.2],
            tint: WEATHERED,
            key: 0x0dea_d000,
        },
    );
}

pub(crate) fn log(seed: u32, build: &mut Build) {
    let dice = Dice::new(seed);
    let length = dice.span(1, [5.0, 8.0]);
    let thick = dice.span(2, [0.17, 0.24]);
    let sag = dice.span(3, [-0.25, 0.25]);
    let rings: Vec<Ring> = (0..=6)
        .map(|ring| {
            let along = ring as f32 / 6.0;
            let radius = thick * (1.0 - 0.4 * along);
            let bow = sag * math::sin(std::f32::consts::PI * along);
            Ring {
                at: Vec3::new((along - 0.5) * length, 0.55 * radius, bow),
                radius,
                tint: ROTTING,
            }
        })
        .collect();
    build.tube(&rings, [8, 6, 4, 0], [End::Broken; 2]);
    for stub in 0..4_u32 {
        let along = dice.span(0x100 + stub, [0.25, 0.9]);
        let at = Vec3::new((along - 0.5) * length, 0.55 * thick, sag * 0.5);
        let limb = Limb {
            start: at,
            out: heading(TAU * dice.roll(0x200 + stub)),
            length: dice.span(0x300 + stub, [0.3, 0.9]),
            rise: dice.span(0x400 + stub, [0.3, 1.2]),
            curl: 0.0,
        };
        build.tube(
            &limb.rings([0.035, 0.015], [ROTTING; 2], 2),
            [4, 3, 0, 0],
            [End::Open, End::Broken],
        );
    }
}

pub(crate) fn stump(seed: u32, build: &mut Build) {
    let dice = Dice::new(seed);
    let height = dice.span(1, [0.35, 0.7]);
    let girth = dice.span(2, [0.2, 0.32]);
    let lean = heading(TAU * dice.roll(3)) * 0.06;
    let rings: Vec<Ring> = STUMP_RINGS
        .iter()
        .map(|&along| {
            let y = match along < 0.0 {
                true => along,
                false => along * height,
            };
            let flare = 1.0 + 0.9 * math::exp(-math::max(y, 0.0) / 0.22);
            Ring {
                at: Vec3::new(0.0, y, 0.0) + lean * math::max(y, 0.0),
                radius: girth * flare,
                tint: CUT,
            }
        })
        .collect();
    build.tube(&rings, [8, 5, 0, 0], [End::Open, End::Broken]);
}
