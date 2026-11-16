use std::f32::consts::TAU;

use engine_core::math;
use glam::Vec3;

use super::sprigs::Sprig;
use super::trees::{Age, Build, Dice, Limb, Stem, blend, heading, rolled, side_of};
use super::wood::End;

const GOLDEN_TURN: f32 = 2.399_963;

const BIRCH_BARK: [[f32; 3]; 2] = [[0.42, 0.4, 0.38], [1.0, 1.0, 1.0]];

const BIRCH_TWIG: [f32; 3] = [0.42, 0.33, 0.3];

const SHRUB_LEAF: [f32; 3] = [0.8, 0.86, 0.7];

const SHRUB_SPRAYS: u32 = 26;

const FORK_NODES: u32 = 4;

struct BirchForm {
    height: [f32; 2],
    girth: [f32; 2],
    limbs: u32,
    first: f32,
    spread: f32,
    spray: f32,
    weep: f32,
    dark: f32,
    nodes: u32,
    forks: u32,
}

const BIRCH_YOUNG: BirchForm = BirchForm {
    height: [4.0, 6.5],
    girth: [0.035, 0.055],
    limbs: 9,
    first: 0.3,
    spread: 0.2,
    spray: 0.5,
    weep: 0.5,
    dark: 0.3,
    nodes: 4,
    forks: 1,
};

const BIRCH_GROWN: BirchForm = BirchForm {
    height: [12.0, 16.0],
    girth: [0.13, 0.18],
    limbs: 12,
    first: 0.32,
    spread: 0.26,
    spray: 0.8,
    weep: 0.9,
    dark: 1.0,
    nodes: 7,
    forks: 3,
};

const BIRCH_OLD: BirchForm = BirchForm {
    height: [17.0, 21.0],
    girth: [0.24, 0.32],
    limbs: 13,
    first: 0.3,
    spread: 0.3,
    spray: 0.9,
    weep: 1.15,
    dark: 2.2,
    nodes: 8,
    forks: 3,
};

struct Crown {
    center: Vec3,
    radius: f32,
}

pub(crate) fn birch(age: Age, seed: u32, build: &mut Build) {
    let form = match age {
        Age::Young => &BIRCH_YOUNG,
        Age::Grown => &BIRCH_GROWN,
        Age::Old => &BIRCH_OLD,
    };
    let dice = Dice::new(seed);
    let stem = Stem {
        height: dice.span(1, form.height),
        girth: dice.span(2, form.girth),
        taper: 0.7,
        flare: 0.35,
        lean: heading(TAU * dice.roll(5)) * dice.span(6, [0.0, 0.05]),
        bend: heading(TAU * dice.roll(7)) * (0.03 * dice.span(8, [0.3, 1.0]) * form.height[0]),
    };
    let crown = Crown {
        center: stem.axis(0.62 * stem.height),
        radius: 0.7 * form.spread * stem.height,
    };
    let bark = |y: f32| {
        let pale = blend(BIRCH_BARK[0], BIRCH_BARK[1], (y - 0.4 * form.dark) / (0.6 * form.dark));
        blend(pale, BIRCH_TWIG, (y / stem.height - 0.8) / 0.12)
    };
    build.tube(&stem.rings(10, bark), [8, 6, 4, 3], [End::Open; 2]);
    for index in 0..form.limbs {
        let key = (index + 1) << 16;
        let rise = (index as f32 + dice.span(key ^ 1, [0.2, 0.8])) / form.limbs as f32;
        let y = stem.height * (form.first + (0.9 - form.first) * rise);
        let limb = Limb {
            start: stem.axis(y),
            out: heading(GOLDEN_TURN * index as f32 + dice.span(key ^ 2, [-0.5, 0.5])),
            length: stem.height
                * form.spread
                * (1.0 - 0.45 * rise)
                * dice.span(key ^ 3, [0.8, 1.2]),
            rise: dice.span(key ^ 4, [0.9, 1.05]),
            curl: -0.45,
        };
        let thick = math::max(0.5 * stem.radius(y), 0.012);
        build.tube(
            &limb.rings([thick, 0.008], [bark(y), BIRCH_TWIG], 4),
            [4, 3, 3, 0],
            [End::Open; 2],
        );
        for node in 0..form.nodes {
            let along = 0.4 + 0.6 * (node as f32 + 0.5) / form.nodes as f32;
            sprays(build, form, dice, (&limb, along, key | (node + 1) << 8), &crown);
        }
        for fork in 0..form.forks {
            let turn = [-0.8, 0.8][fork as usize % 2] * dice.span(key ^ (5 + fork), [0.8, 1.2]);
            let twig = Limb {
                start: limb.at(0.35 + 0.25 * fork as f32),
                out: rolled(limb.out, Vec3::Y, turn),
                length: 0.45 * limb.length,
                rise: 0.5,
                curl: -0.35,
            };
            build.tube(
                &twig.rings([0.35 * thick, 0.006], [BIRCH_TWIG; 2], 2),
                [3, 0, 0, 0],
                [End::Open; 2],
            );
            for node in 0..FORK_NODES {
                let along = 0.3 + 0.7 * (node as f32 + 0.5) / FORK_NODES as f32;
                let id = key | (0x40 + fork * 8 + node) << 8;
                sprays(build, form, dice, (&twig, along, id), &crown);
            }
        }
    }
    let top = Limb {
        start: stem.axis(0.86 * stem.height),
        out: heading(TAU * dice.roll(9)),
        length: 0.14 * stem.height,
        rise: 1.45,
        curl: 0.0,
    };
    for (node, along) in [(1, 0.35), (2, 0.7), (3, 1.0)] {
        sprays(build, form, dice, (&top, along, 0x00ff_0000 | node << 8), &crown);
    }
}

fn sprays(
    build: &mut Build,
    form: &BirchForm,
    dice: Dice,
    (limb, along, key): (&Limb, f32, u32),
    crown: &Crown,
) {
    let (at, way) = (limb.at(along), limb.way(along));
    let tint = [1.0; 3];
    let open = math::clamp(0.3 + 0.7 * at.distance(crown.center) / crown.radius, 0.0, 1.0);
    for (id, hold) in [(key, 0.7), (key | 3, 0.9)] {
        let drift = heading(TAU * dice.roll(id ^ 0x34)) * 0.3 + limb.out * 0.3;
        let hanging = (drift - Vec3::Y).normalize_or(Vec3::NEG_Y);
        build.sprig(&Sprig {
            at,
            along: hanging,
            across: rolled(side_of(hanging), hanging, TAU * dice.roll(id ^ 0x35)),
            length: form.weep * dice.span(id ^ 0x36, [0.7, 1.2]),
            heart: crown.center,
            open,
            tint,
            key: id,
            hold: match id {
                0x00ff_0300 => 0.0,
                _ => hold,
            },
        });
    }
    for (side, turn) in [(1_u32, -1.0_f32), (2, 1.0)] {
        let id = key | side;
        let (sin, cos) = math::sin_cos(dice.span(id ^ 0x4100, [0.7, 1.1]));
        let flat = way * cos + side_of(way) * (turn * sin);
        let pointing = (flat - Vec3::Y * dice.span(id ^ 0x4200, [0.2, 0.7])).normalize_or(way);
        build.sprig(&Sprig {
            at,
            along: pointing,
            across: rolled(side_of(pointing), pointing, TAU * dice.roll(id ^ 0x4300)),
            length: form.spray * dice.span(id ^ 0x4400, [0.7, 1.2]),
            heart: crown.center,
            open,
            tint,
            key: id,
            hold: 1.0,
        });
    }
}

pub(crate) fn shrub(seed: u32, build: &mut Build) {
    let dice = Dice::new(seed);
    let spin = TAU * dice.roll(1);
    for index in 0..SHRUB_SPRAYS {
        let up = 0.95 - 0.75 * (index as f32 + 0.5) / SHRUB_SPRAYS as f32;
        let flat = math::max(1.0 - up * up, 0.0).sqrt();
        let pointing = heading(spin + GOLDEN_TURN * index as f32) * flat + Vec3::Y * up;
        build.sprig(&Sprig {
            at: Vec3::new(0.0, 0.05, 0.0) + pointing * 0.08,
            along: pointing,
            across: rolled(side_of(pointing), pointing, TAU * dice.roll(index ^ 0x4100)),
            length: dice.span(index ^ 0x4200, [0.5, 0.8]),
            heart: Vec3::new(0.0, -0.25, 0.0),
            open: 0.4 + 0.6 * up,
            tint: SHRUB_LEAF,
            key: 0x5000 | index,
            hold: match index {
                0 => 0.0,
                _ => 1.0,
            },
        });
    }
}
