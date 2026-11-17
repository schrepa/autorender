use std::f32::consts::{PI, TAU};

use engine_core::math;
use glam::{Vec2, Vec3};

use super::sprigs::Sprig;
use super::trees::{Age, Build, Dice, Limb, Stem, Stubs, blend, heading, rolled, side_of, stubs};
use super::wood::End;

const GOLDEN_TURN: f32 = 2.399_963;

const DEAD_WOOD: [f32; 3] = [0.62, 0.6, 0.57];

const SPRUCE_BARK: [[f32; 3]; 2] = [[1.0, 1.0, 1.0], [0.88, 0.86, 0.84]];

const BOUGH_WOOD: [[f32; 3]; 2] = [[0.62, 0.58, 0.54], [0.5, 0.46, 0.42]];

const PINE_BARK: [[f32; 3]; 2] = [[0.72, 0.66, 0.62], [1.0, 0.8, 0.62]];

const BARE_FROM: f32 = 0.2;

const LIMB_TUFTS: [(f32, f32); 5] = [
    (1.0, 1.0),
    (0.86, 0.95),
    (0.72, 0.9),
    (0.6, 0.8),
    (0.48, 0.7),
];

const FORK_TUFTS: [(f32, f32); 3] = [(1.0, 0.85), (0.72, 0.8), (0.45, 0.7)];

struct SpruceForm {
    height: [f32; 2],
    girth: [f32; 2],
    skirt: [f32; 2],
    reach: [f32; 2],
    spacing: f32,
    sprig: f32,
    droop: f32,
    gaps: f32,
    dead: u32,
}

const SPRUCE_YOUNG: SpruceForm = SpruceForm {
    height: [3.2, 5.0],
    girth: [0.045, 0.07],
    skirt: [0.2, 0.45],
    reach: [0.9, 1.3],
    spacing: 0.3,
    sprig: 0.4,
    droop: 0.15,
    gaps: 0.0,
    dead: 0,
};

const SPRUCE_GROWN: SpruceForm = SpruceForm {
    height: [14.0, 18.0],
    girth: [0.19, 0.26],
    skirt: [2.2, 3.6],
    reach: [2.5, 3.1],
    spacing: 0.52,
    sprig: 0.66,
    droop: 0.4,
    gaps: 0.08,
    dead: 8,
};

const SPRUCE_OLD: SpruceForm = SpruceForm {
    height: [22.0, 27.0],
    girth: [0.33, 0.43],
    skirt: [5.5, 8.5],
    reach: [3.6, 4.3],
    spacing: 0.62,
    sprig: 0.78,
    droop: 0.6,
    gaps: 0.2,
    dead: 16,
};

pub(crate) fn spruce(age: Age, seed: u32, build: &mut Build) {
    let form = match age {
        Age::Young => &SPRUCE_YOUNG,
        Age::Grown => &SPRUCE_GROWN,
        Age::Old => &SPRUCE_OLD,
    };
    let dice = Dice::new(seed);
    let stem = Stem {
        height: dice.span(1, form.height),
        girth: dice.span(2, form.girth),
        taper: 0.8,
        flare: 0.5,
        lean: Vec3::ZERO,
        bend: Vec3::ZERO,
    };
    let skirt = dice.span(3, form.skirt);
    let reach = dice.span(4, form.reach);
    let shaded = |y: f32| blend(SPRUCE_BARK[0], SPRUCE_BARK[1], (y - skirt) / (0.3 * stem.height));
    build.tube(&stem.rings(10, shaded), [8, 6, 4, 3], [End::Open; 2]);
    stubs(
        build,
        dice,
        &stem,
        &Stubs {
            count: form.dead,
            heights: [0.3 * skirt, skirt + 0.1 * stem.height],
            lengths: [0.4, 0.25 + 0.4 * reach],
            tint: DEAD_WOOD,
            key: 0x0dea_d000,
        },
    );
    let mut y = skirt;
    let mut whorl = 0_u32;
    while y < stem.height - 0.4 * form.spacing {
        let rise = math::clamp((y - skirt) / (stem.height - skirt), 0.0, 1.0);
        let crown = reach * math::pow(1.0 - rise, 0.9) * (0.6 + 0.4 * math::min(rise / 0.1, 1.0))
            + 0.15;
        let count = 4 + dice.pick(100 + whorl, 3);
        let twist = TAU * dice.roll(200 + whorl);
        for branch in 0..count {
            let key = (whorl + 1) << 16 | branch << 12;
            if rise < 0.85 && dice.roll(key ^ 0x21) < form.gaps {
                continue;
            }
            let turn = (branch as f32 + 0.5 * (dice.roll(key ^ 0x22) - 0.5)) / count as f32;
            let out = heading(twist + TAU * turn);
            let limb = Limb {
                start: stem.axis(y) + out * (0.5 * stem.radius(y)),
                out,
                length: crown * dice.span(key ^ 0x23, [0.8, 1.2]),
                rise: -form.droop + (0.75 + form.droop) * math::pow(rise, 1.3),
                curl: 0.75 * form.droop * (1.0 - rise),
            };
            bough(build, form, dice, (&limb, key), (crown, rise));
        }
        y += form.spacing * dice.span(300 + whorl, [0.85, 1.15]);
        whorl += 1;
    }
    let top = stem.axis(stem.height - 0.6 * form.sprig);
    for (key, turn, hold) in [(0x70b0, 0.0, 0.0), (0x70b1, 1.1, 0.6), (0x70b2, 2.2, 1.0)] {
        build.sprig(&Sprig {
            at: top,
            along: Vec3::Y,
            across: heading(turn),
            length: 0.9 * form.sprig,
            heart: top - Vec3::Y,
            open: 1.0,
            tint: [1.0; 3],
            key,
            hold,
        });
    }
}

fn bough(
    build: &mut Build,
    form: &SpruceForm,
    dice: Dice,
    (limb, key): (&Limb, u32),
    (crown, rise): (f32, f32),
) {
    if limb.length > 1.2 {
        let thick = 0.012 + 0.012 * math::min(limb.length, 3.0);
        build.tube(
            &limb.rings([thick, 0.006], BOUGH_WOOD, 2),
            [3, 0, 0, 0],
            [End::Open; 2],
        );
    }
    let nodes = math::clamp((limb.length / (0.42 * form.sprig)).ceil(), 1.0, 11.0) as u32;
    let tint = [1.0; 3];
    let heart = |at: Vec3| Vec3::new(0.0, at.y - 0.6 * Vec2::new(at.x, at.z).length() - 0.4, 0.0);
    let open = |at: Vec3| {
        let out = math::min(Vec2::new(at.x, at.z).length() / crown, 1.0);
        (0.12 + 0.88 * math::pow(out, 1.3)) * (0.7 + 0.3 * rise)
    };
    for node in 0..nodes {
        let along = BARE_FROM + (1.0 - BARE_FROM) * (node as f32 + 0.5) / nodes as f32;
        let (at, way) = (limb.at(along), limb.way(along));
        for (side, turn) in [(0_u32, -1.0_f32), (1, 1.0)] {
            let id = key | node << 4 | side;
            let (sin, cos) = math::sin_cos(dice.span(id ^ 0x4100, [0.75, 1.1]));
            let flat = way * cos + side_of(way) * (turn * sin);
            let hang = form.droop * dice.span(id ^ 0x4200, [0.5, 1.4]);
            let pointing = (flat - Vec3::Y * hang).normalize_or(way);
            let roll = turn * dice.span(id ^ 0x4300, [0.35, 1.3]);
            build.sprig(&Sprig {
                at,
                along: pointing,
                across: rolled(side_of(pointing), pointing, roll),
                length: form.sprig * dice.span(id ^ 0x4400, [0.7, 1.2]) * (1.0 - 0.3 * along),
                heart: heart(at),
                open: open(at),
                tint,
                key: id,
                hold: 1.0,
            });
        }
        let id = key | node << 4 | 2;
        let drift = side_of(way) * dice.span(id ^ 0x4500, [-0.5, 0.5]);
        let sinking = Vec3::Y * (0.6 + form.droop * dice.span(id ^ 0x4600, [0.6, 1.4]));
        let pointing = (way * 0.45 + drift - sinking).normalize_or(Vec3::NEG_Y);
        build.sprig(&Sprig {
            at,
            along: pointing,
            across: rolled(side_of(pointing), pointing, dice.span(id ^ 0x4700, [-0.6, 0.6])),
            length: form.sprig * dice.span(id ^ 0x4800, [0.6, 1.0]) * (1.0 - 0.3 * along),
            heart: heart(at),
            open: 0.85 * open(at),
            tint,
            key: id,
            hold: 0.9,
        });
    }
    let (at, way) = (limb.at(0.9), limb.way(1.0));
    build.sprig(&Sprig {
        at,
        along: way,
        across: rolled(side_of(way), way, dice.span(key ^ 0x34, [-0.5, 0.5])),
        length: 0.95 * form.sprig,
        heart: heart(at),
        open: open(limb.at(1.0)),
        tint,
        key: key | 0xf00,
        hold: 0.5,
    });
}

struct PineForm {
    height: [f32; 2],
    girth: [f32; 2],
    crown: [f32; 2],
    reach: [f32; 2],
    limbs: u32,
    forks: u32,
    clump: f32,
    flat: f32,
    bend: f32,
    dead: u32,
}

const PINE_YOUNG: PineForm = PineForm {
    height: [3.5, 5.5],
    girth: [0.05, 0.075],
    crown: [0.1, 0.18],
    reach: [1.0, 1.4],
    limbs: 14,
    forks: 0,
    clump: 0.4,
    flat: 0.0,
    bend: 0.05,
    dead: 0,
};

const PINE_GROWN: PineForm = PineForm {
    height: [13.0, 17.0],
    girth: [0.2, 0.27],
    crown: [0.5, 0.6],
    reach: [2.4, 3.0],
    limbs: 13,
    forks: 2,
    clump: 0.55,
    flat: 0.3,
    bend: 0.25,
    dead: 2,
};

const PINE_OLD: PineForm = PineForm {
    height: [18.0, 23.0],
    girth: [0.3, 0.4],
    crown: [0.62, 0.72],
    reach: [3.6, 4.6],
    limbs: 12,
    forks: 2,
    clump: 0.65,
    flat: 1.0,
    bend: 0.5,
    dead: 5,
};

pub(crate) fn pine(age: Age, seed: u32, build: &mut Build) {
    let form = match age {
        Age::Young => &PINE_YOUNG,
        Age::Grown => &PINE_GROWN,
        Age::Old => &PINE_OLD,
    };
    let dice = Dice::new(seed);
    let swing = heading(TAU * dice.roll(5));
    let stem = Stem {
        height: dice.span(1, form.height),
        girth: dice.span(2, form.girth),
        taper: 0.6,
        flare: 0.4,
        lean: swing * dice.span(6, [0.0, 0.04]),
        bend: heading(TAU * dice.roll(7)) * (form.bend * dice.span(8, [0.5, 1.0])),
    };
    let crown = dice.span(3, form.crown) * stem.height;
    let reach = dice.span(4, form.reach);
    let bark = |y: f32| blend(PINE_BARK[0], PINE_BARK[1], (y / stem.height - 0.35) / 0.25);
    build.tube(&stem.rings(10, bark), [8, 6, 4, 3], [End::Open; 2]);
    stubs(
        build,
        dice,
        &stem,
        &Stubs {
            count: form.dead,
            heights: [0.3 * stem.height, crown],
            lengths: [0.4, 1.3],
            tint: DEAD_WOOD,
            key: 0x0dea_d000,
        },
    );
    for index in 0..form.limbs {
        let key = (index + 1) << 16;
        let rise = (index as f32 + dice.span(key ^ 1, [0.2, 0.8])) / form.limbs as f32;
        let y = crown + (0.97 * stem.height - crown) * rise;
        let round = math::sin(PI * (0.12 + 0.8 * rise));
        let level = (0.55 + 0.45 * rise) * (1.0 - 0.5 * math::max(rise - 0.8, 0.0) / 0.2);
        let limb = Limb {
            start: stem.axis(y),
            out: heading(GOLDEN_TURN * index as f32 + dice.span(key ^ 2, [-0.4, 0.4])),
            length: reach
                * (round + (level - round) * form.flat)
                * dice.span(key ^ 3, [0.8, 1.2]),
            rise: 0.15 - 0.2 * form.flat + (0.8 - 0.15 * form.flat) * rise,
            curl: 0.3,
        };
        let thick = math::max(0.45 * stem.radius(y), 0.02);
        build.tube(
            &limb.rings([thick, 0.012], [bark(y), PINE_BARK[1]], 4),
            [5, 4, 3, 0],
            [End::Open; 2],
        );
        let shade = 0.75 + 0.25 * rise;
        let size = form.clump * dice.span(key ^ 4, [0.85, 1.15]);
        for (tuft, (along, grown)) in LIMB_TUFTS.into_iter().enumerate() {
            let at = limb.at(along) + Vec3::Y * (0.3 * grown * size);
            let id = key | (tuft as u32 + 1) << 8;
            clump(build, dice, (at, grown * size), (id, shade * (0.8 + 0.2 * along)));
        }
        for fork in 0..form.forks {
            let turn = [-0.75, 0.75][fork as usize % 2] * dice.span(key ^ (5 + fork), [0.8, 1.2]);
            let twig = Limb {
                start: limb.at(0.5),
                out: rolled(limb.out, Vec3::Y, turn),
                length: 0.45 * limb.length,
                rise: limb.rise + 0.2,
                curl: 0.3,
            };
            build.tube(
                &twig.rings([0.4 * thick, 0.008], [PINE_BARK[1]; 2], 2),
                [3, 3, 0, 0],
                [End::Open; 2],
            );
            for (tuft, (along, grown)) in FORK_TUFTS.into_iter().enumerate() {
                let at = twig.at(along) + Vec3::Y * (0.25 * grown * size);
                let id = key | (0x10 + fork * 4 + tuft as u32) << 8;
                clump(build, dice, (at, grown * size), (id, shade));
            }
        }
    }
    let top = stem.axis(stem.height) - Vec3::Y * (0.2 * form.clump);
    clump(build, dice, (top, form.clump), (0x00ff_0000, 1.0));
}

fn clump(build: &mut Build, dice: Dice, (center, size): (Vec3, f32), (key, shade): (u32, f32)) {
    let count = match size >= 0.4 {
        true => 9_u32,
        false => 6,
    };
    let tint = [1.0; 3];
    let spin = TAU * dice.roll(key ^ 0x34);
    for index in 0..count {
        let id = key | index;
        let up = 1.0 - 1.25 * (index as f32 + 0.5) / count as f32;
        let flat = math::max(1.0 - up * up, 0.0).sqrt();
        let pointing = heading(spin + GOLDEN_TURN * index as f32) * flat + Vec3::Y * up;
        build.sprig(&Sprig {
            at: center - Vec3::Y * (0.15 * size) + pointing * (0.1 * size),
            along: pointing,
            across: rolled(side_of(pointing), pointing, TAU * dice.roll(id ^ 0x4100)),
            length: size * dice.span(id ^ 0x4200, [0.9, 1.2]),
            heart: center - Vec3::Y * (0.6 * size),
            open: shade * (0.35 + 0.65 * (up + 0.25) / 1.25),
            tint,
            key: id,
            hold: match (index, key) {
                (0, 0x00ff_0000) => 0.0,
                (0, _) => 0.4,
                _ => 1.0,
            },
        });
    }
}
