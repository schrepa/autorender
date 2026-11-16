use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{FrameTime, Var, Views};
use engine_heightfield::errors::ERROR_CELLS;
use engine_math::{GRAVITY, cos, max, sin_cos};
use engine_places_rendering::wind::ViewWind;
use engine_render::{Allocation, Bind, Buffer, Install, Lifetime, Prepare};

use crate::{TileWater, Water};
use crate::gpu::{
    WATER_GUST_FINE_SALT, WATER_GUST_SALT, WATER_NOISE_REACH, WATER_NOISE_WRAP,
    WATER_PUSH_FINE_SALT, WATER_PUSH_SALT, WATER_STEEPEST, WaterWave, WaterWaveRow, WaterWavesLook,
};
use crate::settings::WaterWaves;
use crate::shaders::water::waves;
use crate::water::WaterGpu;

const GOLDEN_TURN: f64 = 0.618_033_99;
const FIRST_TURN: f64 = 0.37;
const GOLDEN_ANGLE: f64 = 2.399_963_2;
const NOISE_SLOPE_RMS: f64 = 0.625_2;
const SPAN_SQUARED_PER_VARIANCE: f64 = 12.0;

struct Wave {
    heading: [f64; 2],
    number: f64,
    pace: f64,
    start: f64,
}

#[derive(Default)]
pub(crate) struct Swell {
    look: WaterWavesLook,
    rows: Vec<WaterWaveRow>,
}

impl Swell {
    pub(crate) fn calm(&self, speed: f32) -> f32 {
        let [low, high] = self.look.still.map(f64::from);
        1.0 - smoothstep(low, high, f64::from(speed)) as f32
    }

    pub(crate) fn left(&self, piece: f32) -> f32 {
        let look = &self.look;
        let blurs = [0, 1].map(|octave| [look.pushes[octave][2], look.rides[octave][2]]);
        left(&self.rows, look.shown, blurs, piece)
    }

    pub(crate) fn height(&self, [x, z]: [f32; 2], depth: f32, speed: f32) -> f32 {
        let look = &self.look;
        let calm = self.calm(speed);
        if calm <= 0.0 || depth <= 0.0 {
            return 0.0;
        }
        let spot = [x - look.anchor[0], z - look.anchor[1]];
        let gusts = self.gusts(spot);
        let common = 1.0 + look.gusts[0][3] * gusts[0] + look.gusts[1][3] * gusts[1];
        let lift = max(common, look.calmest) - 1.0;
        let tight = 1.0 / (std::f32::consts::PI * max(look.breaking * depth, 1.0e-4));
        let mut at = self.pushed(spot);
        let mut height = 0.0;
        for WaterWaveRow { wave, .. } in &self.rows {
            let grouped = gusts[0] * wave.group[0] + gusts[1] * wave.group[1];
            let open = calm * max(1.0 + lift * wave.gusted + grouped, look.calmest) * wave.steep;
            let tall = open * max(wave.wavelength * tight, 1.0 / WATER_STEEPEST);
            let steep = open / (1.0 + tall * tall * tall * tall).sqrt().sqrt();
            let phase = wave.number * (wave.heading[0] * at[0] + wave.heading[1] * at[1])
                + wave.phase;
            let squeeze = look.chop * steep;
            let first = phase - squeeze * cos(phase);
            let (risen, level) = sin_cos(phase - squeeze * cos(first));
            let reach = steep / wave.number;
            let moved = look.chop * reach * level;
            height += reach * risen;
            at = [
                at[0] - wave.heading[0] * moved,
                at[1] - wave.heading[1] * moved,
            ];
        }
        height
    }

    pub(crate) fn reach(&self) -> f32 {
        let look = &self.look;
        let gust = std::f32::consts::SQRT_2 / WATER_NOISE_REACH;
        let lift = max(
            gust * (look.gusts[0][3].abs() + look.gusts[1][3].abs()),
            (look.calmest - 1.0).abs(),
        );
        self.rows
            .iter()
            .map(|WaterWaveRow { wave, .. }| {
                let grouped = gust * (wave.group[0].abs() + wave.group[1].abs());
                let open = max(1.0 + lift * wave.gusted.abs() + grouped, look.calmest.abs());
                open * wave.steep.abs() / wave.number.abs()
            })
            .sum()
    }

    pub(crate) fn errors(&self, still: &[u16], water: &TileWater, spacing: f32) -> Arc<[u16]> {
        let cells = ERROR_CELLS as usize;
        let mut calm: Vec<f32> = (0..cells * cells)
            .map(|cell| {
                let (x, z) = ((2 * (cell % cells)) as i32, (2 * (cell / cells)) as i32);
                (0..9)
                    .map(|corner| {
                        let (flow, ..) = water.sample_info(x + corner % 3, z + corner / 3);
                        self.calm(flow.length())
                    })
                    .fold(0.0, max)
            })
            .collect();
        let mut waved = Vec::with_capacity(still.len());
        let mut side = cells;
        let mut piece = 2.0 * spacing;
        while side > 0 && waved.len() + calm.len() <= still.len() {
            let left = self.left(piece);
            let level = &still[waved.len()..waved.len() + calm.len()];
            waved.extend(level.iter().zip(&calm).map(|(&bits, &calm)| {
                let held = f16_value(bits);
                if held < 0.0 {
                    bits
                } else {
                    half_up(held + calm * left)
                }
            }));
            let row = side;
            side /= 2;
            piece *= 2.0;
            calm = (0..side * side)
                .map(|cell| {
                    let first = 2 * (cell / side) * row + 2 * (cell % side);
                    [0, 1, row, row + 1]
                        .iter()
                        .map(|step| calm[first + step])
                        .fold(0.0, max)
                })
                .collect();
        }
        Arc::from(waved)
    }

    fn pushed(&self, spot: [f32; 2]) -> [f32; 2] {
        let mut at = spot;
        for (field, salt) in self.look.pushes.iter().zip([WATER_PUSH_SALT, WATER_PUSH_FINE_SALT]) {
            let [x, z, per_meter, reach] = *field;
            let [east, north] = slope([spot[0] * per_meter + x, spot[1] * per_meter + z], salt);
            at = [at[0] - reach * north, at[1] + reach * east];
        }
        at
    }

    fn gusts(&self, spot: [f32; 2]) -> [f32; 2] {
        let [along, across] = self.look.downwind;
        let blown = [
            spot[0] * along + spot[1] * across,
            spot[1] * along - spot[0] * across,
        ];
        let patch = |octave: usize, salt: u32| {
            let [x, z, per_meter, _] = self.look.gusts[octave];
            noise([blown[0] * per_meter + x, blown[1] * per_meter + z], salt) / WATER_NOISE_REACH
        };
        [patch(0, WATER_GUST_SALT), patch(1, WATER_GUST_FINE_SALT)]
    }
}

#[derive(Resource, Default)]
pub(crate) struct Waves {
    made: Option<(WaterWaves, [f32; 2])>,
    phased: Option<([f32; 2], f64, f32)>,
    makes: u64,
    waves: Vec<Wave>,
    table: Vec<WaterWaveRow>,
    fan: [f32; 2],
    travel: [f64; 2],
    blurs: [[f32; 2]; 2],
    swell: Arc<Swell>,
}

impl Waves {
    pub(crate) fn makes(&self) -> u64 {
        self.makes
    }

    pub(crate) fn swell(&self) -> &Arc<Swell> {
        &self.swell
    }

    fn step(&mut self, water: &Water, wind: [f32; 2], (eye, seconds): ([f32; 2], f64)) {
        let look = &water.waves;
        let (downwind, speed) = downwind(wind);
        let remade = self.made != Some((*look, wind));
        if remade {
            self.make(look, downwind);
            self.made = Some((*look, wind));
            self.makes += 1;
        }
        let phased = Some((eye, seconds, water.subdivision_error));
        if !remade && self.phased == phased {
            return;
        }
        self.phased = phased;
        let anchor = eye.map(f64::from);
        let along = |way: [f64; 2]| way[0] * anchor[0] + way[1] * anchor[1];
        for (row, wave) in self.table.iter_mut().zip(&self.waves) {
            let phase = wave.number * along(wave.heading) - wave.pace * seconds + wave.start;
            row.wave.phase = phase.rem_euclid(TAU) as f32;
        }
        let wrap = f64::from(WATER_NOISE_WRAP);
        let gusts = [0, 1].map(|octave| {
            let per_meter = 1.0 / f64::from(look.gust_size[octave]);
            let drifted = f64::from(look.gust_drift[octave]) * speed * seconds;
            [
                ((along(downwind) - drifted) * per_meter).rem_euclid(wrap) as f32,
                (along([-downwind[1], downwind[0]]) * per_meter) as f32,
                per_meter as f32,
                look.gust_strength[octave],
            ]
        });
        let rides =
            [0, 1].map(|octave| self.travel.map(|axis| axis * f64::from(look.push_ride[octave])));
        let pushes = [0, 1].map(|octave| {
            let [per_meter, _] = self.blurs[octave];
            let [x, z] = [0, 1].map(|axis| {
                let ridden = rides[octave][axis] * seconds;
                ((anchor[axis] - ridden) * f64::from(per_meter)).rem_euclid(wrap) as f32
            });
            let reach = f64::from(look.push[octave]) / NOISE_SLOPE_RMS;
            [x, z, per_meter, reach as f32]
        });
        self.swell = Arc::new(Swell {
            look: WaterWavesLook {
                gusts,
                pushes,
                rides: [0, 1].map(|octave| {
                    let [x, z] = rides[octave].map(|axis| axis as f32);
                    [x, z, self.blurs[octave][1], 0.0]
                }),
                anchor: eye,
                downwind: downwind.map(|axis| axis as f32),
                still: look.still,
                shown: look.shown,
                fan: self.fan,
                count: self.table.len() as u32,
                chop: look.chop,
                breaking: look.breaking,
                calmest: look.gust_calmest,
                mesh_error: water.subdivision_error,
                pad: 0.0,
            },
            rows: self.table.clone(),
        });
    }

    fn make(&mut self, look: &WaterWaves, downwind: [f64; 2]) {
        let side = [-downwind[1], downwind[0]];
        let longest = f64::from(look.longest);
        let ratio = f64::from(look.shortest) / longest;
        let tilt = f64::from(look.tilt);
        let spread = look.spread.map(f64::from);
        let grouping = f64::from(look.grouping);
        let drawn: Vec<(f64, u32)> = (0..look.count)
            .map(|index| {
                let draw = pcg(index ^ look.seed);
                let place = (f64::from(index) + unit(draw)) / f64::from(look.count);
                (place, draw)
            })
            .collect();
        let total: f64 = drawn
            .iter()
            .map(|&(place, _)| libm::pow(ratio, 2.0 * tilt * place))
            .sum();
        let scale = f64::from(look.steepness) * (2.0 / total).sqrt();
        let mut widest = 0.0_f64;
        let mut travel = [0.0_f64; 2];
        self.waves.clear();
        self.table.clear();
        for (index, &(place, draw)) in drawn.iter().enumerate() {
            let wavelength = longest * libm::pow(ratio, place);
            let number = (TAU / wavelength) as f32;
            let turn = 2.0 * (index as f64 * GOLDEN_TURN + FIRST_TURN).fract() - 1.0;
            let angle = turn * (spread[0] + (spread[1] - spread[0]) * place);
            let (sine, cosine) = (libm::sin(angle), libm::cos(angle));
            let heading = [
                (downwind[0] * cosine + side[0] * sine) as f32,
                (downwind[1] * cosine + side[1] * sine) as f32,
            ];
            let led = index as f64 * GOLDEN_ANGLE;
            widest = widest.max(angle.abs());
            let pace = (f64::from(GRAVITY) * f64::from(number)).sqrt();
            let weight = libm::pow(ratio, 2.0 * tilt * place) * pace / f64::from(number) / total;
            travel = [0, 1].map(|axis| travel[axis] + weight * f64::from(heading[axis]));
            self.waves.push(Wave {
                heading: heading.map(f64::from),
                number: f64::from(number),
                pace,
                start: TAU * unit(pcg(draw)),
            });
            self.table.push(WaterWaveRow {
                wave: WaterWave {
                    heading,
                    group: [
                        (grouping * libm::cos(led)) as f32,
                        (grouping * libm::sin(led)) as f32,
                    ],
                    number,
                    steep: (scale * libm::pow(ratio, tilt * place)) as f32,
                    gusted: f64::from(look.gust_long).max((2.0 * place).min(1.0)) as f32,
                    wavelength: wavelength as f32,
                    speed: (pace / f64::from(number)) as f32,
                    ..WaterWave::default()
                },
                ..WaterWaveRow::default()
            });
        }
        let mut rests = [[0.0_f64; 3]; 10];
        for row in self.table.iter_mut().rev() {
            let wave = &row.wave;
            let [x, z] = wave.heading.map(f64::from);
            let variance = 0.5 * f64::from(wave.steep) * f64::from(wave.steep);
            let gusted = f64::from(wave.gusted);
            let [broad, fine] = wave.group.map(f64::from);
            let terms = [
                1.0,
                2.0 * gusted,
                gusted * gusted,
                2.0 * broad,
                2.0 * fine,
                broad * broad,
                fine * fine,
                2.0 * broad * fine,
                2.0 * gusted * broad,
                2.0 * gusted * fine,
            ];
            for (sum, term) in rests.iter_mut().zip(terms) {
                for (axis, spread) in sum.iter_mut().zip([x * x, z * z, x * z]) {
                    *axis += variance * term * spread;
                }
            }
            row.rests = rests.map(|[x, z, mixed]| [x as f32, z as f32, mixed as f32, 0.0]);
        }
        self.travel = travel;
        self.blurs = [0, 1].map(|octave| {
            let amount = f64::from(look.push[octave]);
            [
                (1.0 / f64::from(look.push_size[octave])) as f32,
                (SPAN_SQUARED_PER_VARIANCE * amount * amount) as f32,
            ]
        });
        let blurs = self.blurs;
        let lefts: Vec<f32> = self
            .table
            .iter()
            .map(|row| left(&self.table, look.shown, blurs, row.wave.wavelength / look.shown[0]))
            .collect();
        for (row, left) in self.table.iter_mut().zip(lefts) {
            row.wave.left = left;
        }
        let fan = widest.min(FRAC_PI_2);
        self.fan = [libm::cos(fan) as f32, libm::sin(fan) as f32];
    }
}

pub(crate) fn follow(
    water: Res<Water>,
    wind: Res<ViewWind>,
    views: Res<Views>,
    time: Res<FrameTime>,
    mut waves: ResMut<Waves>,
) {
    let eye = views
        .main()
        .map_or([0.0; 2], |main| [main.eye.x, main.eye.z]);
    waves.step(&water, wind.velocity, (eye, time.elapsed));
}

pub(crate) struct WavesGpu {
    rows: Buffer<WaterGpu>,
    written: Option<Arc<Swell>>,
}

impl WavesGpu {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>) -> Self {
        Self {
            rows: cx.buffer("water_waves", Lifetime::Kept),
            written: None,
        }
    }

    pub(crate) fn bound(&self) -> [(Var, Bind<WaterGpu>); 1] {
        [(waves::WATER_WAVES, Bind::Buffer(self.rows))]
    }

    pub(crate) fn prepare(
        &mut self,
        waves: &Waves,
        cx: &mut Prepare<'_, WaterGpu>,
    ) -> WaterWavesLook {
        let swell = &waves.swell;
        let bytes = size_of_val(&swell.rows[..]).max(size_of::<WaterWaveRow>()) as u64;
        let fresh = cx.allocate_buffer(self.rows, bytes) == Allocation::Fresh;
        let written = self
            .written
            .as_ref()
            .is_some_and(|written| Arc::ptr_eq(written, swell));
        if (fresh || !written) && !swell.rows.is_empty() {
            cx.write(self.rows, 0, &swell.rows);
            self.written = Some(Arc::clone(swell));
        }
        swell.look
    }
}

fn left(rows: &[WaterWaveRow], shown: [f32; 2], blurs: [[f32; 2]; 2], piece: f32) -> f32 {
    let [fewest, full] = shown.map(f64::from);
    let piece = f64::from(piece);
    let blur: f64 = blurs
        .iter()
        .map(|&[per_meter, spread]| {
            let cells = (f64::from(per_meter) * piece).max(1.0e-12);
            let kept = smoothstep(fewest, full, 1.0 / cells);
            (1.0 - kept * kept) * f64::from(spread)
        })
        .sum();
    let span = (piece * piece + blur).sqrt();
    let squared: f64 = rows
        .iter()
        .map(|WaterWaveRow { wave, .. }| {
            let wavelength = f64::from(wave.wavelength);
            let carried = smoothstep(fewest, full, wavelength / span);
            let left = 1.0 - carried * libm::cos(PI * piece / wavelength);
            let amplitude = f64::from(wave.steep) / f64::from(wave.number);
            0.5 * amplitude * amplitude * left * left
        })
        .sum();
    squared.sqrt() as f32
}

fn half_up(value: f32) -> u16 {
    let bits = f16_bits(value);
    if value > 0.0 && f16_value(bits) < value {
        bits + 1
    } else {
        bits
    }
}

fn smoothstep(low: f64, high: f64, value: f64) -> f64 {
    if value <= low {
        return 0.0;
    }
    if value >= high {
        return 1.0;
    }
    let share = (value - low) / (high - low);
    share * share * (3.0 - 2.0 * share)
}

fn noise([x, z]: [f32; 2], seed: u32) -> f32 {
    let cell = [x.floor(), z.floor()];
    let within = [x - cell[0], z - cell[1]];
    let [eased_x, eased_z] = within.map(|f| f * f * f * (f * (f * 6.0 - 15.0) + 10.0));
    let low = cell.map(|cell| cell as i32);
    let corner = |east: i32, north: i32| {
        let column = ((low[0] + east) & (WATER_NOISE_WRAP - 1)) as u32;
        let [cosine, sine] = turned(column, low[1].wrapping_add(north) as u32, seed);
        cosine * (within[0] - east as f32) + sine * (within[1] - north as f32)
    };
    let [south_west, south_east, north_west, north_east] =
        [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)];
    south_west
        + eased_x * (south_east - south_west)
        + eased_z * (north_west - south_west)
        + eased_x * eased_z * (south_west - south_east - north_west + north_east)
}

fn slope([x, z]: [f32; 2], seed: u32) -> [f32; 2] {
    let cell = [x.floor(), z.floor()];
    let within = [x - cell[0], z - cell[1]];
    let eased = within.map(|f| f * f * f * (f * (f * 6.0 - 15.0) + 10.0));
    let easing = within.map(|f| 30.0 * f * f * (f * (f - 2.0) + 1.0));
    let low = cell.map(|cell| cell as i32);
    let wrapped =
        |index: i32, step: i32| (index.wrapping_add(step) & (WATER_NOISE_WRAP - 1)) as u32;
    let corner = |east: i32, north: i32| {
        let [cosine, sine] = turned(wrapped(low[0], east), wrapped(low[1], north), seed);
        let value = cosine * (within[0] - east as f32) + sine * (within[1] - north as f32);
        ([cosine, sine], value)
    };
    let [(south_west, a), (south_east, b), (north_west, c), (north_east, d)] =
        [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)];
    let mixed = a - b - c + d;
    [0, 1].map(|axis| {
        let bent = south_west[axis] - south_east[axis] - north_west[axis] + north_east[axis];
        let rise = [b, c][axis] - a + eased[1 - axis] * mixed;
        south_west[axis]
            + eased[0] * (south_east[axis] - south_west[axis])
            + eased[1] * (north_west[axis] - south_west[axis])
            + eased[0] * eased[1] * bent
            + easing[axis] * rise
    })
}

fn turned(column: u32, row: u32, seed: u32) -> [f32; 2] {
    let turn = std::f32::consts::TAU * unit(pcg(column ^ pcg(row ^ seed))) as f32;
    let (sine, cosine) = sin_cos(turn);
    [cosine, sine]
}

fn downwind([x, z]: [f32; 2]) -> ([f64; 2], f64) {
    let (x, z) = (f64::from(x), f64::from(z));
    let speed = (x * x + z * z).sqrt();
    if speed > 0.0 {
        ([x / speed, z / speed].map(|axis| f64::from(axis as f32)), speed)
    } else {
        ([1.0, 0.0], 0.0)
    }
}

fn pcg(value: u32) -> u32 {
    let state = value
        .wrapping_mul(747_796_405)
        .wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn unit(value: u32) -> f64 {
    f64::from(value >> 8) / 16_777_216.0
}

fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if biased == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let exponent = biased as i32 - 127 + 15;
    let (kept, whole, shift) = if exponent > 0 {
        (((exponent as u32) << 10) | (mantissa >> 13), mantissa, 13)
    } else if exponent >= -10 {
        let whole = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        (whole >> shift, whole, shift)
    } else {
        return sign;
    };
    let rest = whole & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let rounded = kept + u32::from(rest > halfway || (rest == halfway && (kept & 1) == 1));
    sign | rounded.min(0x7bff) as u16
}

fn f16_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => (f32::from(bits & 0x3ff) / 16_777_216.0).to_bits(),
        0x1f => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}
