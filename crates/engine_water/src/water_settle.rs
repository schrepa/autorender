use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bytemuck::Zeroable;
use engine_core::{CookMode, DeviceLimits, Error, FrameTime, Jobs, Readback};
use engine_places::ground::GroundTiles;
use engine_render::{AfterRead, ComputePipeline, Count, Encode, Install, Pass, Prepare};

use crate::gpu::{
    SIM_BLOCK_LOOKS, SIM_BURSTING, SIM_CELL_SUMS, SIM_KEEPING, SIM_LOCAL, SIM_LOOKING, SIM_OPEN,
    SIM_SLOTS, SIM_SPARSE, SIM_STEPPING, SIM_SUMMING, SimSetup, SimSlot,
};
use crate::jobs::SETTLED;
use crate::shallow_water::{CELL_MEANS, DRY, step_seconds};
use crate::state::WaterState;
use crate::water::WaterGpu;
use crate::water_kernels::{
    Capacity, Kernels, Layout, SetupHandles, SlotHandles, grid, slot_bytes, slot_largest_bytes,
    squares,
};
use crate::{BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, SettleStart, SteadyWater};

const LOOK_EVERY: u32 = 256;
const SETTLE_STEPS: [u32; 2] = [1024, 16_384];
const STIR_STEPS: [u32; 2] = [1024, SETTLE_STEPS[1] / 2];
const SETTLE_LOOKS: u64 = ((SETTLE_STEPS[1] + STIR_STEPS[1]) / LOOK_EVERY) as u64;
const STILL_LOOKS: usize = 3;
const STILL_CHANGE: f64 = 0.01;
const COOK_BYTES: u64 = 1 << 30;
const PLAY_BYTES: u64 = 128 << 20;
const COOK_CELL_STEPS: u64 = 160_000_000;
const PLAY_CELL_STEPS: u64 = 2_000_000;
const LOST_FRAMES: u64 = 120;
pub(crate) const TALLIES: u32 = 0;
const LOOK: u32 = 0;
const MEANS: u32 = 1;
const GROUND: u32 = 2;

fn tag(slot: usize, kind: u32) -> u32 {
    1 + 3 * slot as u32 + kind
}

pub(crate) struct SettleAsk {
    pub(crate) terrain: Entity,
    pub(crate) inputs: Vec<Arc<[u8]>>,
    pub(crate) start: Arc<SettleStart>,
}

struct Square {
    inputs: Vec<Arc<[u8]>>,
    terrain: Entity,
    start: Arc<SettleStart>,
    load: u32,
    raised: bool,
    steps: u32,
    since_look: u32,
    stirred: u32,
    looks: Vec<Look>,
    settled: bool,
    stepping: u64,
    flowing: u64,
    phase: Phase,
    asked_means: Option<u64>,
    means: Option<Vec<u8>>,
    ground: Option<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Stepping,
    Stirring,
    Reading,
}

#[derive(Clone, Copy)]
struct Look {
    volume: f64,
    speed: f64,
    energy: f64,
    heave: f64,
    swirl: f64,
    clamped: f64,
    still: bool,
    against: f64,
    against_cells: f64,
}

impl Look {
    fn holds_to(&self, before: &Look) -> bool {
        let floor = f64::from(engine_codec::HEIGHT_STEP);
        let pairs = [
            (self.volume, before.volume),
            (self.speed, before.speed),
            (self.energy, before.energy),
            (self.heave, before.heave),
            (self.swirl, before.swirl),
        ];
        pairs.iter().all(|&(now, then)| {
            let change = (now - then).abs();
            change < STILL_CHANGE * then.abs() || change <= floor
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    Averages,
    Ground,
}

pub(crate) struct Read {
    pub(crate) tag: u32,
    pub(crate) source: Source,
    pub(crate) range: Range<u64>,
}

pub(crate) struct Raise {
    pub(crate) list: Vec<[u32; 4]>,
    pub(crate) frame: SimSetup,
    pub(crate) blocks: u32,
}

#[derive(Default)]
pub(crate) struct SettleWork {
    pub(crate) layout: Option<(Layout, u64)>,
    pub(crate) loads: Vec<Option<(u32, Arc<SettleStart>)>>,
    pub(crate) room: Option<(usize, u64)>,
    pub(crate) raise: Option<Raise>,
    pub(crate) squares: Vec<SimSlot>,
    pub(crate) steps: u32,
    pub(crate) sourced: u32,
    pub(crate) blocks: u32,
    pub(crate) stirring: bool,
    pub(crate) reads: Vec<Read>,
}

pub(crate) type SettleMade = (Vec<u8>, Result<Vec<u8>, String>);

#[derive(Resource, Default)]
pub struct SettleFeedback {
    waiting: VecDeque<SettleAsk>,
    squares: Vec<Option<Square>>,
    loads: u32,
    layout: Option<Layout>,
    layout_generation: u64,
    room: Option<usize>,
    room_generation: u64,
    steps: u32,
    asked: BTreeMap<(u32, u64), (usize, u32)>,
    arrivals: Vec<(u64, u32, Vec<u8>)>,
    made: Vec<SettleMade>,
    pub(crate) work: SettleWork,
}

impl Readback for SettleFeedback {
    fn arrived(&mut self, frame: u64, tag: u32, bytes: &[u8]) {
        self.arrivals.push((frame, tag, bytes.to_vec()));
    }
}

fn needs(start: &SettleStart) -> Capacity {
    Capacity::of(&start.stored, start.sources.len())
}

impl SettleFeedback {
    pub(crate) fn restart(&mut self) {
        for square in std::mem::take(&mut self.squares)
            .into_iter()
            .flatten()
            .rev()
        {
            self.waiting.push_front(SettleAsk {
                terrain: square.terrain,
                inputs: square.inputs,
                start: square.start,
            });
        }
        self.layout = None;
        self.room = None;
        self.steps = 0;
        self.asked.clear();
        self.arrivals.clear();
    }

    fn settle(&mut self, starts: Vec<SettleAsk>) {
        for ask in starts {
            let asked = self.waiting.iter().any(|held| held.inputs == ask.inputs)
                || self
                    .squares
                    .iter()
                    .flatten()
                    .any(|square| square.inputs == ask.inputs);
            if asked {
                continue;
            }
            if ask.start.stored.stored() == 0 {
                let made = ask.start.finish(&[], &[], 0, true);
                self.made.extend(
                    ask.inputs
                        .into_iter()
                        .zip(made)
                        .map(|(input, water)| (input.to_vec(), Ok(engine_codec::encode(&water)))),
                );
                continue;
            }
            self.waiting.push_back(ask);
        }
    }

    pub(crate) fn looks_left(&self) -> u64 {
        let waiting = self.waiting.len() as u64 * SETTLE_LOOKS;
        let settling = self.squares.iter().flatten().map(|square| {
            let taken = u64::from((square.steps + square.stirred) / LOOK_EVERY);
            SETTLE_LOOKS.saturating_sub(taken)
        });
        waiting + settling.sum::<u64>()
    }

    fn reading(&self) -> bool {
        !self.asked.is_empty()
    }

    fn prepare(
        &mut self,
        (cooking, binding, frame): (bool, u64, u64),
        ground: &dyn Fn(Entity) -> Option<u32>,
    ) {
        self.work = SettleWork::default();
        let ran = std::mem::take(&mut self.steps);
        let wanted = self.take_readbacks(ran, cooking, frame);
        let idle = self.waiting.is_empty() && self.squares.iter().all(Option::is_none);
        if idle {
            if !self.reading() {
                self.layout = None;
                self.room = None;
                self.squares.clear();
            }
            return;
        }
        self.ensure_buffers(binding, cooking);
        self.fill_slots();
        let Some(layout) = self.layout else {
            return;
        };
        self.work.layout = Some((layout, self.layout_generation));
        self.raise(&layout, ground);
        self.work.room = self.room.map(|room| (room, self.room_generation));
        self.work.loads = self
            .squares
            .iter()
            .map(|square| {
                square
                    .as_ref()
                    .map(|square| (square.load, Arc::clone(&square.start)))
            })
            .collect();
        let stepping: u64 = self
            .squares
            .iter()
            .flatten()
            .filter(|square| square.raised && square.phase != Phase::Reading)
            .map(|square| square.stepping.max(1))
            .sum();
        let budget = if cooking {
            COOK_CELL_STEPS
        } else {
            PLAY_CELL_STEPS
        };
        if let Some(most) = budget.checked_div(stepping) {
            let most = most.clamp(1, u64::from(LOOK_EVERY)) as u32;
            self.steps = 1 << (u32::BITS - 1 - most.leading_zeros());
        }
        let looks = self.write_squares();
        self.ask_readbacks(&layout, (looks, wanted), frame);
    }

    fn raise(&mut self, layout: &Layout, ground: &dyn Fn(Entity) -> Option<u32>) {
        let found = self.squares.iter().enumerate().find_map(|(slot, square)| {
            let square = square.as_ref().filter(|square| !square.raised)?;
            Some((slot, ground(square.terrain)?))
        });
        let Some((slot, terrain)) = found else {
            return;
        };
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        let start = &square.start;
        let blocks = start.stored.stored();
        let window = (start.window * start.window) as usize;
        let entries = 2 * blocks + blocks * window;
        if self.room.is_none_or(|room| room < entries) {
            self.room = Some(entries.max(1).next_power_of_two());
            self.room_generation += 1;
        }
        let side = i64::from(BLOCK_SIDE);
        let first = (slot as u32) * layout.capacity.blocks;
        let mut list: Vec<[u32; 4]> = Vec::with_capacity(entries);
        let blocks_and_samples = start.stored.places.iter().zip(&start.windows);
        for (block, (&place, sample)) in blocks_and_samples.enumerate() {
            let corner = [
                start.first[0] + i64::from(place & 0xffff) * side,
                start.first[1] + i64::from(place >> 16) * side,
            ];
            let bits = |value: i64| value as i32 as u32;
            list.push([
                first + block as u32,
                bits(corner[0]),
                bits(corner[1]),
                (2 * blocks + block * window) as u32,
            ]);
            list.push([bits(sample[0]), bits(sample[1]), 0, 0]);
        }
        list.extend_from_slice(&start.samples);
        let frame = SimSetup {
            counts: [blocks as u32, 0, 0, 0],
            square: [start.window, start.last[0] as u32, start.last[1] as u32, 0],
            meters: [start.cell, start.datum, 0.0, 0.0],
            ring: [0; 4],
            tiles: [0; 4],
            terrain: [terrain, 0, 0, 0],
        };
        square.raised = true;
        self.work.raise = Some(Raise {
            list,
            frame,
            blocks: blocks as u32,
        });
    }

    fn fill_slots(&mut self) {
        let Some(capacity) = self.layout.map(|layout| layout.capacity) else {
            return;
        };
        for slot in 0..self.squares.len() {
            if self.squares[slot].is_some() {
                continue;
            }
            let fits = |start: &SettleStart| capacity.holds(needs(start));
            let Some(at) = self.waiting.iter().position(|ask| fits(ask.start.as_ref())) else {
                break;
            };
            let Some(ask) = self.waiting.remove(at) else {
                break;
            };
            self.loads += 1;
            let steps = stepping(&ask.start);
            self.squares[slot] = Some(Square {
                inputs: ask.inputs,
                terrain: ask.terrain,
                start: ask.start,
                load: self.loads,
                raised: false,
                steps: 0,
                since_look: 0,
                stirred: 0,
                looks: Vec::new(),
                settled: false,
                stepping: steps,
                flowing: 0,
                phase: Phase::Stepping,
                asked_means: None,
                means: None,
                ground: None,
            });
        }
    }

    fn write_squares(&mut self) -> Vec<usize> {
        let steps = self.steps;
        let (mut blocks, mut sourced, mut stirring) = (0, 0, false);
        let mut written = vec![SimSlot::zeroed(); self.squares.len()];
        let mut looks = Vec::new();
        for (slot, held) in self.squares.iter_mut().enumerate() {
            let Some(square) = held else {
                continue;
            };
            let start = &square.start;
            let stored = &start.stored;
            let stirs = square.phase == Phase::Stirring;
            let mut flags = SIM_OPEN | SIM_SUMMING | SIM_SPARSE;
            flags |= if stirs {
                SIM_BURSTING | SIM_KEEPING
            } else {
                SIM_LOCAL
            };
            let index = square.stirred;
            let runs = square.raised && square.phase != Phase::Reading && steps > 0;
            if runs {
                flags |= SIM_STEPPING;
                square.since_look += steps;
                if stirs {
                    square.stirred += steps;
                    stirring = true;
                } else {
                    square.steps += steps;
                }
                sourced = sourced.max(start.sources.len() as u32);
                if square.since_look >= LOOK_EVERY {
                    flags |= SIM_LOOKING;
                    square.since_look = 0;
                    looks.push(slot);
                }
            }
            blocks = blocks.max(stored.stored() as u32);
            let [x, z] = stored.cells;
            let [first_x, first_z] = start.first.map(|first| first as i32 as u32);
            written[slot] = SimSlot {
                shape: [x, z, flags, start.sources.len() as u32],
                step: [start.cell, step_seconds(start.cell), 0.0, 0.0],
                blocks: [
                    stored.blocks[0],
                    stored.blocks[1],
                    stored.stored() as u32,
                    0,
                ],
                place: [first_x, first_z, index, 0],
            };
        }
        self.work.squares = written;
        self.work.steps = steps;
        self.work.sourced = sourced;
        self.work.blocks = blocks;
        self.work.stirring = stirring;
        looks
    }

    fn ask_readbacks(
        &mut self,
        layout: &Layout,
        (looks, wanted): (Vec<usize>, Vec<usize>),
        frame: u64,
    ) {
        let held = layout.held();
        let means = CELL_MEANS as u64;
        for slot in looks {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let blocks = square.start.stored.stored() as u64;
            let offset =
                (layout.averages_at(slot as u32) + (u64::from(SIM_CELL_SUMS) + means) * held) * 16;
            let bytes = u64::from(SIM_BLOCK_LOOKS) * blocks * 16;
            self.asked
                .insert((tag(slot, LOOK), frame), (slot, square.load));
            self.work.reads.push(Read {
                tag: tag(slot, LOOK),
                source: Source::Averages,
                range: offset..offset + bytes,
            });
        }
        let lost: Vec<usize> = self
            .squares
            .iter()
            .enumerate()
            .filter_map(|(slot, square)| {
                let square = square.as_ref()?;
                let asked = square.asked_means?;
                (square.phase == Phase::Reading && frame >= asked + LOST_FRAMES).then_some(slot)
            })
            .collect();
        for slot in wanted.into_iter().chain(lost) {
            let Some(square) = self.squares[slot].as_mut() else {
                continue;
            };
            let cells = square.start.water.len() as u64;
            let blocks = square.start.stored.stored() as u64;
            let offset = (layout.averages_at(slot as u32) + u64::from(SIM_CELL_SUMS) * held) * 16;
            let ground = layout.ground_at(slot as u32) * 4;
            square.asked_means = Some(frame);
            square.means = None;
            square.ground = None;
            for kind in [MEANS, GROUND] {
                self.asked
                    .insert((tag(slot, kind), frame), (slot, square.load));
            }
            self.work.reads.push(Read {
                tag: tag(slot, MEANS),
                source: Source::Averages,
                range: offset..offset + means * cells * 16,
            });
            self.work.reads.push(Read {
                tag: tag(slot, GROUND),
                source: Source::Ground,
                range: ground..ground + blocks * BLOCK_GROUND as u64 * 4,
            });
        }
    }

    fn ensure_buffers(&mut self, binding: u64, cooking: bool) {
        let mut kept = VecDeque::with_capacity(self.waiting.len());
        for ask in std::mem::take(&mut self.waiting) {
            if slot_largest_bytes(needs(&ask.start)) <= binding {
                kept.push_back(ask);
                continue;
            }
            let [x, z] = ask.start.stored.cells;
            let error = format!(
                "settling water over {} blocks of {x} by {z} cells takes more than the {binding} \
                 bytes the GPU binds",
                ask.start.stored.stored()
            );
            self.made.extend(
                ask.inputs
                    .into_iter()
                    .map(|input| (input.to_vec(), Err(error.clone()))),
            );
        }
        self.waiting = kept;
        let largest = self
            .waiting
            .iter()
            .map(|ask| &ask.start)
            .chain(self.squares.iter().flatten().map(|square| &square.start))
            .fold(Capacity::default(), |most, start| most.max(needs(start)));
        let settling = self.squares.iter().any(Option::is_some) || self.reading();
        let fits = self
            .layout
            .is_some_and(|layout| layout.capacity.holds(largest));
        if fits || (settling && self.layout.is_some()) {
            return;
        }
        let budget = if cooking { COOK_BYTES } else { PLAY_BYTES };
        let slots = (budget / slot_bytes(largest).max(1))
            .min(binding / slot_largest_bytes(largest).max(1))
            .clamp(1, u64::from(SIM_SLOTS)) as u32;
        self.layout = Some(Layout {
            capacity: largest,
            slots,
            live: false,
        });
        self.layout_generation += 1;
        self.room = None;
        self.squares = (0..slots).map(|_| None).collect();
    }

    fn take_readbacks(&mut self, ran: u32, cooking: bool, frame: u64) -> Vec<usize> {
        let mut wanted = Vec::new();
        let mut arrivals = std::mem::take(&mut self.arrivals);
        arrivals.sort_by_key(|(asked, tag, _)| (*asked, *tag));
        for (asked, tag, bytes) in arrivals {
            let Some((slot, load)) = self.asked.remove(&(tag, asked)) else {
                continue;
            };
            let kind = (tag - 1) % 3;
            let Some(square) = self
                .squares
                .get_mut(slot)
                .and_then(Option::as_mut)
                .filter(|square| square.load == load)
            else {
                continue;
            };
            if kind != LOOK {
                if kind == MEANS {
                    square.means = Some(bytes);
                } else {
                    square.ground = Some(bytes);
                }
                if square.means.is_some() && square.ground.is_some() {
                    self.finish(slot, cooking);
                }
                continue;
            }
            let values = floats(&bytes);
            drop(bytes);
            if square.phase == Phase::Reading {
                continue;
            }
            let done = looked(square, &values);
            if cooking && (done || square.looks.len() % 16 == 1) {
                say(square, ran, done);
            }
            if !done {
                continue;
            }
            if square.phase == Phase::Stepping {
                square.settled = square.steps < SETTLE_STEPS[1];
                square.phase = Phase::Stirring;
                square.looks.clear();
                square.since_look = 0;
            } else {
                square.phase = Phase::Reading;
                wanted.push(slot);
            }
        }
        self.asked
            .retain(|&(_, asked), _| frame < asked + LOST_FRAMES);
        wanted
    }

    fn finish(&mut self, slot: usize, cooking: bool) {
        let ready = self.squares[slot]
            .as_ref()
            .is_some_and(|square| square.means.is_some() && square.ground.is_some());
        if !ready {
            return;
        }
        let Some(square) = self.squares[slot].take() else {
            return;
        };
        let (Some(means), Some(ground)) = (square.means.as_deref(), square.ground.as_deref())
        else {
            return;
        };
        let start = &square.start;
        let split = (start.water.len() * CELL_MEANS * 16).min(means.len());
        let means = floats(&means[..split]);
        let ground: Vec<f32> = ground
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let settled = square.settled && square.stirred < STIR_STEPS[1];
        let made = start.finish(&means, &ground, square.steps + square.stirred, settled);
        if cooking {
            say_made(&square, &made);
        }
        for (input, water) in square.inputs.iter().zip(&made) {
            self.made
                .push((input.to_vec(), Ok(engine_codec::encode(water))));
        }
    }
}

fn blocks(start: &SettleStart) -> (Vec<[f32; 2]>, u64) {
    let stored = &start.stored;
    let mut marked = vec![false; stored.stored()];
    for (at, cell) in start.water.iter().enumerate() {
        if start.held(at).is_none() && cell[0] - start.bed(at) > DRY {
            marked[at / BLOCK_CELLS] = true;
        }
    }
    for source in &start.sources {
        if let Some(mark) = marked.get_mut(source.cell as usize / BLOCK_CELLS) {
            *mark = true;
        }
    }
    let mut entries = vec![[0.0_f32; 2]; 2 * marked.len()];
    for (block, &wet) in marked.iter().enumerate() {
        if wet {
            entries[2 * block] = [1.0, 0.0];
        }
    }
    let stepping = stored
        .places
        .iter()
        .filter(|&&place| {
            let (x, z) = (i64::from(place & 0xffff), i64::from(place >> 16));
            (-1..=1).any(|dz| {
                (-1..=1).any(|dx| {
                    stored
                        .slot(x + dx, z + dz)
                        .is_some_and(|beside| marked[beside])
                })
            })
        })
        .count();
    (entries, (stepping * BLOCK_CELLS) as u64)
}

fn looked(square: &mut Square, sums: &[[f32; 4]]) -> bool {
    let (mut volume, mut speed, mut flowing, mut stepping) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    let (mut energy, mut heave, mut swirl, mut clamped) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    let (mut against, mut against_cells) = (0.0_f64, 0.0_f64);
    let blocks = square.start.stored.stored();
    for block in sums
        .as_chunks::<{ SIM_BLOCK_LOOKS as usize }>()
        .0
        .iter()
        .take(blocks)
    {
        let [water, motion, edge] = [block[0], block[1], block[2]];
        volume += f64::from(water[0]);
        speed += f64::from(water[1]);
        flowing += f64::from(water[2]);
        stepping += f64::from(water[3]);
        energy += f64::from(motion[0]);
        heave += f64::from(motion[1]);
        swirl += f64::from(motion[2]);
        clamped += f64::from(motion[3]);
        against += f64::from(edge[0]);
        against_cells += f64::from(edge[1]);
    }
    square.stepping = stepping as u64;
    square.flowing = flowing as u64;
    let cells = at_least_one(flowing);
    let mut look = Look {
        volume,
        speed: speed / cells,
        energy: energy / cells,
        heave: heave / cells,
        swirl: swirl / cells,
        clamped,
        still: false,
        against,
        against_cells,
    };
    look.still = square.looks.last().is_some_and(|last| look.holds_to(last));
    square.looks.push(look);
    let looks = &square.looks;
    let count = looks.len();
    let still = count > STILL_LOOKS && looks[count - STILL_LOOKS..].iter().all(|look| look.still);
    let ([least, most], steps) = match square.phase {
        Phase::Stirring => (STIR_STEPS, square.stirred),
        Phase::Stepping | Phase::Reading => (SETTLE_STEPS, square.steps),
    };
    (steps >= least && still) || steps >= most
}

fn say(square: &Square, steps: u32, done: bool) {
    let start = &square.start;
    let [x, z] = start.stored.cells;
    let stirs = square.phase == Phase::Stirring;
    let (taken, most) = if stirs {
        (square.stirred, STIR_STEPS[1])
    } else {
        (square.steps, SETTLE_STEPS[1])
    };
    let state = match (stirs, done, taken < most) {
        (false, false, _) => "settling",
        (false, true, true) => "settled_stirring",
        (false, true, false) => "steps_spent_stirring",
        (true, false, _) => "stirring",
        (true, true, true) => "stirred",
        (true, true, false) => "stir_steps_spent",
    };
    let sources: f32 = start.sources.iter().map(|source| source.rate).sum();
    let area = f64::from(start.cell * start.cell);
    let look = square.looks.last().copied();
    let record = serde_json::json!({
        "event": "water_settling",
        "cells": [x, z],
        "cell_m": start.cell,
        "blocks_stored": start.stored.stored(),
        "tiles_kept": square.inputs.len(),
        "sources_m3_per_s": sources,
        "state": state,
        "steps": square.steps,
        "stirred_steps": square.stirred,
        "steps_per_frame": steps,
        "cells_stepping": square.stepping,
        "cells_flowing": square.flowing,
        "held_m3": look.map_or(0.0, |look| look.volume * area),
        "mean_speed_m_per_s": look.map_or(0.0, |look| look.speed),
        "mean_loss_w_per_m2": look.map_or(0.0, |look| look.energy),
        "mean_heave_m": look.map_or(0.0, |look| look.heave),
        "mean_swirl_per_s": look.map_or(0.0, |look| look.swirl),
        "cells_at_fastest": look.map_or(0.0, |look| look.clamped),
        "against_edges_m3": look.map_or(0.0, |look| look.against * area),
        "cells_against_edges": look.map_or(0.0, |look| look.against_cells),
    });
    eprintln!("{record}");
}

const REACH_NAMES: [&str; 5] = ["pools", "plane_beds", "steps", "cascades", "falls"];

fn say_made(square: &Square, made: &[SteadyWater]) {
    let [x, z] = square.start.stored.cells;
    let standing: usize = made.iter().map(|water| water.solids.len()).sum();
    let mut clamped = 0_u64;
    let mut reaches = REACH_NAMES.map(|_| (0.0_f64, 0.0_f64, 0_u64));
    for water in made {
        let stats = &water.stats;
        clamped += u64::from(stats.clamped);
        let counted = stats.friction.iter().zip(&stats.fast).zip(&stats.reaches);
        for (reach, ((&friction, &fast), &cells)) in reaches.iter_mut().zip(counted) {
            reach.0 += f64::from(friction) * f64::from(cells);
            reach.1 += f64::from(fast) * f64::from(cells);
            reach.2 += u64::from(cells);
        }
    }
    let channels: serde_json::Map<String, serde_json::Value> = REACH_NAMES
        .iter()
        .zip(reaches)
        .map(|(name, (friction, fast, cells))| {
            let count = at_least_one(cells as f64);
            let reach = serde_json::json!({
                "cells": cells,
                "mean_manning_n": friction / count,
                "share_above_froude_1": fast / count,
            });
            ((*name).to_owned(), reach)
        })
        .collect();
    let record = serde_json::json!({
        "event": "water_settled",
        "cells": [x, z],
        "tiles": made.len(),
        "cells_at_fastest": clamped,
        "solids_standing": standing,
        "solids": square.start.solids.len(),
        "channels": channels,
    });
    eprintln!("{record}");
}

fn at_least_one(value: f64) -> f64 {
    if value > 1.0 { value } else { 1.0 }
}

fn floats(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|chunk| {
            let float = |at: usize| {
                f32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]])
            };
            [float(0), float(4), float(8), float(12)]
        })
        .collect()
}

pub(crate) fn hand_settles(mut state: ResMut<WaterState>, mut feedback: ResMut<SettleFeedback>) {
    let asked = std::mem::take(&mut state.settling.asked);
    if !asked.is_empty() {
        feedback.settle(asked);
    }
}

pub(crate) fn plan_settles(
    ground: Res<GroundTiles>,
    (limits, cook, time): (Res<DeviceLimits>, Res<CookMode>, Res<FrameTime>),
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
    mut feedback: ResMut<SettleFeedback>,
) {
    let index_of = |entity: Entity| {
        ground
            .tables
            .iter()
            .find(|(terrain, _)| *terrain == entity)
            .map(|&(_, index)| index)
    };
    let feedback = &mut *feedback;
    feedback.prepare((cook.cooking, limits.storage_room, time.frame), &index_of);
    for (input, made) in std::mem::take(&mut feedback.made) {
        if let Ok(bytes) = &made {
            let kept = jobs.keep(&SETTLED, &input, bytes.clone());
            jobs.cancel(kept);
        }
        state.settling.settled.push((input, made));
    }
    state.looks_left = feedback.looks_left();
}

fn load(
    slots: &SlotHandles,
    layout: &Layout,
    slot: u32,
    start: &SettleStart,
    cx: &mut Prepare<'_, WaterGpu>,
) {
    let n = start.water.len() as u64;
    slots.write_table(layout, slot, &start.stored, cx);
    cx.write(slots.ground, layout.ground_at(slot) * 4, &start.ground);
    let at = layout.cells_at(slot);
    cx.write(slots.cells, at * 16, &start.water);
    cx.write(slots.cells, (at + layout.held()) * 16, &start.water);
    let sources: Vec<[f32; 4]> = start
        .sources
        .iter()
        .map(|source| {
            [
                f32::from_bits(source.cell),
                source.rate,
                source.way[0],
                source.way[1],
            ]
        })
        .collect();
    if !sources.is_empty() {
        cx.write(slots.inputs, layout.sources_at(slot) * 16, &sources);
    }
    let averages = layout.averages_at(slot);
    let (blocks, _) = blocks(start);
    let steps = layout.steps_at(slot);
    cx.write(slots.steps, (steps + layout.held()) * 8, &blocks);
    cx.write(
        slots.averages,
        averages * 16,
        &vec![[0.0_f32; 4]; (u64::from(SIM_CELL_SUMS) * n) as usize],
    );
    cx.write(slots.steps, steps * 8, &vec![[0.0_f32; 2]; n as usize]);
}

pub(crate) fn stepping(start: &SettleStart) -> u64 {
    blocks(start).1
}

pub(crate) struct SettleGpu {
    slots: SlotHandles,
    setup: SetupHandles,
    kernels: Kernels,
    raising: ComputePipeline<WaterGpu>,
    raise: Pass<WaterGpu>,
    pass: Pass<WaterGpu>,
    layout: Option<(Layout, u64)>,
    room: Option<(usize, u64)>,
    loaded: Vec<u32>,
    raised: u32,
    run: Option<(u32, u32, u32, bool, u32)>,
}

impl SettleGpu {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>) -> Result<Self, Error> {
        use crate::shaders::water::setup;
        let slots = SlotHandles::install(
            cx,
            [
                "water_settle_ground",
                "water_settle_cells",
                "water_settle_depths",
                "water_settle_fluxes",
                "water_settle_inputs",
                "water_settle_blocks",
                "water_settle_steps",
                "water_settle_averages",
                "water_settle_slots",
                "water_settle_first",
                "water_settle_second",
                "water_settle_touches",
            ],
        );
        let setup_handles =
            SetupHandles::install(cx, ["water_settle_setup", "water_settle_setup_list"]);
        let raise = cx
            .pass("water_settle_raise")
            .label("engine/water-settle-raise")
            .id();
        let pass = cx
            .pass("water_settle")
            .after(raise)
            .label("engine/water-settle")
            .id();
        let kernels = Kernels::install(cx, pass, &slots)?;
        let bound = crate::water::reached(cx, &[setup::SETUP_RAISE], setup_handles.bound(&slots));
        let raising = cx.compute_pipeline(raise, setup::SETUP_RAISE, &bound)?;
        Ok(Self {
            slots,
            setup: setup_handles,
            kernels,
            raising,
            raise,
            pass,
            layout: None,
            room: None,
            loaded: Vec::new(),
            raised: 0,
            run: None,
        })
    }

    pub(crate) fn prepare(&mut self, work: &SettleWork, cx: &mut Prepare<'_, WaterGpu>) {
        self.run = None;
        self.raised = 0;
        let Some((layout, generation)) = work.layout else {
            self.layout = None;
            self.room = None;
            self.slots.release(cx);
            self.setup.release(cx);
            self.loaded.clear();
            return;
        };
        let fresh = self.layout.is_none_or(|(_, held)| held != generation);
        self.slots.hold(layout, fresh, cx);
        if fresh {
            self.loaded = vec![0; layout.slots as usize];
        }
        self.layout = Some((layout, generation));
        for (slot, wanted) in work.loads.iter().enumerate() {
            let Some((count, start)) = wanted else {
                continue;
            };
            if self.loaded.get(slot) == Some(count) {
                continue;
            }
            load(&self.slots, &layout, slot as u32, start, cx);
            if let Some(held) = self.loaded.get_mut(slot) {
                *held = *count;
            }
        }
        match work.room {
            Some((room, generation)) => {
                let fresh = fresh || self.room != Some((room, generation));
                self.setup.hold((room, 0), fresh, cx);
                self.room = Some((room, generation));
            }
            None => {
                self.room = None;
                self.setup.release(cx);
            }
        }
        if let Some(raise) = &work.raise {
            cx.write(self.setup.list, 0, &raise.list);
            cx.write(self.setup.frame, 0, &[raise.frame]);
            self.raised = raise.blocks;
            cx.run(self.raise);
        }
        self.slots.write_frames(&layout, cx);
        cx.write(self.slots.squares, 0, &[squares(&work.squares)]);
        if work.steps > 0 {
            self.run = Some((
                work.steps,
                work.sourced,
                work.blocks,
                work.stirring,
                layout.slots,
            ));
            cx.run(self.pass);
        }
        for read in &work.reads {
            let buffer = match read.source {
                Source::Averages => self.slots.averages,
                Source::Ground => self.slots.ground,
            };
            cx.read_back::<SettleFeedback>(read.tag, buffer, read.range.clone(), AfterRead::Keep);
        }
    }

    pub(crate) fn forget(&mut self) {
        self.layout = None;
        self.room = None;
        self.loaded.clear();
    }

    pub(crate) fn encode(&mut self, pass: Pass<WaterGpu>, cx: &mut Encode<'_, WaterGpu>) -> bool {
        if pass == self.raise {
            if self.raised > 0 {
                let [groups_x, groups_z] = grid(self.raised);
                cx.dispatch(&self.raising, Count::Groups([groups_x, groups_z, 1]));
            }
            return true;
        }
        if pass != self.pass {
            return false;
        }
        let Some((steps, sourced, blocks, stirring, slots)) = self.run else {
            return true;
        };
        self.kernels
            .encode_steps(cx, (steps, sourced, true, stirring), (blocks, slots, None));
        self.kernels.encode_look(cx, (blocks, slots));
        true
    }
}
