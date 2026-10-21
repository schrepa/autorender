//! Water settling on the GPU for tiles' records (`settled_water`): terrains
//! hand the renderer what a tile's water settles from ([`SettleStart`]),
//! and it runs the shallow-water scheme ([`crate::water_kernels`]) over as
//! many tiles' squares at once as its buffers hold, each a slot stepping
//! its cells by their own seconds with open edges, from its coarsest level
//! up: each level runs until its water settles (or its steps run out),
//! looked at every [`LOOK_EVERY`] steps (its cells' water averaged over the
//! steps since the look before, summed workgroup by workgroup and read
//! back), and the next starts from its averaged water
//! ([`SettleStart::prolong`]); the finest's averaged water makes the
//! tile's record ([`SettleStart::finish`]), which the renderer hands its
//! host to keep and to the terrain that asked. While a session cooks it
//! takes more memory and more steps a frame.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use engine_core::RecordKey;
use engine_core::terrain_data::shallow_water::{ROUGHNESS, UNHELD};
use engine_core::terrain_data::{SettleStart, codec};

use crate::gpu_layouts::constants::{
    SIM_LOCAL, SIM_LOOKING, SIM_OPEN, SIM_SLOTS, SIM_STEPPING, SIM_SUMMING,
};
use crate::gpu_layouts::water::SimSlot;
use crate::water_kernels::{
    Kernels, SlotBuffers, encode_look, encode_steps, slot_bytes, slot_faces_bytes,
};

/// Steps between a square's looks at its water, which its frames' steps
/// divide.
pub(crate) const LOOK_EVERY: u32 = 256;
/// The steps a level runs at least and at most.
const LEVEL_STEPS: [u32; 2] = [1024, 16_384];
/// A level settled once, over [`STILL_LOOKS`] looks in a row, the water
/// its square holds changed by no more than [`SETTLED_VOLUME`] of it a
/// look, and its flowing cells' averaged velocities moved by
/// [`SETTLED_SPEED`] meters a second or less on average, or by no less
/// than [`STILL_SHARE`] of what they moved [`STILL_LOOKS`] looks before:
/// water running unsteadily (down rough steep rock) keeps varying, and its
/// averages stop settling further.
const STILL_LOOKS: usize = 3;
const SETTLED_VOLUME: f64 = 1.0e-3;
const SETTLED_SPEED: f64 = 0.01;
const STILL_SHARE: f64 = 0.75;
/// Bytes of the GPU's memory settling takes at most, while a session cooks
/// and else.
const COOK_BYTES: u64 = 1 << 30;
const PLAY_BYTES: u64 = 128 << 20;
/// Cell steps a frame runs at most, while a session cooks and else: its
/// steps are the most a power of two below them, at least one.
const COOK_CELL_STEPS: u64 = 40_000_000;
const PLAY_CELL_STEPS: u64 = 2_000_000;

/// Water settling: the scheme's passes and buffers; the starts waiting for
/// a slot, each with its record's key; the slots' squares; what is read
/// back; the records made since its host last took them (or why one
/// failed); whether its session cooks; and this frame's steps, slots'
/// squares and copies to read back.
pub(crate) struct WaterSettle {
    kernels: Kernels,
    buffers: Option<SlotBuffers>,
    waiting: VecDeque<(RecordKey, Arc<SettleStart>)>,
    squares: Vec<Option<Square>>,
    readbacks: Vec<Readback>,
    made: Vec<(RecordKey, Result<Vec<u8>, String>)>,
    cooking: bool,
    steps: u32,
    sourced: u32,
    copies: Vec<CopyBack>,
}

/// A tile's square settling in a slot: its record's key and start, the
/// level it settles at and that level's count (a level's loads and looks
/// carry it), its steps at the level, all told and since its last look,
/// what its looks found, where it stands, and the water its level starts
/// from, until written into its slot.
struct Square {
    record: RecordKey,
    start: Arc<SettleStart>,
    level: usize,
    loads: u32,
    steps: u32,
    total: u32,
    since_look: u32,
    looks: Vec<Look>,
    phase: Phase,
    load: Option<Vec<[f32; 4]>>,
}

/// Where a square stands: stepping, or its level settled and its averaged
/// water reading back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Stepping,
    Reading,
}

/// What a look found: the water its square held (meters of depth summed
/// over its cells), the mean of how far its flowing cells' averaged
/// velocities moved since the look before, and whether its water held
/// still since then.
#[derive(Clone, Copy)]
struct Look {
    volume: f64,
    moved: f64,
    still: bool,
}

/// A copy this frame reads back: from the averages at `offset` (bytes),
/// `bytes` of them, into its read back.
struct CopyBack {
    offset: u64,
    bytes: u64,
    readback: usize,
}

/// A read back: the slot, level load and kind it reads (a look's sums, or
/// the level's averaged water), and its buffer and map.
struct Readback {
    slot: usize,
    loads: u32,
    means: bool,
    buffer: wgpu::Buffer,
    mapped: Rc<RefCell<Mapped>>,
}

enum Mapped {
    Recorded,
    Mapping,
    Read(Vec<u8>),
    Failed,
}

impl WaterSettle {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            buffers: None,
            waiting: VecDeque::new(),
            squares: Vec::new(),
            readbacks: Vec::new(),
            made: Vec::new(),
            cooking: false,
            steps: 0,
            sourced: 0,
            copies: Vec::new(),
        }
    }

    /// Settles each of `starts` for its record, those already asked for
    /// left as they stand.
    pub(crate) fn settle(&mut self, starts: Vec<(RecordKey, Arc<SettleStart>)>) {
        for (record, start) in starts {
            let asked = self.waiting.iter().any(|(held, _)| *held == record)
                || self
                    .squares
                    .iter()
                    .flatten()
                    .any(|square| square.record == record);
            if !asked {
                self.waiting.push_back((record, start));
            }
        }
    }

    /// Whether its session cooks, which takes more memory and steps.
    pub(crate) fn set_cooking(&mut self, cooking: bool) {
        self.cooking = cooking;
    }

    /// The records it made since the last call, or why one failed.
    pub(crate) fn take_made(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        std::mem::take(&mut self.made)
    }

    /// Tiles waiting or settling.
    pub(crate) fn left(&self) -> u64 {
        (self.waiting.len() + self.squares.iter().flatten().count()) as u64
    }

    /// Bytes of its buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, SlotBuffers::bytes)
    }

    /// Prepares this frame's settling: takes what read back (deciding the
    /// looked squares' levels, starting the next level or finishing the
    /// tile), its buffers for the squares to come, the waiting starts into
    /// free slots, each square's water to start from, the steps this frame
    /// runs and which squares look at its end, and their slots' squares.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.copies.clear();
        self.steps = 0;
        let wanted = self.take_readbacks();
        let idle = self.waiting.is_empty() && self.squares.iter().all(Option::is_none);
        if idle {
            if self.readbacks.is_empty() {
                self.buffers = None;
                self.squares.clear();
            }
            return;
        }
        self.kernels.pipelines(device);
        self.ensure_buffers(device);
        let Some(capacity) = self.buffers.as_ref().map(|buffers| buffers.capacity) else {
            return;
        };
        // Waiting starts fill free slots, those wider than the slots hold
        // waiting until every slot is free and the buffers grow.
        for slot in 0..self.squares.len() {
            if self.squares[slot].is_some() {
                continue;
            }
            let Some(at) = self
                .waiting
                .iter()
                .position(|(_, start)| start.finest().side <= capacity)
            else {
                break;
            };
            let Some((record, start)) = self.waiting.remove(at) else {
                break;
            };
            let first = &start.levels[0];
            let water: Vec<[f32; 4]> = first
                .water
                .iter()
                .map(|&[surface, x, z]| [surface, x, z, 0.0])
                .collect();
            self.squares[slot] = Some(Square {
                record,
                start,
                level: 0,
                loads: 0,
                steps: 0,
                total: 0,
                since_look: 0,
                looks: Vec::new(),
                phase: Phase::Stepping,
                load: Some(water),
            });
        }
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        for (slot, square) in self.squares.iter_mut().enumerate() {
            if let Some(square) = square
                && let Some(water) = square.load.take()
            {
                load(queue, buffers, slot as u32, square, &water);
            }
        }
        let stepping: u64 = self
            .squares
            .iter()
            .flatten()
            .filter(|square| square.phase == Phase::Stepping)
            .map(|square| {
                let side = u64::from(square.start.levels[square.level].side);
                side * side
            })
            .sum();
        if stepping > 0 {
            let budget = if self.cooking {
                COOK_CELL_STEPS
            } else {
                PLAY_CELL_STEPS
            };
            let most = (budget / stepping).clamp(1, u64::from(LOOK_EVERY)) as u32;
            self.steps = 1 << (u32::BITS - 1 - most.leading_zeros());
        }
        let steps = self.steps;
        let mut squares = vec![
            SimSlot {
                shape: [0; 4],
                step: [0.0; 4],
            };
            self.squares.len()
        ];
        self.sourced = 0;
        let mut looks = Vec::new();
        for (slot, held) in self.squares.iter_mut().enumerate() {
            let Some(square) = held else {
                continue;
            };
            let level = &square.start.levels[square.level];
            let mut flags = SIM_LOCAL | SIM_OPEN | SIM_SUMMING;
            let mut summed = 0.0;
            if square.phase == Phase::Stepping && steps > 0 {
                flags |= SIM_STEPPING;
                square.steps += steps;
                square.total += steps;
                square.since_look += steps;
                self.sourced = self.sourced.max(level.sources.len() as u32);
                if square.since_look >= LOOK_EVERY {
                    flags |= SIM_LOOKING;
                    summed = square.since_look as f32;
                    square.since_look = 0;
                    looks.push(slot);
                }
            }
            squares[slot] = SimSlot {
                shape: [level.side, flags, level.sources.len() as u32, 0],
                step: [level.cell, 0.0, ROUGHNESS, summed],
            };
        }
        buffers.write_frames(queue, [0, 0]);
        buffers.write_squares(queue, &squares);
        // Each look's sums, and the averaged water of each level settled,
        // read back at the frame's end.
        let groups = u64::from(buffers.groups_along());
        let sums_at = 2 * u64::from(buffers.capacity) * u64::from(buffers.capacity);
        let mut copies = Vec::new();
        for slot in looks {
            let loads = self.squares[slot].as_ref().map_or(0, |square| square.loads);
            let offset = (buffers.averages_at(slot as u32) + sums_at) * 16;
            copies.push(((slot, loads, false), (offset, groups * groups * 16)));
        }
        for slot in wanted {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let side = u64::from(square.start.levels[square.level].side);
            let offset = (buffers.averages_at(slot as u32) + side * side) * 16;
            copies.push(((slot, square.loads, true), (offset, side * side * 16)));
        }
        for (read, range) in copies {
            self.read_back(device, read, range);
        }
    }

    /// Sets up a read back of `bytes` of the averages from `offset` for
    /// `slot`'s level load `loads`, of its averaged water where `means`,
    /// else of a look's sums.
    fn read_back(
        &mut self,
        device: &wgpu::Device,
        (slot, loads, means): (usize, u32, bool),
        (offset, bytes): (u64, u64),
    ) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/water-settle-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.copies.push(CopyBack {
            offset,
            bytes,
            readback: self.readbacks.len(),
        });
        self.readbacks.push(Readback {
            slot,
            loads,
            means,
            buffer,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    /// Its buffers for the squares waiting and settling: as wide as the
    /// widest of them, as many slots as its memory holds, made anew where
    /// wider are wanted and no square settles.
    fn ensure_buffers(&mut self, device: &wgpu::Device) {
        let widest = self
            .waiting
            .iter()
            .map(|(_, start)| start.finest().side)
            .chain(
                self.squares
                    .iter()
                    .flatten()
                    .map(|square| square.start.finest().side),
            )
            .max()
            .unwrap_or(1);
        let settling = self.squares.iter().any(Option::is_some) || !self.readbacks.is_empty();
        let fits = self
            .buffers
            .as_ref()
            .is_some_and(|buffers| buffers.capacity >= widest);
        if fits || (settling && self.buffers.is_some()) {
            return;
        }
        let budget = if self.cooking { COOK_BYTES } else { PLAY_BYTES };
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let slots = (budget / slot_bytes(widest))
            .min(binding / slot_faces_bytes(widest))
            .clamp(1, SIM_SLOTS as u64) as u32;
        self.buffers = Some(SlotBuffers::new(
            device,
            &self.kernels.layout,
            (widest, slots),
            false,
        ));
        self.squares = (0..slots).map(|_| None).collect();
    }

    /// Takes what read back: a look's sums decide its square's level,
    /// settled or out of steps, whose averaged water then reads back (the
    /// slots returned); a level's averaged water starts the next level, or
    /// finishes the tile's record. A read back of an earlier load of its
    /// slot is dropped; one that failed fails its tile.
    fn take_readbacks(&mut self) -> Vec<usize> {
        let mut kept = Vec::with_capacity(self.readbacks.len());
        let mut wanted = Vec::new();
        for readback in std::mem::take(&mut self.readbacks) {
            let read = match &*readback.mapped.borrow() {
                Mapped::Read(bytes) => Some(Ok(bytes.clone())),
                Mapped::Failed => Some(Err(())),
                Mapped::Recorded | Mapped::Mapping => None,
            };
            let Some(read) = read else {
                kept.push(readback);
                continue;
            };
            let Some(square) = self
                .squares
                .get_mut(readback.slot)
                .and_then(Option::as_mut)
                .filter(|square| square.loads == readback.loads)
            else {
                continue;
            };
            let Ok(bytes) = read else {
                let record = square.record;
                self.made
                    .push((record, Err("settled water did not read back".into())));
                self.squares[readback.slot] = None;
                continue;
            };
            let values = floats(&bytes);
            if readback.means {
                self.next_level(readback.slot, &values);
            } else if square.phase == Phase::Stepping && looked(square, &values) {
                square.phase = Phase::Reading;
                wanted.push(readback.slot);
            }
        }
        self.readbacks = kept;
        wanted
    }

    /// Slot `slot`'s level settled to `means` (each cell's surface, flow
    /// along x and z and foam, averaged over its last steps, row by row):
    /// the next level starts from them, or, its last, they make its tile's
    /// record, the slot then free.
    fn next_level(&mut self, slot: usize, means: &[[f32; 4]]) {
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        let side = square.start.levels[square.level].side as usize;
        let means = &means[..means.len().min(side * side)];
        if square.level + 1 < square.start.levels.len() {
            let water = square.start.prolong(square.level, means);
            square.level += 1;
            square.loads += 1;
            square.steps = 0;
            square.since_look = 0;
            square.looks.clear();
            square.phase = Phase::Stepping;
            square.load = Some(water);
            return;
        }
        let settled = square.steps < LEVEL_STEPS[1];
        let water = square.start.finish(means, square.total, settled);
        self.made.push((square.record, Ok(codec::encode(&water))));
        self.squares[slot] = None;
    }

    /// Encodes this frame's settling into `encoder`: its steps over every
    /// slot, the looks at the frame's end, and the copies read back.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipelines), Some(buffers)) = (self.kernels.created(), &self.buffers) else {
            return;
        };
        if self.steps > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-settle"),
                timestamp_writes: None,
            });
            encode_steps(&mut pass, pipelines, buffers, (self.steps, self.sourced, true));
            encode_look(&mut pass, pipelines, buffers);
        }
        for copy in &self.copies {
            let readback = &self.readbacks[copy.readback];
            encoder.copy_buffer_to_buffer(
                &buffers.averages,
                copy.offset,
                &readback.buffer,
                0,
                copy.bytes,
            );
        }
    }

    /// Maps the read backs the frame just submitted, whose bytes come with
    /// a later frame.
    pub(crate) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if !matches!(*readback.mapped.borrow(), Mapped::Recorded) {
                continue;
            }
            *readback.mapped.borrow_mut() = Mapped::Mapping;
            let buffer = readback.buffer.clone();
            let mapped = Rc::clone(&readback.mapped);
            readback.buffer.slice(..).map_async(
                wgpu::MapMode::Read,
                crate::platform::map_callback(move |result| {
                    let read = if result.is_ok() {
                        let bytes = buffer
                            .slice(..)
                            .get_mapped_range()
                            .ok()
                            .map(|view| view.to_vec());
                        buffer.unmap();
                        bytes
                    } else {
                        None
                    };
                    *mapped.borrow_mut() = read.map_or(Mapped::Failed, Mapped::Read);
                }),
            );
        }
    }
}

/// Writes `square`'s level's ground, its water `water` to start from
/// (each cell's surface, flow along x and z and foam) with its lakes'
/// levels, its rivers' sources and its sums starting anew into slot
/// `slot` of `buffers`.
fn load(queue: &wgpu::Queue, buffers: &SlotBuffers, slot: u32, square: &Square, water: &[[f32; 4]]) {
    let level = &square.start.levels[square.level];
    let n = level.side as usize;
    let cells: Vec<[f32; 4]> = (0..n * n)
        .map(|at| {
            let [surface, x, z, _] = water.get(at).copied().unwrap_or_default();
            let held = level.held[at];
            let held = if held.is_nan() { UNHELD } else { held };
            [surface, x, z, held]
        })
        .collect();
    let foam: Vec<f32> = (0..n * n)
        .map(|at| water.get(at).map_or(0.0, |cell| cell[3]))
        .collect();
    let at = buffers.cells_at(slot);
    let both = [cells.as_slice(), cells.as_slice()].concat();
    queue.write_buffer(&buffers.cells, at * 16, bytemuck::cast_slice(&both));
    let foams = [foam.as_slice(), foam.as_slice()].concat();
    queue.write_buffer(&buffers.foam, at * 4, bytemuck::cast_slice(&foams));
    queue.write_buffer(
        &buffers.ground,
        buffers.ground_at(slot) * 4,
        bytemuck::cast_slice(&level.corners),
    );
    let sources: Vec<[f32; 4]> = level
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
        queue.write_buffer(
            &buffers.inputs,
            buffers.sources_at(slot) * 16,
            bytemuck::cast_slice(&sources),
        );
    }
    // Sums start anew; the means before the first look are where the
    // level starts.
    let mut averages = vec![[0.0_f32; 4]; 2 * n * n];
    for (at, cell) in cells.iter().enumerate() {
        averages[n * n + at] = [cell[0], cell[1], cell[2], foam[at]];
    }
    queue.write_buffer(
        &buffers.averages,
        buffers.averages_at(slot) * 16,
        bytemuck::cast_slice(&averages),
    );
}

/// Takes look `sums` (each workgroup's: the water its cells held, how far
/// their averaged velocities moved, its cells flowing and stepping) into
/// `square`'s looks: whether its level is done, settled or out of steps.
fn looked(square: &mut Square, sums: &[[f32; 4]]) -> bool {
    let (mut volume, mut moved, mut flowing) = (0.0_f64, 0.0_f64, 0.0_f64);
    for sum in sums {
        volume += f64::from(sum[0]);
        moved += f64::from(sum[1]);
        flowing += f64::from(sum[2]);
    }
    let moved = moved / at_least_one(flowing);
    let still = square
        .looks
        .last()
        .is_some_and(|last| (volume - last.volume).abs() <= SETTLED_VOLUME * at_least_one(volume));
    square.looks.push(Look {
        volume,
        moved,
        still,
    });
    let looks = &square.looks;
    let count = looks.len();
    let settled = square.steps >= LEVEL_STEPS[0]
        && count > STILL_LOOKS
        && looks[count - STILL_LOOKS..].iter().all(|look| look.still)
        && (moved <= SETTLED_SPEED || moved >= STILL_SHARE * looks[count - 1 - STILL_LOOKS].moved);
    settled || square.steps >= LEVEL_STEPS[1]
}

/// `value`, or one where it is less.
fn at_least_one(value: f64) -> f64 {
    if value > 1.0 { value } else { 1.0 }
}

/// `bytes` read back as groups of four floats.
fn floats(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .chunks_exact(16)
        .map(|chunk| {
            let float = |at: usize| {
                f32::from_le_bytes([chunk[at], chunk[at + 1], chunk[at + 2], chunk[at + 3]])
            };
            [float(0), float(4), float(8), float(12)]
        })
        .collect()
}
