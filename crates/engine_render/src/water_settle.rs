//! Water settling on the GPU for tiles' records (`settled_water`): terrains
//! hand the renderer what a region's water settles from ([`SettleStart`]),
//! the whole terrain's while a session cooks, and it runs the shallow-water
//! scheme ([`crate::water_kernels`]) over as many regions at once as its
//! buffers hold, each a slot holding the blocks of cells its water can
//! reach and stepping its cells by their own seconds with open edges, only
//! the blocks water can reach this step stepping, until its water settles
//! (or its steps run out), looked at every [`LOOK_EVERY`] steps (its cells'
//! water averaged over the seconds since the look before, with how much it
//! varied, summed block by block and read back); its averaged water then
//! makes the records of the tiles the region keeps
//! ([`SettleStart::finish`]), which the renderer hands its host to keep and
//! to the terrain that asked. While a session cooks it takes more memory and
//! more steps a frame, and says how its regions settle, how much water
//! stands against the edges of the blocks they store, and what their
//! records came to.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use engine_core::RecordKey;
use engine_core::terrain_data::shallow_water::{CELL_MEANS, DRY};
use engine_core::terrain_data::{BLOCK_CELLS, SettleStart, SteadyWater, codec};

use crate::gpu_layouts::constants::{
    SIM_BLOCK_LOOKS, SIM_CELL_SUMS, SIM_LOCAL, SIM_LOOKING, SIM_OPEN, SIM_SLOTS, SIM_SPARSE,
    SIM_STEPPING, SIM_SUMMING,
};
use crate::gpu_layouts::water::SimSlot;
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_look, encode_steps, slot_bytes, slot_largest_bytes,
};

/// Steps between a region's looks at its water, which its frames' steps
/// divide.
pub(crate) const LOOK_EVERY: u32 = 256;
/// The steps a region runs at least and at most, and its looks at most.
const SETTLE_STEPS: [u32; 2] = [1024, 16_384];
const SETTLE_LOOKS: u64 = (SETTLE_STEPS[1] / LOOK_EVERY) as u64;
/// A region settled once its means are stationary: over [`STILL_LOOKS`]
/// looks in a row, the water it holds and its flowing cells' mean speed,
/// energy lost, heave and swirl each changed from the look before by under
/// [`STILL_CHANGE`] of itself, or by no more than the step records keep
/// them at (`codec::HEIGHT_STEP` of their unit), which records cannot tell.
/// Water running unsteadily (down rough steep rock) keeps varying, and its
/// means and spreads settle.
const STILL_LOOKS: usize = 3;
const STILL_CHANGE: f64 = 0.01;
/// Bytes of the GPU's memory settling takes at most, while a session cooks
/// and else: a region larger takes a slot of its own regardless.
const COOK_BYTES: u64 = 1 << 30;
const PLAY_BYTES: u64 = 128 << 20;
/// Cell steps a frame runs at most, while a session cooks and else,
/// counting the cells whose blocks step: its steps are the most a power of
/// two below them, at least one. A cook's frames run long, its steps
/// outweighing the rest of a frame.
const COOK_CELL_STEPS: u64 = 160_000_000;
const PLAY_CELL_STEPS: u64 = 2_000_000;

/// Water settling: the scheme's passes and buffers; the starts waiting for
/// a slot, each with the keys of the records it makes; the slots' regions,
/// and the regions loaded so far; what is read back; the records made
/// since its host last took them (or why one failed); whether its session
/// cooks; and this frame's steps, the blocks its passes cover (those of the
/// region storing the most), the cells rivers bring water to its passes
/// cover, slots' rectangles, buffer ranges cleared and copies to read back.
pub(crate) struct WaterSettle {
    kernels: Kernels,
    buffers: Option<SlotBuffers>,
    waiting: VecDeque<(Vec<RecordKey>, Arc<SettleStart>)>,
    squares: Vec<Option<Square>>,
    loads: u32,
    readbacks: Vec<Readback>,
    made: Vec<(RecordKey, Result<Vec<u8>, String>)>,
    cooking: bool,
    steps: u32,
    blocks: u32,
    sourced: u32,
    clears: Vec<Clear>,
    copies: Vec<CopyBack>,
}

/// A region settling in a slot: the keys of its records and its start, its
/// load's count (its looks and its read backs carry it), its steps, all
/// told and since its last look, what its looks found, its cells stepping
/// and flowing as its last look found them (stepping as it started,
/// before), where it stands, and whether its start is still to be written
/// into its slot.
struct Square {
    records: Vec<RecordKey>,
    start: Arc<SettleStart>,
    load: u32,
    steps: u32,
    since_look: u32,
    looks: Vec<Look>,
    stepping: u64,
    flowing: u64,
    phase: Phase,
    fresh: bool,
}

/// Where a region stands: stepping, or settled and its averaged water
/// reading back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Stepping,
    Reading,
}

/// What a look found: the water its region held (meters of depth summed
/// over its cells); its flowing cells' mean speed (meters a second), energy
/// lost (watts a square meter), heave (meters) and swirl (a second); those
/// whose water ran at the fastest water runs; whether its means held still
/// since the look before; and the water held by its cells beside blocks it
/// does not store (meters of depth summed over them) and those cells.
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
    /// Whether its means changed from those of look `before` by under
    /// [`STILL_CHANGE`] of themselves, or by no more than records keep.
    fn holds_to(&self, before: &Look) -> bool {
        let floor = f64::from(codec::HEIGHT_STEP);
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

/// A range of a buffer this frame clears before it steps: of the averages
/// or the steps, from `offset`, `bytes` of it.
struct Clear {
    averages: bool,
    offset: u64,
    bytes: u64,
}

/// A copy this frame reads back: from the averages at `offset` (bytes),
/// `bytes` of them, into its read back.
struct CopyBack {
    offset: u64,
    bytes: u64,
    readback: usize,
}

/// A read back: the slot, region load and kind it reads (a look's sums, or
/// the region's averaged water), and its buffer and map.
struct Readback {
    slot: usize,
    load: u32,
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

/// What a slot settling `start` takes of the buffers.
fn needs(start: &SettleStart) -> Capacity {
    Capacity::of(&start.stored, start.sources.len())
}

impl WaterSettle {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            buffers: None,
            waiting: VecDeque::new(),
            squares: Vec::new(),
            loads: 0,
            readbacks: Vec::new(),
            made: Vec::new(),
            cooking: false,
            steps: 0,
            blocks: 0,
            sourced: 0,
            clears: Vec::new(),
            copies: Vec::new(),
        }
    }

    /// Settles each of `starts` for its records, those already asked for
    /// left as they stand; a start storing no block holds no water and
    /// makes its records dry at once.
    pub(crate) fn settle(&mut self, starts: Vec<(Vec<RecordKey>, Arc<SettleStart>)>) {
        for (records, start) in starts {
            let asked = self.waiting.iter().any(|(held, _)| *held == records)
                || self
                    .squares
                    .iter()
                    .flatten()
                    .any(|square| square.records == records);
            if asked {
                continue;
            }
            if start.stored.stored() == 0 {
                let made = start.finish(&[], 0, true);
                self.made.extend(
                    records
                        .into_iter()
                        .zip(made)
                        .map(|(record, water)| (record, Ok(codec::encode(&water)))),
                );
                continue;
            }
            self.waiting.push_back((records, start));
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

    /// The looks the regions waiting and settling have left at most, each
    /// [`SETTLE_LOOKS`] less those it took.
    pub(crate) fn looks_left(&self) -> u64 {
        let waiting = self.waiting.len() as u64 * SETTLE_LOOKS;
        let settling = self
            .squares
            .iter()
            .flatten()
            .map(|square| SETTLE_LOOKS.saturating_sub(square.looks.len() as u64));
        waiting + settling.sum::<u64>()
    }

    /// Bytes of its buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, SlotBuffers::bytes)
    }

    /// Prepares this frame's settling: takes what read back (deciding
    /// whether the looked regions settled, or making a settled region's
    /// records), its buffers for the regions to come, the waiting starts
    /// into free slots, each region's water to start from, the steps this
    /// frame runs and which regions look at its end, and their slots'
    /// rectangles.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.clears.clear();
        self.copies.clear();
        let ran = std::mem::take(&mut self.steps);
        let wanted = self.take_readbacks(ran);
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
        self.fill_slots();
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        for (slot, square) in self.squares.iter_mut().enumerate() {
            if let Some(square) = square
                && std::mem::take(&mut square.fresh)
            {
                let clears = load(queue, buffers, slot as u32, square);
                self.clears.extend(clears);
            }
        }
        let stepping: u64 = self
            .squares
            .iter()
            .flatten()
            .filter(|square| square.phase == Phase::Stepping)
            .map(|square| square.stepping.max(1))
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
        let looks = self.write_squares(queue);
        self.ask_readbacks(device, (looks, wanted));
    }

    /// Fills its free slots with the starts waiting that fit them, those
    /// larger than the slots hold waiting until every slot is free and the
    /// buffers grow.
    fn fill_slots(&mut self) {
        let Some(capacity) = self.buffers.as_ref().map(|buffers| buffers.capacity) else {
            return;
        };
        for slot in 0..self.squares.len() {
            if self.squares[slot].is_some() {
                continue;
            }
            let fits = |start: &SettleStart| capacity.holds(needs(start));
            let Some(at) = self
                .waiting
                .iter()
                .position(|(_, start)| fits(start.as_ref()))
            else {
                break;
            };
            let Some((records, start)) = self.waiting.remove(at) else {
                break;
            };
            self.loads += 1;
            self.squares[slot] = Some(Square {
                records,
                start,
                load: self.loads,
                steps: 0,
                since_look: 0,
                looks: Vec::new(),
                stepping: 0,
                flowing: 0,
                phase: Phase::Stepping,
                fresh: true,
            });
        }
    }

    /// Writes the frames and each slot's rectangle for this frame's
    /// steps, counting each region's steps; the slots whose regions look at
    /// the frame's end.
    fn write_squares(&mut self, queue: &wgpu::Queue) -> Vec<usize> {
        let steps = self.steps;
        self.blocks = 0;
        self.sourced = 0;
        let mut squares = vec![
            SimSlot {
                shape: [0; 4],
                step: [0.0; 4],
                blocks: [0; 4],
            };
            self.squares.len()
        ];
        let mut looks = Vec::new();
        for (slot, held) in self.squares.iter_mut().enumerate() {
            let Some(square) = held else {
                continue;
            };
            let start = &square.start;
            let stored = &start.stored;
            let mut flags = SIM_LOCAL | SIM_OPEN | SIM_SUMMING | SIM_SPARSE;
            if square.phase == Phase::Stepping && steps > 0 {
                flags |= SIM_STEPPING;
                square.steps += steps;
                square.since_look += steps;
                self.sourced = self.sourced.max(start.sources.len() as u32);
                if square.since_look >= LOOK_EVERY {
                    flags |= SIM_LOOKING;
                    square.since_look = 0;
                    looks.push(slot);
                }
            }
            self.blocks = self.blocks.max(stored.stored() as u32);
            let [x, z] = stored.cells;
            squares[slot] = SimSlot {
                shape: [x, z, flags, start.sources.len() as u32],
                step: [start.cell, 0.0, 0.0, 0.0],
                blocks: [stored.blocks[0], stored.blocks[1], stored.stored() as u32, 0],
            };
        }
        if let Some(buffers) = self.buffers.as_ref() {
            buffers.write_frames(queue, [0, 0]);
            buffers.write_squares(queue, &squares);
        }
        looks
    }

    /// Sets up the read backs at the frame's end: the sums of each slot
    /// of `looks` whose region looks, and the averaged water of each of
    /// `wanted`, whose regions settled.
    fn ask_readbacks(
        &mut self,
        device: &wgpu::Device,
        (looks, wanted): (Vec<usize>, Vec<usize>),
    ) {
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        let held = buffers.held();
        let means = CELL_MEANS as u64;
        let mut copies = Vec::new();
        for slot in looks {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let blocks = square.start.stored.stored() as u64;
            let offset = (buffers.averages_at(slot as u32) + (SIM_CELL_SUMS + means) * held) * 16;
            copies.push(((slot, square.load, false), (offset, SIM_BLOCK_LOOKS * blocks * 16)));
        }
        for slot in wanted {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let cells = square.start.water.len() as u64;
            let offset = (buffers.averages_at(slot as u32) + SIM_CELL_SUMS * held) * 16;
            copies.push(((slot, square.load, true), (offset, means * cells * 16)));
        }
        for (read, range) in copies {
            self.read_back(device, read, range);
        }
    }

    /// Sets up a read back of `bytes` of the averages from `offset` for
    /// `slot`'s region load `load`, of its averaged water where `means`,
    /// else of a look's sums.
    fn read_back(
        &mut self,
        device: &wgpu::Device,
        (slot, load, means): (usize, u32, bool),
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
            load,
            means,
            buffer,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    /// Its buffers for the regions waiting and settling: as many blocks a
    /// slot as the region storing the most stores, as many of them mapped
    /// as the largest rectangle's, as many faces along its edges and cells
    /// rivers bring water to as the most of them, and as many slots as its
    /// memory holds, made anew where more are wanted and no region settles.
    /// A start larger than the GPU binds fails its records.
    fn ensure_buffers(&mut self, device: &wgpu::Device) {
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let mut kept = VecDeque::with_capacity(self.waiting.len());
        for (records, start) in std::mem::take(&mut self.waiting) {
            if slot_largest_bytes(needs(&start)) <= binding {
                kept.push_back((records, start));
                continue;
            }
            let [x, z] = start.stored.cells;
            let error = format!(
                "settling water over {} blocks of {x} by {z} cells takes more than the {binding} \
                 bytes the GPU binds",
                start.stored.stored()
            );
            self.made.extend(
                records
                    .into_iter()
                    .map(|record| (record, Err(error.clone()))),
            );
        }
        self.waiting = kept;
        let largest = self
            .waiting
            .iter()
            .map(|(_, start)| start)
            .chain(self.squares.iter().flatten().map(|square| &square.start))
            .fold(Capacity::default(), |most, start| most.max(needs(start)));
        let settling = self.squares.iter().any(Option::is_some) || !self.readbacks.is_empty();
        let fits = self
            .buffers
            .as_ref()
            .is_some_and(|buffers| buffers.capacity.holds(largest));
        if fits || (settling && self.buffers.is_some()) {
            return;
        }
        let budget = if self.cooking { COOK_BYTES } else { PLAY_BYTES };
        let slots = (budget / slot_bytes(largest).max(1))
            .min(binding / slot_largest_bytes(largest).max(1))
            .clamp(1, SIM_SLOTS as u64) as u32;
        self.buffers = Some(SlotBuffers::new(
            device,
            &self.kernels.layout,
            (largest, slots),
            false,
        ));
        self.squares = (0..slots).map(|_| None).collect();
    }

    /// Takes what read back: a look's sums decide whether its region is
    /// done, settled or out of steps, its averaged water then read back (the
    /// slots returned), a frame having run `ran` steps; a region's averaged
    /// water makes its records. A read back of an earlier load of its slot
    /// is dropped; one that failed fails its region's records.
    fn take_readbacks(&mut self, ran: u32) -> Vec<usize> {
        let mut kept = Vec::with_capacity(self.readbacks.len());
        let mut wanted = Vec::new();
        for readback in std::mem::take(&mut self.readbacks) {
            let taken = std::mem::replace(&mut *readback.mapped.borrow_mut(), Mapped::Failed);
            let read = match taken {
                Mapped::Read(bytes) => Some(Ok(bytes)),
                Mapped::Failed => Some(Err(())),
                waiting @ (Mapped::Recorded | Mapped::Mapping) => {
                    *readback.mapped.borrow_mut() = waiting;
                    None
                }
            };
            let Some(read) = read else {
                kept.push(readback);
                continue;
            };
            let Some(square) = self
                .squares
                .get_mut(readback.slot)
                .and_then(Option::as_mut)
                .filter(|square| square.load == readback.load)
            else {
                continue;
            };
            let Ok(bytes) = read else {
                let error = "settled water did not read back";
                let records = std::mem::take(&mut square.records);
                self.made.extend(
                    records
                        .into_iter()
                        .map(|record| (record, Err(error.into()))),
                );
                self.squares[readback.slot] = None;
                continue;
            };
            let values = floats(&bytes);
            drop(bytes);
            if readback.means {
                self.finish(readback.slot, &values);
            } else if square.phase == Phase::Stepping {
                let done = looked(square, &values);
                // A cook says how its regions settle, a line every sixteen
                // looks from the first and as each one ends.
                if self.cooking && (done || square.looks.len() % 16 == 1) {
                    say(square, ran, done);
                }
                if done {
                    square.phase = Phase::Reading;
                    wanted.push(readback.slot);
                }
            }
        }
        self.readbacks = kept;
        wanted
    }

    /// Slot `slot`'s region settled to `means` (each stored cell's
    /// [`CELL_MEANS`] entries of its means over its last steps, in its
    /// slots' order), which make its records, the slot then free; a cook
    /// says what they came to.
    fn finish(&mut self, slot: usize, means: &[[f32; 4]]) {
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        let means = &means[..means.len().min(square.start.water.len() * CELL_MEANS)];
        let settled = square.steps < SETTLE_STEPS[1];
        let made = square.start.finish(means, square.steps, settled);
        if self.cooking {
            say_made(square, &made);
        }
        for (record, water) in square.records.iter().zip(&made) {
            self.made.push((*record, Ok(codec::encode(water))));
        }
        self.squares[slot] = None;
    }

    /// Encodes this frame's settling into `encoder`: the ranges cleared for
    /// the regions starting, its steps over every slot, the looks at the
    /// frame's end, and the copies read back.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipelines), Some(buffers)) = (self.kernels.created(), &self.buffers) else {
            return;
        };
        for clear in &self.clears {
            let buffer = if clear.averages {
                &buffers.averages
            } else {
                &buffers.steps
            };
            encoder.clear_buffer(buffer, clear.offset, Some(clear.bytes));
        }
        if self.steps > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-settle"),
                timestamp_writes: None,
            });
            let run = (self.steps, self.sourced, true, true);
            encode_steps(&mut pass, pipelines, buffers, run, self.blocks);
            encode_look(&mut pass, pipelines, buffers, self.blocks);
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

/// Writes `square`'s start into slot `slot` of `buffers`: its table, its
/// blocks' ground, their cells' water to start from (into both states),
/// its rivers' sources, and its blocks as they start (each marked where a
/// cell of it holds water no lake holds or rivers bring water to one), its
/// cells' stepping counted; the ranges to clear before it steps: its sums,
/// and its cells' waves, outflow's shares and seconds.
fn load(
    queue: &wgpu::Queue,
    buffers: &SlotBuffers,
    slot: u32,
    square: &mut Square,
) -> [Clear; 2] {
    let start = &square.start;
    let n = start.water.len() as u64;
    buffers.write_table(queue, slot, &start.stored);
    queue.write_buffer(
        &buffers.ground,
        buffers.ground_at(slot) * 4,
        bytemuck::cast_slice(&start.ground),
    );
    let cells: &[u8] = bytemuck::cast_slice(&start.water);
    let at = buffers.cells_at(slot);
    queue.write_buffer(&buffers.cells, at * 16, cells);
    queue.write_buffer(&buffers.cells, (at + buffers.held()) * 16, cells);
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
        queue.write_buffer(
            &buffers.inputs,
            buffers.sources_at(slot) * 16,
            bytemuck::cast_slice(&sources),
        );
    }
    let averages = buffers.averages_at(slot);
    let (blocks, stepping) = blocks(start);
    let steps = buffers.steps_at(slot);
    queue.write_buffer(
        &buffers.steps,
        (steps + buffers.held()) * 8,
        bytemuck::cast_slice(&blocks),
    );
    square.stepping = stepping;
    [
        Clear {
            averages: true,
            offset: averages * 16,
            bytes: SIM_CELL_SUMS * n * 16,
        },
        Clear {
            averages: false,
            offset: steps * 8,
            bytes: n * 8,
        },
    ]
}

/// The blocks `start` stores as the GPU starts them, in its slots' order:
/// each marked where a cell of it holds water no lake holds or rivers bring
/// water to one, then neither stepping nor standing still yet; and the
/// cells of the blocks stepping first, those marked and those beside them.
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

/// Takes look `sums` (each stored block's [`SIM_BLOCK_LOOKS`]: over its
/// cells flowing, the water they held, their speeds and their count, and
/// its cells stepping; the energy they lose, their heave, their swirl and
/// those that ran at the fastest water runs; and the water its cells beside
/// blocks the slot does not store held, and those cells) into `square`'s
/// looks: whether it is done, settled (its means stationary) or out of
/// steps.
fn looked(square: &mut Square, sums: &[[f32; 4]]) -> bool {
    let (mut volume, mut speed, mut flowing, mut stepping) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    let (mut energy, mut heave, mut swirl, mut clamped) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    let (mut against, mut against_cells) = (0.0_f64, 0.0_f64);
    let blocks = square.start.stored.stored();
    for block in sums.chunks_exact(SIM_BLOCK_LOOKS as usize).take(blocks) {
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
    let settled = square.steps >= SETTLE_STEPS[0]
        && count > STILL_LOOKS
        && looks[count - STILL_LOOKS..].iter().all(|look| look.still);
    settled || square.steps >= SETTLE_STEPS[1]
}

/// Prints how `square` settles at its last look, a frame running `steps`
/// steps: its cells, the blocks of them it stores and its rivers' water,
/// those stepping and flowing, the water it holds, its flowing cells' mean
/// speed, energy lost, heave and swirl and those that ran at the fastest
/// water runs, the water standing against the edges of what it stores, and
/// whether it is `done`, settled or out of steps.
fn say(square: &Square, steps: u32, done: bool) {
    let start = &square.start;
    let [x, z] = start.stored.cells;
    let state = match (done, square.steps < SETTLE_STEPS[1]) {
        (false, _) => "settling",
        (true, true) => "settled",
        (true, false) => "out of steps",
    };
    let sources: f32 = start.sources.iter().map(|source| source.rate).sum();
    let area = f64::from(start.cell * start.cell);
    let look = square.looks.last().copied();
    eprintln!(
        "Water settling {x} by {z} cells of {} m, {} blocks of them stored ({} tiles kept, \
         rivers bringing {sources:.0} m3/s): {state} after {} steps, {steps} a frame; {} cells \
         stepping, {} flowing; {:.0} m3 held, flowing at {:.3} m/s, losing {:.1} W/m2, heaving \
         {:.4} m and swirling {:.3}/s on average, {} cells at the fastest water runs; {:.0} m3 \
         against the edges of the blocks stored, over {} cells",
        start.cell,
        start.stored.stored(),
        square.records.len(),
        square.steps,
        square.stepping,
        square.flowing,
        look.map_or(0.0, |look| look.volume * area),
        look.map_or(0.0, |look| look.speed),
        look.map_or(0.0, |look| look.energy),
        look.map_or(0.0, |look| look.heave),
        look.map_or(0.0, |look| look.swirl),
        look.map_or(0.0, |look| look.clamped),
        look.map_or(0.0, |look| look.against * area),
        look.map_or(0.0, |look| look.against_cells),
    );
}

/// The reaches channels run in, as records' stats count them
/// (`engine_terrain::Reach`'s order).
const REACH_NAMES: [&str; 5] = ["pools", "plane beds", "steps", "cascades", "falls"];

/// Prints what `square`'s records `made` came to, their stats summed over
/// its tiles: its cells whose water ran at the fastest water runs, and of
/// its flowing cells in channels, by reach, their mean effective Manning's
/// n, the share of them running faster than their waves and how many there
/// are.
fn say_made(square: &Square, made: &[SteadyWater]) {
    let [x, z] = square.start.stored.cells;
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
    let channels: Vec<String> = REACH_NAMES
        .iter()
        .zip(reaches)
        .map(|(name, (friction, fast, cells))| {
            if cells == 0 {
                return format!("{name} none");
            }
            let count = cells as f64;
            format!(
                "{name} Manning's n {:.3}, {:.0}% above Froude 1, over {cells} cells",
                friction / count,
                100.0 * fast / count
            )
        })
        .collect();
    eprintln!(
        "Water settled over {x} by {z} cells for {} tiles: {clamped} cells at the fastest water \
         runs; in channels, {}",
        made.len(),
        channels.join("; ")
    );
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
