//! Water settling on the GPU for tiles' records (`settled_water`): terrains
//! hand the renderer what a region's water settles from ([`SettleStart`]),
//! the whole terrain's while a session cooks, and it runs the shallow-water
//! scheme ([`crate::water_kernels`]) over as many regions at once as its
//! buffers hold, each a slot stepping its cells by their own seconds with
//! open edges, only the blocks water can reach stepping, until its water
//! settles (or its steps run out), looked at every [`LOOK_EVERY`] steps
//! (its cells' water averaged over the steps since the look before, summed
//! workgroup by workgroup and read back); its averaged water then makes the
//! records of the tiles the region keeps ([`SettleStart::finish`]), which
//! the renderer hands its host to keep and to the terrain that asked. While
//! a session cooks it takes more memory and more steps a frame, and says
//! how its regions settle.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use engine_core::RecordKey;
use engine_core::terrain_data::shallow_water::{DRY, ROUGHNESS, UNHELD};
use engine_core::terrain_data::{SettleLevel, SettleStart, codec};

use crate::gpu_layouts::constants::{
    SIM_LOCAL, SIM_LOOKING, SIM_OPEN, SIM_SLOTS, SIM_SPARSE, SIM_STEPPING, SIM_SUMMING,
    WATER_SIM_GROUP,
};
use crate::gpu_layouts::water::SimSlot;
use crate::water_kernels::{
    Kernels, SlotBuffers, encode_look, encode_steps, groups_of, slot_bytes, slot_largest_bytes,
};

/// Steps between a region's looks at its water, which its frames' steps
/// divide.
pub(crate) const LOOK_EVERY: u32 = 256;
/// The steps a region runs at least and at most, and its looks at most.
const SETTLE_STEPS: [u32; 2] = [1024, 16_384];
const SETTLE_LOOKS: u64 = (SETTLE_STEPS[1] / LOOK_EVERY) as u64;
/// A region settled once, over [`STILL_LOOKS`] looks in a row, the water
/// its region holds changed by no more than [`SETTLED_VOLUME`] of it a
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
/// cooks; and this frame's steps, the cells along x and z its passes cover
/// (the largest region stepping), slots' rectangles, buffer ranges cleared
/// and copies to read back.
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
    extent: [u32; 2],
    sourced: u32,
    clears: Vec<Clear>,
    copies: Vec<CopyBack>,
}

/// A region settling in a slot: the keys of its records and its start, its
/// load's count (its looks and its read backs carry it), its steps, all
/// told and since its last look, what its looks found, its cells stepping
/// and flowing as its last look found them (stepping as it started,
/// before), where it stands, and its water to start from, until written
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
    water: Option<Vec<[f32; 4]>>,
}

/// Where a region stands: stepping, or settled and its averaged water
/// reading back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Stepping,
    Reading,
}

/// What a look found: the water its region held (meters of depth summed
/// over its cells), the mean of how far its flowing cells' averaged
/// velocities moved since the look before, and whether its water held
/// still since then.
#[derive(Clone, Copy)]
struct Look {
    volume: f64,
    moved: f64,
    still: bool,
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
            extent: [0; 2],
            sourced: 0,
            clears: Vec::new(),
            copies: Vec::new(),
        }
    }

    /// Settles each of `starts` for its records, those already asked for
    /// left as they stand.
    pub(crate) fn settle(&mut self, starts: Vec<(Vec<RecordKey>, Arc<SettleStart>)>) {
        for (records, start) in starts {
            let asked = self.waiting.iter().any(|(held, _)| *held == records)
                || self
                    .squares
                    .iter()
                    .flatten()
                    .any(|square| square.records == records);
            if !asked {
                self.waiting.push_back((records, start));
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
        let Some((capacity, sourced)) = self
            .buffers
            .as_ref()
            .map(|buffers| (buffers.capacity, buffers.sources))
        else {
            return;
        };
        // Waiting starts fill free slots, those larger than the slots hold
        // waiting until every slot is free and the buffers grow.
        for slot in 0..self.squares.len() {
            if self.squares[slot].is_some() {
                continue;
            }
            let fits = |start: &SettleStart| {
                let cells = start.level.cells;
                cells[0] <= capacity[0]
                    && cells[1] <= capacity[1]
                    && start.level.sources.len() <= sourced as usize
            };
            let Some(at) = self.waiting.iter().position(|(_, start)| fits(start)) else {
                break;
            };
            let Some((records, start)) = self.waiting.remove(at) else {
                break;
            };
            let water: Vec<[f32; 4]> = start
                .level
                .water
                .iter()
                .map(|&[surface, x, z]| [surface, x, z, 0.0])
                .collect();
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
                water: Some(water),
            });
        }
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        for (slot, square) in self.squares.iter_mut().enumerate() {
            if let Some(square) = square
                && let Some(water) = square.water.take()
            {
                let clears = load(queue, buffers, slot as u32, square, &water);
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
        let steps = self.steps;
        self.extent = [0; 2];
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
            let level = &square.start.level;
            let mut flags = SIM_LOCAL | SIM_OPEN | SIM_SUMMING | SIM_SPARSE;
            let mut summed = 0.0;
            if square.phase == Phase::Stepping && steps > 0 {
                flags |= SIM_STEPPING;
                square.steps += steps;
                square.since_look += steps;
                self.sourced = self.sourced.max(level.sources.len() as u32);
                if square.since_look >= LOOK_EVERY {
                    flags |= SIM_LOOKING;
                    summed = square.since_look as f32;
                    square.since_look = 0;
                    looks.push(slot);
                }
            }
            let [x, z] = level.cells;
            self.extent = [self.extent[0].max(x), self.extent[1].max(z)];
            squares[slot] = SimSlot {
                shape: [x, z, flags, level.sources.len() as u32],
                step: [level.cell, 0.0, ROUGHNESS, summed],
            };
        }
        buffers.write_frames(queue, [0, 0]);
        buffers.write_squares(queue, &squares);
        // Each look's sums, and the averaged water of each region settled,
        // read back at the frame's end.
        let sums_at = 2 * buffers.held();
        let blocks = buffers.blocks();
        let mut copies = Vec::new();
        for slot in looks {
            let load = self.squares[slot].as_ref().map_or(0, |square| square.load);
            let offset = (buffers.averages_at(slot as u32) + sums_at) * 16;
            copies.push(((slot, load, false), (offset, blocks * 16)));
        }
        for slot in wanted {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let [x, z] = square.start.level.cells.map(u64::from);
            let offset = (buffers.averages_at(slot as u32) + x * z) * 16;
            copies.push(((slot, square.load, true), (offset, x * z * 16)));
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

    /// Its buffers for the regions waiting and settling: as large along x
    /// and z as the largest of them, as many cells rivers bring water to
    /// as the most of them, and as many slots as its memory holds, made
    /// anew where more are wanted and no region settles. A start larger than
    /// the GPU binds fails its records.
    fn ensure_buffers(&mut self, device: &wgpu::Device) {
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let mut kept = VecDeque::with_capacity(self.waiting.len());
        for (records, start) in std::mem::take(&mut self.waiting) {
            let cells = start.level.cells;
            let sources = start.level.sources.len() as u64 * 16;
            if slot_largest_bytes(cells).max(sources) <= binding {
                kept.push_back((records, start));
                continue;
            }
            let error = format!(
                "settling water over {} by {} cells takes more than the {binding} bytes the GPU binds",
                cells[0], cells[1]
            );
            self.made.extend(
                records
                    .into_iter()
                    .map(|record| (record, Err(error.clone()))),
            );
        }
        self.waiting = kept;
        let starts = self
            .waiting
            .iter()
            .map(|(_, start)| start)
            .chain(self.squares.iter().flatten().map(|square| &square.start));
        let (mut largest, mut sources) = ([1, 1], 1);
        for start in starts {
            let cells = start.level.cells;
            largest = [largest[0].max(cells[0]), largest[1].max(cells[1])];
            sources = sources.max(start.level.sources.len() as u32);
        }
        let settling = self.squares.iter().any(Option::is_some) || !self.readbacks.is_empty();
        let fits = self.buffers.as_ref().is_some_and(|buffers| {
            buffers.capacity[0] >= largest[0]
                && buffers.capacity[1] >= largest[1]
                && buffers.sources >= sources
        });
        if fits || (settling && self.buffers.is_some()) {
            return;
        }
        let budget = if self.cooking { COOK_BYTES } else { PLAY_BYTES };
        let slots = (budget / slot_bytes(largest, sources))
            .min(binding / slot_largest_bytes(largest))
            .min(binding / (u64::from(sources) * 16))
            .clamp(1, SIM_SLOTS as u64) as u32;
        self.buffers = Some(SlotBuffers::new(
            device,
            &self.kernels.layout,
            (largest, slots),
            (sources, false),
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
                let stride = self
                    .buffers
                    .as_ref()
                    .map_or(1, |buffers| buffers.groups()[0]);
                let done = looked(square, &values, stride);
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

    /// Slot `slot`'s region settled to `means` (each cell's surface, flow
    /// along x and z and foam, averaged over its last steps, row by row),
    /// which make its records, the slot then free.
    fn finish(&mut self, slot: usize, means: &[[f32; 4]]) {
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        let [x, z] = square.start.level.cells.map(|count| count as usize);
        let means = &means[..means.len().min(x * z)];
        let settled = square.steps < SETTLE_STEPS[1];
        let made = square.start.finish(means, square.steps, settled);
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
            encode_steps(&mut pass, pipelines, buffers, run, self.extent);
            encode_look(&mut pass, pipelines, buffers, self.extent);
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

/// Writes `square`'s ground, its water `water` to start from (each cell's
/// surface, flow along x and z and foam) with its lakes' levels, its
/// rivers' sources, the means its looks start from and its blocks as they
/// start (each marked where a cell of it holds water no lake holds or
/// rivers bring water to one) into slot `slot` of `buffers`, its cells'
/// stepping counted; the ranges to clear before it steps: its sums, and
/// its cells' waves and seconds.
fn load(
    queue: &wgpu::Queue,
    buffers: &SlotBuffers,
    slot: u32,
    square: &mut Square,
    water: &[[f32; 4]],
) -> [Clear; 2] {
    let level = &square.start.level;
    let [x, z] = level.cells.map(|count| count as usize);
    let n = x * z;
    let cells: Vec<[f32; 4]> = (0..n)
        .map(|at| {
            let [surface, x, z, _] = water.get(at).copied().unwrap_or_default();
            let held = level.held[at];
            let held = if held.is_nan() { UNHELD } else { held };
            [surface, x, z, held]
        })
        .collect();
    let foam: Vec<f32> = (0..n)
        .map(|at| water.get(at).map_or(0.0, |cell| cell[3]))
        .collect();
    let at = buffers.cells_at(slot);
    let bytes: &[u8] = bytemuck::cast_slice(&cells);
    queue.write_buffer(&buffers.cells, at * 16, bytes);
    queue.write_buffer(&buffers.cells, (at + n as u64) * 16, bytes);
    let foams: &[u8] = bytemuck::cast_slice(&foam);
    queue.write_buffer(&buffers.foam, at * 4, foams);
    queue.write_buffer(&buffers.foam, (at + n as u64) * 4, foams);
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
    // The means before the first look are where the level starts.
    let means: Vec<[f32; 4]> = cells
        .iter()
        .zip(&foam)
        .map(|(cell, &foam)| [cell[0], cell[1], cell[2], foam])
        .collect();
    let averages = buffers.averages_at(slot);
    queue.write_buffer(
        &buffers.averages,
        (averages + n as u64) * 16,
        bytemuck::cast_slice(&means),
    );
    let (blocks, stepping) = blocks(level, &cells, buffers.groups());
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
            bytes: n as u64 * 16,
        },
        Clear {
            averages: false,
            offset: steps * 8,
            bytes: n as u64 * 8,
        },
    ]
}

/// The blocks of a region's cells `level` as the GPU starts them, in rows
/// of the largest
/// rectangle's `groups` along x, from its cells `cells` (surface, flow
/// along x and z, and the level a lake holds them at): each marked where a
/// cell of it holds water no lake holds or rivers bring water to one, then
/// neither stepping nor standing still yet; and the cells of the blocks
/// stepping first, those marked and those beside them.
fn blocks(level: &SettleLevel, cells: &[[f32; 4]], groups: [u32; 2]) -> (Vec<[f32; 2]>, u64) {
    let group = WATER_SIM_GROUP as usize;
    let x = level.cells[0] as usize;
    let stride = groups[0] as usize;
    let mut marked = vec![false; stride * groups[1] as usize];
    let mark = |marked: &mut Vec<bool>, at: usize| {
        marked[(at / x / group) * stride + (at % x) / group] = true;
    };
    for (at, cell) in cells.iter().enumerate() {
        if cell[3] <= 0.5 * UNHELD && cell[0] - level.bed(at) > DRY {
            mark(&mut marked, at);
        }
    }
    for source in &level.sources {
        mark(&mut marked, source.cell as usize);
    }
    let mut entries = vec![[0.0_f32; 2]; 2 * marked.len()];
    for (block, &wet) in marked.iter().enumerate() {
        if wet {
            entries[2 * block] = [1.0, 0.0];
        }
    }
    let [along_x, along_z] = groups_of(level.cells).map(|count| count as usize);
    let mut stepping = 0;
    for j in 0..along_z {
        for i in 0..along_x {
            let rows = j.saturating_sub(1)..=(j + 1).min(along_z - 1);
            let beside = rows.clone().any(|row| {
                (i.saturating_sub(1)..=(i + 1).min(along_x - 1))
                    .any(|column| marked[row * stride + column])
            });
            if beside {
                stepping += (group * group) as u64;
            }
        }
    }
    (entries, stepping)
}

/// Takes look `sums` (each workgroup's: the water its cells held, how far
/// their averaged velocities moved, its cells flowing and stepping, in rows
/// of `stride`) into `square`'s looks: whether it is done, settled or out
/// of steps.
fn looked(square: &mut Square, sums: &[[f32; 4]], stride: u32) -> bool {
    let (mut volume, mut moved, mut flowing, mut stepping) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    let [along_x, along_z] = groups_of(square.start.level.cells);
    for z in 0..along_z {
        for x in 0..along_x {
            let Some(sum) = sums.get((z * stride + x) as usize) else {
                continue;
            };
            volume += f64::from(sum[0]);
            moved += f64::from(sum[1]);
            flowing += f64::from(sum[2]);
            stepping += f64::from(sum[3]);
        }
    }
    square.stepping = stepping as u64;
    square.flowing = flowing as u64;
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
    let settled = square.steps >= SETTLE_STEPS[0]
        && count > STILL_LOOKS
        && looks[count - STILL_LOOKS..].iter().all(|look| look.still)
        && (moved <= SETTLED_SPEED || moved >= STILL_SHARE * looks[count - 1 - STILL_LOOKS].moved);
    settled || square.steps >= SETTLE_STEPS[1]
}

/// Prints how `square` settles at its last look, a frame running `steps`
/// steps: its cells and its rivers' water, those stepping and flowing, the
/// water it holds and how far its velocities moved, and whether it is
/// `done`, settled or out of steps.
fn say(square: &Square, steps: u32, done: bool) {
    let level = &square.start.level;
    let [x, z] = level.cells;
    let state = match (done, square.steps < SETTLE_STEPS[1]) {
        (false, _) => "settling",
        (true, true) => "settled",
        (true, false) => "out of steps",
    };
    let sources: f32 = level.sources.iter().map(|source| source.rate).sum();
    let area = f64::from(level.cell * level.cell);
    let look = square.looks.last().copied();
    eprintln!(
        "Water settling {x} by {z} cells of {} m ({} tiles kept, rivers bringing {sources:.0} m3/s): \
         {state} after {} steps, {steps} a frame; {} cells stepping, {} flowing; \
         {:.0} m3 held, velocities moved {:.4} m/s",
        level.cell,
        square.records.len(),
        square.steps,
        square.stepping,
        square.flowing,
        look.map_or(0.0, |look| look.volume * area),
        look.map_or(0.0, |look| look.moved),
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
