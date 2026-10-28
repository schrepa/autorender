use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::RecordKey;
use engine_core::terrain_data::shallow_water::{CELL_MEANS, DRY, step_seconds};
use engine_core::terrain_data::{
    BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, SettleStart, SteadyWater, codec,
};

use crate::gpu_layouts::constants::{
    SIM_BLOCK_LOOKS, SIM_BURSTING, SIM_CELL_SUMS, SIM_KEEPING, SIM_LOCAL, SIM_LOOKING, SIM_OPEN,
    SIM_SLOTS, SIM_SPARSE, SIM_STEPPING, SIM_SUMMING,
};
use crate::gpu_layouts::water::{SimSetup, SimSlot};
use crate::terrain::TerrainPass;
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_look, encode_steps, slot_bytes, slot_largest_bytes,
};
use crate::water_sim::{Setup, SetupBuffers};

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

pub(crate) struct WaterSettle {
    kernels: Kernels,
    buffers: Option<SlotBuffers>,
    setup: Setup,
    setup_buffers: Option<SetupBuffers>,
    waiting: VecDeque<(u64, Vec<RecordKey>, Arc<SettleStart>)>,
    squares: Vec<Option<Square>>,
    loads: u32,
    readbacks: Vec<Readback>,
    made: Vec<(RecordKey, Result<Vec<u8>, String>)>,
    cooking: bool,
    steps: u32,
    blocks: u32,
    sourced: u32,
    stirring: bool,
    raising: u32,
    clears: Vec<Clear>,
    copies: Vec<CopyBack>,
}

struct Square {
    records: Vec<RecordKey>,
    terrain: u64,
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
    fresh: bool,
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

struct Clear {
    averages: bool,
    offset: u64,
    bytes: u64,
}

struct CopyBack {
    ground: bool,
    offset: u64,
    bytes: u64,
    readback: usize,
    into: u64,
}

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

fn needs(start: &SettleStart) -> Capacity {
    Capacity::of(&start.stored, start.sources.len())
}

impl WaterSettle {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            buffers: None,
            setup: Setup::new(device),
            setup_buffers: None,
            waiting: VecDeque::new(),
            squares: Vec::new(),
            loads: 0,
            readbacks: Vec::new(),
            made: Vec::new(),
            cooking: false,
            steps: 0,
            blocks: 0,
            sourced: 0,
            stirring: false,
            raising: 0,
            clears: Vec::new(),
            copies: Vec::new(),
        }
    }

    pub(crate) fn settle(&mut self, starts: Vec<(u64, Vec<RecordKey>, Arc<SettleStart>)>) {
        for (terrain, records, start) in starts {
            let asked = self.waiting.iter().any(|(_, held, _)| *held == records)
                || self
                    .squares
                    .iter()
                    .flatten()
                    .any(|square| square.records == records);
            if asked {
                continue;
            }
            if start.stored.stored() == 0 {
                let made = start.finish(&[], &[], 0, true);
                self.made.extend(
                    records
                        .into_iter()
                        .zip(made)
                        .map(|(record, water)| (record, Ok(codec::encode(&water)))),
                );
                continue;
            }
            self.waiting.push_back((terrain, records, start));
        }
    }

    pub(crate) fn set_cooking(&mut self, cooking: bool) {
        self.cooking = cooking;
    }

    pub(crate) fn take_made(&mut self) -> Vec<(RecordKey, Result<Vec<u8>, String>)> {
        std::mem::take(&mut self.made)
    }

    pub(crate) fn looks_left(&self) -> u64 {
        let waiting = self.waiting.len() as u64 * SETTLE_LOOKS;
        let settling = self.squares.iter().flatten().map(|square| {
            let taken = u64::from((square.steps + square.stirred) / LOOK_EVERY);
            SETTLE_LOOKS.saturating_sub(taken)
        });
        waiting + settling.sum::<u64>()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, SlotBuffers::bytes)
            + self.setup_buffers.as_ref().map_or(0, SetupBuffers::bytes)
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (shader, terrain): (&wgpu::ShaderModule, &TerrainPass),
    ) {
        self.clears.clear();
        self.copies.clear();
        self.raising = 0;
        let ran = std::mem::take(&mut self.steps);
        let wanted = self.take_readbacks(ran);
        let idle = self.waiting.is_empty() && self.squares.iter().all(Option::is_none);
        if idle {
            if self.readbacks.is_empty() {
                self.buffers = None;
                self.setup_buffers = None;
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
        self.raise((device, queue), (shader, terrain));
        let stepping: u64 = self
            .squares
            .iter()
            .flatten()
            .filter(|square| square.raised && square.phase != Phase::Reading)
            .map(|square| square.stepping.max(1))
            .sum();
        let budget = if self.cooking {
            COOK_CELL_STEPS
        } else {
            PLAY_CELL_STEPS
        };
        if let Some(most) = budget.checked_div(stepping) {
            let most = most.clamp(1, u64::from(LOOK_EVERY)) as u32;
            self.steps = 1 << (u32::BITS - 1 - most.leading_zeros());
        }
        let looks = self.write_squares(queue);
        self.ask_readbacks(device, (looks, wanted));
    }

    fn raise(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        (shader, terrain): (&wgpu::ShaderModule, &TerrainPass),
    ) {
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        let found = self.squares.iter().enumerate().find_map(|(slot, square)| {
            let square = square.as_ref().filter(|square| !square.raised)?;
            Some((slot, terrain.ground(square.terrain)?))
        });
        self.setup.hold(found.as_ref().map(|(_, ground)| ground));
        let Some((slot, _)) = found else {
            return;
        };
        self.setup.ensure(device, shader);
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        if !self.setup.sets_up() {
            return;
        }
        let start = &square.start;
        let blocks = start.stored.stored();
        let window = (start.window * start.window) as usize;
        let entries = 2 * blocks + blocks * window;
        if self
            .setup_buffers
            .as_ref()
            .is_none_or(|setup| setup.room < entries)
        {
            self.setup_buffers = Some(SetupBuffers::new(
                device,
                &self.setup.layout,
                buffers,
                (entries, 0),
            ));
        }
        let Some(setup) = self.setup_buffers.as_ref() else {
            return;
        };
        let side = i64::from(BLOCK_SIDE);
        let first = (slot as u32) * buffers.capacity.blocks;
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
        queue.write_buffer(&setup.list, 0, bytemuck::cast_slice(&list));
        let frame = SimSetup {
            counts: [blocks as u32, 0, 0, 0],
            square: [start.window, start.last[0] as u32, start.last[1] as u32, 0],
            meters: [start.cell, start.datum, 0.0, 0.0],
            ring: [0; 4],
            tiles: [0; 4],
        };
        queue.write_buffer(&setup.frame, 0, bytemuck::bytes_of(&frame));
        square.raised = true;
        self.raising = blocks as u32;
    }

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
                .position(|(_, _, start)| fits(start.as_ref()))
            else {
                break;
            };
            let Some((terrain, records, start)) = self.waiting.remove(at) else {
                break;
            };
            self.loads += 1;
            self.squares[slot] = Some(Square {
                records,
                terrain,
                start,
                load: self.loads,
                raised: false,
                steps: 0,
                since_look: 0,
                stirred: 0,
                looks: Vec::new(),
                settled: false,
                stepping: 0,
                flowing: 0,
                phase: Phase::Stepping,
                fresh: true,
            });
        }
    }

    fn write_squares(&mut self, queue: &wgpu::Queue) -> Vec<usize> {
        let steps = self.steps;
        self.blocks = 0;
        self.sourced = 0;
        self.stirring = false;
        let mut squares = vec![SimSlot::zeroed(); self.squares.len()];
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
                    self.stirring = true;
                } else {
                    square.steps += steps;
                }
                self.sourced = self.sourced.max(start.sources.len() as u32);
                if square.since_look >= LOOK_EVERY {
                    flags |= SIM_LOOKING;
                    square.since_look = 0;
                    looks.push(slot);
                }
            }
            self.blocks = self.blocks.max(stored.stored() as u32);
            let [x, z] = stored.cells;
            let [first_x, first_z] = start.first.map(|first| first as i32 as u32);
            squares[slot] = SimSlot {
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
        if let Some(buffers) = self.buffers.as_ref() {
            buffers.write_frames(queue);
            buffers.write_squares(queue, &squares);
        }
        looks
    }

    fn ask_readbacks(&mut self, device: &wgpu::Device, (looks, wanted): (Vec<usize>, Vec<usize>)) {
        let Some(buffers) = self.buffers.as_ref() else {
            return;
        };
        let held = buffers.held();
        let means = CELL_MEANS as u64;
        let mut reads = Vec::new();
        for slot in looks {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let blocks = square.start.stored.stored() as u64;
            let offset = (buffers.averages_at(slot as u32) + (SIM_CELL_SUMS + means) * held) * 16;
            reads.push((
                (slot, square.load, false),
                [(false, offset, SIM_BLOCK_LOOKS * blocks * 16)],
                None,
            ));
        }
        for slot in wanted {
            let Some(square) = self.squares[slot].as_ref() else {
                continue;
            };
            let cells = square.start.water.len() as u64;
            let blocks = square.start.stored.stored() as u64;
            let offset = (buffers.averages_at(slot as u32) + SIM_CELL_SUMS * held) * 16;
            let ground = buffers.ground_at(slot as u32) * 4;
            reads.push((
                (slot, square.load, true),
                [(false, offset, means * cells * 16)],
                Some((true, ground, blocks * BLOCK_GROUND as u64 * 4)),
            ));
        }
        for (read, [first], then) in reads {
            let copies: Vec<(bool, u64, u64)> = std::iter::once(first).chain(then).collect();
            self.read_back(device, read, &copies);
        }
    }

    fn read_back(
        &mut self,
        device: &wgpu::Device,
        (slot, load, means): (usize, u32, bool),
        copies: &[(bool, u64, u64)],
    ) {
        let bytes: u64 = copies.iter().map(|&(_, _, bytes)| bytes).sum();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/water-settle-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut into = 0;
        for &(ground, offset, bytes) in copies {
            self.copies.push(CopyBack {
                ground,
                offset,
                bytes,
                readback: self.readbacks.len(),
                into,
            });
            into += bytes;
        }
        self.readbacks.push(Readback {
            slot,
            load,
            means,
            buffer,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    fn ensure_buffers(&mut self, device: &wgpu::Device) {
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let mut kept = VecDeque::with_capacity(self.waiting.len());
        for (terrain, records, start) in std::mem::take(&mut self.waiting) {
            if slot_largest_bytes(needs(&start)) <= binding {
                kept.push_back((terrain, records, start));
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
            .map(|(_, _, start)| start)
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
        self.setup_buffers = None;
        self.squares = (0..slots).map(|_| None).collect();
    }

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
            if readback.means {
                self.finish(readback.slot, &bytes);
                continue;
            }
            let values = floats(&bytes);
            drop(bytes);
            if square.phase == Phase::Reading {
                continue;
            }
            let done = looked(square, &values);
            if self.cooking && (done || square.looks.len() % 16 == 1) {
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
                wanted.push(readback.slot);
            }
        }
        self.readbacks = kept;
        wanted
    }

    fn finish(&mut self, slot: usize, bytes: &[u8]) {
        let Some(square) = self.squares[slot].as_mut() else {
            return;
        };
        let start = &square.start;
        let split = (start.water.len() * CELL_MEANS * 16).min(bytes.len());
        let means = floats(&bytes[..split]);
        let ground: Vec<f32> = bytes[split..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let settled = square.settled && square.stirred < STIR_STEPS[1];
        let made = start.finish(&means, &ground, square.steps + square.stirred, settled);
        if self.cooking {
            say_made(square, &made);
        }
        for (record, water) in square.records.iter().zip(&made) {
            self.made.push((*record, Ok(codec::encode(water))));
        }
        self.squares[slot] = None;
    }

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
        if self.raising > 0
            && let Some(setup) = self.setup_buffers.as_ref()
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-settle-raise"),
                timestamp_writes: None,
            });
            self.setup
                .encode_raise(&mut pass, &setup.group, self.raising);
        }
        if self.steps > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-settle"),
                timestamp_writes: None,
            });
            let run = (self.steps, self.sourced, true, self.stirring);
            encode_steps(&mut pass, pipelines, buffers, run, self.blocks);
            encode_look(&mut pass, pipelines, buffers, self.blocks);
        }
        for copy in &self.copies {
            let readback = &self.readbacks[copy.readback];
            let source = if copy.ground {
                &buffers.ground
            } else {
                &buffers.averages
            };
            encoder.copy_buffer_to_buffer(
                source,
                copy.offset,
                &readback.buffer,
                copy.into,
                copy.bytes,
            );
        }
    }

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

fn load(queue: &wgpu::Queue, buffers: &SlotBuffers, slot: u32, square: &mut Square) -> [Clear; 2] {
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
        (false, true, true) => "settled, now stirring",
        (false, true, false) => "out of steps, now stirring",
        (true, false, _) => "stirring",
        (true, true, true) => "stirred",
        (true, true, false) => "out of stirring's steps",
    };
    let sources: f32 = start.sources.iter().map(|source| source.rate).sum();
    let area = f64::from(start.cell * start.cell);
    let look = square.looks.last().copied();
    eprintln!(
        "Water settling {x} by {z} cells of {} m, {} blocks of them stored ({} tiles kept, \
         rivers bringing {sources:.0} m3/s): {state} after {} steps, {} stirred, {steps} a \
         frame; {} cells stepping, {} flowing; {:.0} m3 held, flowing at {:.3} m/s, losing \
         {:.1} W/m2, heaving {:.4} m and swirling {:.3}/s on average, {} cells at the fastest \
         water runs; {:.0} m3 against the edges of the blocks stored, over {} cells",
        start.cell,
        start.stored.stored(),
        square.records.len(),
        square.steps,
        square.stirred,
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

const REACH_NAMES: [&str; 5] = ["pools", "plane beds", "steps", "cascades", "falls"];

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
         runs; {standing} of {} solids standing in its water; in channels, {}",
        made.len(),
        square.start.solids.len(),
        channels.join("; ")
    );
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
