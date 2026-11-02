use std::sync::Arc;

use engine_core::Error;
use engine_places::ground::{TerrainEntry, TerrainTiles};
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, Encode, Install, Lifetime, Pass, Prepare, SamplerDesc,
    Texture, wgpu,
};
use glam::{Vec2, Vec3};

use crate::bounce::{PROBE_BYTES, TerrainBounce, bin_moments};
use crate::gpu::{BOUNCE_GROUP, BounceFrame, BounceTerrain};
use crate::settings::{MAX_BOUNCE_LEVELS, TerrainShading};
use crate::state::Bakes;
use crate::terrain::{TerrainGpu, reached};
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_sky::SkyMaps;

pub(crate) const BOUNCE_LAYERS: usize = 4 * crate::bounce::LAYERS;

pub(crate) const MOST_LEVELS: usize = MAX_BOUNCE_LEVELS as usize;

const WINDOW_MARGIN: f32 = 2.5;

const SCRATCH_PROBE_BYTES: u64 = (BOUNCE_LAYERS * 16) as u64;

const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GROUND_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
const GROUND_LAYERS: u32 = 2;
const LIGHT_LEVELS: usize = (u32::BITS - (crate::bounce::MAX_PROBES / 2).leading_zeros()) as usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BounceLevels {
    pub(crate) slot: u32,
    pub(crate) probes: u32,
    pub(crate) columns: u32,
    pub(crate) levels: u32,
    pub(crate) finest: u32,
    pub(crate) first: [[u32; 2]; MOST_LEVELS],
    pub(crate) reach: f32,
    pub(crate) row: u32,
}

impl BounceLevels {
    fn window(&self, level: u32) -> u32 {
        self.columns.min(self.probes >> level)
    }

    fn region(&self, level: u32) -> [u32; 2] {
        [
            level % self.row * self.columns,
            level / self.row * self.columns,
        ]
    }

    fn scratch_bytes(&self) -> u64 {
        (0..self.levels)
            .map(|level| u64::from(self.window(level)).pow(2) * SCRATCH_PROBE_BYTES)
            .sum()
    }
}

pub(crate) struct BounceSources<'a> {
    pub(crate) horizons: &'a Horizons,
    pub(crate) sky: &'a SkyMaps,
    pub(crate) colors: &'a MacroMaps,
    pub(crate) palette: Buffer<TerrainGpu>,
}

struct Pipelines {
    ground: ComputePipeline<TerrainGpu>,
    downs: Vec<ComputePipeline<TerrainGpu>>,
    gathers: Vec<ComputePipeline<TerrainGpu>>,
}

pub(crate) struct Bounce {
    pipelines: Pipelines,
    frame: Buffer<TerrainGpu>,
    terrains: Vec<Buffer<TerrainGpu>>,
    blocks: Buffer<TerrainGpu>,
    probes: Buffer<TerrainGpu>,
    scratch: Buffer<TerrainGpu>,
    ground: Texture<TerrainGpu>,
    light: Texture<TerrainGpu>,
    stand_ins: [Texture<TerrainGpu>; 2],
    framed: bool,
    maps: Option<(u32, usize)>,
    slots: Vec<Option<u32>>,
    windows: Vec<Option<BounceLevels>>,
    held: Vec<Option<Arc<TerrainBounce>>>,
    means: Vec<Vec<u8>>,
    bases: Vec<[u32; 3]>,
    records: Vec<Vec<BounceTerrain>>,
    widest: [u32; MOST_LEVELS],
    pub(crate) relit: u32,
    pub(crate) relit_probes: u32,
    pub(crate) levels: u32,
    pub(crate) dropped: u32,
}

impl Bounce {
    pub(crate) fn install(
        cx: &mut Install<'_, TerrainGpu>,
        sources: &BounceSources<'_>,
        pass: Pass<TerrainGpu>,
    ) -> Result<Self, Error> {
        use crate::shaders::terrain_shading::bounce_maps;
        let frame = cx.buffer("bounce_frame", Lifetime::Kept);
        let terrains: Vec<Buffer<TerrainGpu>> = (0..MOST_LEVELS)
            .map(|_| cx.buffer("bounce_terrains", Lifetime::Kept))
            .collect();
        let blocks = cx.buffer("bounce_blocks", Lifetime::Kept);
        let probes = cx.buffer("bounce_records", Lifetime::Kept);
        let scratch = cx.buffer("bounce_scratch", Lifetime::Kept);
        let ground = cx.texture("bounce_ground", GROUND_FORMAT, Lifetime::Kept);
        let light = cx.texture("bounce_light", LIGHT_FORMAT, Lifetime::Kept);
        let light_stand_in = cx.texture("bounce_light_stand_in", LIGHT_FORMAT, Lifetime::Kept);
        let out_stand_in = cx.texture("bounce_out_stand_in", LIGHT_FORMAT, Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        let level = |mip: u32| Bind::Slice {
            texture: light,
            mips: mip..mip + 1,
            layers: 0..u32::MAX,
        };
        let shared = |records: Buffer<TerrainGpu>,
                      light_in: Bind<TerrainGpu>,
                      light_out: Bind<TerrainGpu>| {
            [
                (bounce_maps::BOUNCE_FRAME, Bind::Buffer(frame)),
                (bounce_maps::BOUNCE_TERRAINS, Bind::Buffer(records)),
                (bounce_maps::BOUNCE_SKY, Bind::Texture(sources.sky.texture)),
                (bounce_maps::BOUNCE_SURFACE, Bind::Texture(ground)),
                (bounce_maps::BOUNCE_BLOCKS, Bind::Buffer(blocks)),
                (bounce_maps::BOUNCE_RECORDS, Bind::Buffer(probes)),
                (bounce_maps::BOUNCE_LIGHT_IN, light_in),
                (bounce_maps::BOUNCE_LIGHT_OUT, light_out),
                (
                    bounce_maps::BOUNCE_PROBES,
                    Bind::Texture(sources.horizons.texture),
                ),
                (bounce_maps::BOUNCE_SAMPLER, Bind::Sampler(sampler)),
                (bounce_maps::BOUNCE_PALETTE, Bind::Buffer(sources.palette)),
                (
                    bounce_maps::BOUNCE_MACRO,
                    Bind::Texture(sources.colors.texture),
                ),
                (bounce_maps::BOUNCE_SCRATCH, Bind::Buffer(scratch)),
            ]
        };
        let bound = reached(
            cx,
            bounce_maps::BOUNCE_GROUND,
            shared(terrains[0], Bind::Texture(light_stand_in), level(0)),
        );
        let ground_pipeline = cx.compute_pipeline(pass, bounce_maps::BOUNCE_GROUND, &bound)?;
        let mut downs = Vec::with_capacity(LIGHT_LEVELS - 1);
        for mip in 0..LIGHT_LEVELS as u32 - 1 {
            let bound = reached(
                cx,
                bounce_maps::BOUNCE_DOWN,
                shared(terrains[0], level(mip), level(mip + 1)),
            );
            downs.push(cx.compute_pipeline(pass, bounce_maps::BOUNCE_DOWN, &bound)?);
        }
        let mut gathers = Vec::with_capacity(MOST_LEVELS);
        for &records in &terrains {
            let bound = reached(
                cx,
                bounce_maps::BOUNCE_GATHER,
                shared(records, Bind::Texture(light), Bind::Texture(out_stand_in)),
            );
            gathers.push(cx.compute_pipeline(pass, bounce_maps::BOUNCE_GATHER, &bound)?);
        }
        Ok(Self {
            pipelines: Pipelines {
                ground: ground_pipeline,
                downs,
                gathers,
            },
            frame,
            terrains,
            blocks,
            probes,
            scratch,
            ground,
            light,
            framed: false,
            maps: None,
            slots: Vec::new(),
            windows: Vec::new(),
            held: Vec::new(),
            means: Vec::new(),
            bases: Vec::new(),
            records: Vec::new(),
            widest: [0; MOST_LEVELS],
            relit: 0,
            relit_probes: 0,
            levels: 0,
            dropped: 0,
            stand_ins: [light_stand_in, out_stand_in],
        })
    }

    pub(crate) fn relights(&self) -> bool {
        self.widest.iter().any(|&widest| widest > 0)
    }

    fn release(&mut self) {
        self.maps = None;
        self.held.clear();
        self.means.clear();
        self.bases.clear();
        self.records.clear();
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, bakes, settings, eye, limits): (
            &TerrainTiles,
            &Bakes,
            &TerrainShading,
            Vec3,
            (u32, u64),
        ),
        (horizons, sky, colors, palette): (&Horizons, &SkyMaps, &MacroMaps, &[u32]),
    ) -> bool {
        if !self.framed {
            self.framed = true;
            cx.allocate_buffer(self.frame, size_of::<BounceFrame>() as u64);
            cx.write(
                self.frame,
                0,
                &[BounceFrame {
                    bins: bin_moments(),
                }],
            );
            for stand_in in self.stand_ins {
                cx.allocate(
                    stand_in,
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    1,
                );
            }
        }
        self.widest = [0; MOST_LEVELS];
        self.relit = 0;
        self.relit_probes = 0;
        self.levels = 0;
        self.place(&tiles.terrains, (bakes, settings, eye), limits);
        let count = self.slots.iter().flatten().count();
        if count == 0 {
            self.release();
            return false;
        }
        if !self.ensure_maps(cx, (&tiles.terrains, bakes, horizons), count) {
            return false;
        }
        let mut changed = self.bases.len() != count;
        let listed = tiles.terrains.iter().zip(&self.windows);
        for ((terrain, window), &slot) in listed.zip(&self.slots) {
            let Some(slot) = slot else {
                continue;
            };
            let made = bakes
                .of(terrain.terrain)
                .and_then(|held| held.bounce.made.as_ref());
            let wanted = made.filter(|_| window.is_some());
            let held = &mut self.held[slot as usize];
            match (wanted, &*held) {
                (Some(bounce), Some(current)) if Arc::ptr_eq(bounce, current) => {}
                (Some(bounce), _) => {
                    write_ground(cx, self.ground, slot as usize, bounce);
                    self.means[slot as usize] = means(bounce);
                    *held = Some(Arc::clone(bounce));
                    changed = true;
                }
                (None, Some(_)) if made.is_some() => {}
                (None, current) => {
                    changed |= current.is_some();
                    *held = None;
                    self.means[slot as usize].clear();
                }
            }
        }
        if changed {
            self.upload(cx);
        }
        let levels = self.windows.iter().flatten().map(|window| window.levels);
        let levels = levels.max().unwrap_or(1) as usize;
        self.records.resize_with(levels, Vec::new);
        for records in &mut self.records {
            records.clear();
            records.resize(count, BounceTerrain::default());
        }
        let mut scratch = 0_u32;
        let (casting, _) = crate::terrain_horizons::room(limits.0, tiles.terrains.len());
        let horizon_layers = (crate::lighting::HORIZON_LAYERS * casting) as u32;
        let listed = tiles.terrains.iter().zip(&self.windows).enumerate();
        for (index, (terrain, window)) in listed {
            let Some(window) = window else {
                continue;
            };
            let slot = window.slot as usize;
            let Some(bounce) = &self.held[slot] else {
                continue;
            };
            if !horizons.held(index) {
                continue;
            }
            self.relit += 1;
            self.levels = self.levels.max(window.levels - window.finest);
            let shared = terrain_record(
                terrain,
                bounce,
                (window, index, horizon_layers),
                self.bases[slot],
                (
                    (sky.layer(index), colors.layer(index)),
                    palette.get(index).copied().unwrap_or(0),
                ),
            );
            let mut finer = [0; 4];
            for level in 0..window.levels {
                let mut record = shared;
                if level >= 2 {
                    record.sources[2] += mean_offset(bounce.probes, level);
                }
                if level >= window.finest {
                    let columns = window.window(level);
                    let [x, z] = window.first[level as usize];
                    let [region_x, region_y] = window.region(level);
                    record.level = [level, columns, x, z];
                    record.kept = [region_x, region_y, scratch, 0];
                    record.finer = finer;
                    finer = [x, z, columns, scratch];
                    scratch += columns * columns;
                    self.relit_probes += columns * columns;
                    let invocations = if level == 0 { columns / 2 } else { columns };
                    let widest = &mut self.widest[level as usize];
                    *widest = (*widest).max(invocations);
                }
                self.records[level as usize][slot] = record;
            }
        }
        for (records, &buffer) in self.records.iter().zip(&self.terrains) {
            let bytes = (records.len() * size_of::<BounceTerrain>()) as u64;
            cx.allocate_buffer(buffer, bytes.next_power_of_two().max(16));
            cx.write(buffer, 0, records);
        }
        let scratch_bytes = u64::from(scratch) * SCRATCH_PROBE_BYTES;
        cx.allocate_buffer(self.scratch, scratch_bytes.next_power_of_two().max(16));
        self.relights()
    }

    fn place(
        &mut self,
        terrains: &[TerrainEntry],
        (bakes, settings, eye): (&Bakes, &TerrainShading, Vec3),
        limits: (u32, u64),
    ) {
        let (casting, slots) = crate::terrain_horizons::room(limits.0, terrains.len());
        let on = settings.sun_shadows && settings.bounce;
        let mut next = 0;
        self.slots.clear();
        self.slots
            .extend(terrains.iter().enumerate().map(|(index, terrain)| {
                let relit = on && !terrain.surroundings && index < casting && next < slots;
                let slot = relit.then_some(next);
                next += u32::from(relit);
                slot
            }));
        self.windows = windows(terrains, bakes, (eye, settings), limits);
        self.dropped = self
            .slots
            .iter()
            .zip(&self.windows)
            .zip(terrains)
            .filter(|((slot, window), terrain)| {
                slot.is_some()
                    && window.is_none()
                    && bakes
                        .of(terrain.terrain)
                        .is_some_and(|held| held.bounce.made.is_some())
            })
            .count() as u32;
    }

    fn ensure_maps(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (terrains, bakes, horizons): (&[TerrainEntry], &Bakes, &Horizons),
        count: usize,
    ) -> bool {
        let side = terrains
            .iter()
            .zip(&self.windows)
            .filter_map(|(terrain, window)| {
                window.and(
                    bakes
                        .of(terrain.terrain)
                        .and_then(|held| held.bounce.made.as_ref()),
                )
            })
            .map(|bounce| bounce.texels())
            .max();
        let Some(side) = side.filter(|_| horizons.bounces()) else {
            return false;
        };
        if self
            .maps
            .is_none_or(|(held, slots)| held < side || slots != count)
        {
            let layers = count.max(1) as u32;
            cx.allocate(
                self.ground,
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: GROUND_LAYERS * layers,
                },
                1,
            );
            cx.allocate(
                self.light,
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: layers,
                },
                u32::BITS - side.leading_zeros(),
            );
            self.maps = Some((side, count));
            self.held = vec![None; count];
            self.means = vec![Vec::new(); count];
        }
        true
    }

    fn upload(&mut self, cx: &mut Prepare<'_, TerrainGpu>) {
        let mut words: Vec<u32> = Vec::new();
        let mut probes: Vec<u8> = Vec::new();
        self.bases.clear();
        for (held, means) in self.held.iter().zip(&self.means) {
            let probe_base = (probes.len() / PROBE_BYTES) as u32;
            let count = held.as_ref().map_or(0, |bounce| bounce.probe_count());
            self.bases
                .push([words.len() as u32, probe_base, probe_base + count as u32]);
            if let Some(bounce) = held {
                words.extend(bounce.block_words());
                probes.extend_from_slice(bounce.probe_bytes());
                probes.extend_from_slice(means);
            }
        }
        cx.allocate_buffer(
            self.blocks,
            ((words.len() * 4) as u64).next_power_of_two().max(16),
        );
        cx.allocate_buffer(
            self.probes,
            (probes.len() as u64).next_power_of_two().max(16),
        );
        if !words.is_empty() {
            cx.write(self.blocks, 0, &words);
        }
        if !probes.is_empty() {
            cx.write(self.probes, 0, &probes);
        }
    }

    pub(crate) fn encode(&self, cx: &mut Encode<'_, TerrainGpu>) -> u32 {
        let Some((side, slots)) = self.maps.filter(|_| self.relights()) else {
            return 0;
        };
        let layers = slots as u32;
        let groups = |texels: u32| texels.max(1).div_ceil(BOUNCE_GROUP);
        let mut dispatched = cx.dispatch(
            &self.pipelines.ground,
            Count::Groups([groups(side), groups(side), layers]),
        );
        let mips = (u32::BITS - side.leading_zeros()) as usize;
        for (index, pipeline) in self.pipelines.downs.iter().take(mips - 1).enumerate() {
            let side = side >> (index + 1);
            dispatched += cx.dispatch(
                pipeline,
                Count::Groups([groups(side), groups(side), layers]),
            );
        }
        for (pipeline, &widest) in self.pipelines.gathers.iter().zip(&self.widest) {
            if widest == 0 {
                continue;
            }
            dispatched += cx.dispatch(
                pipeline,
                Count::Groups([groups(widest), groups(widest), layers]),
            );
        }
        dispatched
    }
}

fn terrain_record(
    terrain: &TerrainEntry,
    bounce: &TerrainBounce,
    (window, index, horizon_layers): (&BounceLevels, usize, u32),
    bases: [u32; 3],
    ((sky, colors), palette): ((u32, u32), u32),
) -> BounceTerrain {
    let extent = terrain.pyramid.extent();
    let slot = window.slot;
    BounceTerrain {
        place: [terrain.origin.x, terrain.origin.z, extent.x, extent.y],
        grid: [bounce.probes, bounce.texels(), palette, colors],
        layers: [
            GROUND_LAYERS * slot,
            slot,
            horizon_layers + BOUNCE_LAYERS as u32 * slot,
            sky,
        ],
        sources: [bases[0], bases[1], bases[2], index as u32],
        level: [0; 4],
        kept: [0; 4],
        finer: [0; 4],
    }
}

pub(crate) fn relit_slots(
    terrains: &[TerrainEntry],
    settings: &TerrainShading,
    layers: u32,
) -> u32 {
    let (casting, slots) = crate::terrain_horizons::room(layers, terrains.len());
    if !(settings.sun_shadows && settings.bounce) {
        return 0;
    }
    let wanted = terrains
        .iter()
        .take(casting)
        .filter(|terrain| !terrain.surroundings)
        .count();
    (wanted as u32).min(slots)
}

pub(crate) fn windows(
    terrains: &[TerrainEntry],
    bakes: &Bakes,
    (eye, settings): (Vec3, &TerrainShading),
    (layers, storage_room): (u32, u64),
) -> Vec<Option<BounceLevels>> {
    let (casting, slots) = crate::terrain_horizons::room(layers, terrains.len());
    let on = settings.sun_shadows && settings.bounce;
    let mut next = 0;
    let mut room = [storage_room / 16 * 16; 3];
    terrains
        .iter()
        .enumerate()
        .map(|(index, terrain)| {
            let relit = on && !terrain.surroundings && index < casting && next < slots;
            let slot = relit.then_some(next);
            next += u32::from(relit);
            let bounce = bakes.of(terrain.terrain)?.bounce.made.as_ref()?;
            levels(terrain, (bounce, slot?), eye, settings).filter(|levels| {
                let blocks = u64::from(bounce.texels());
                let records = bounce.probe_bytes().len() + mean_bytes(bounce.probes);
                let needs = [blocks * blocks * 4, records as u64, levels.scratch_bytes()];
                let fits = needs.iter().zip(&room).all(|(need, left)| need <= left);
                if fits {
                    room = [room[0] - needs[0], room[1] - needs[1], room[2] - needs[2]];
                }
                fits
            })
        })
        .collect()
}

fn levels(
    terrain: &TerrainEntry,
    (bounce, slot): (&TerrainBounce, u32),
    eye: Vec3,
    settings: &TerrainShading,
) -> Option<BounceLevels> {
    let probes = bounce.probes;
    let extent = terrain.pyramid.extent();
    let most = settings.bounce_levels.clamp(1, MAX_BOUNCE_LEVELS);
    let short = probes.div_ceil(1 << (most - 1)).next_multiple_of(2);
    let columns = settings
        .bounce_columns
        .max(short)
        .next_multiple_of(2)
        .min(probes);
    let mut levels: u32 = 1;
    while probes >> (levels - 1) > columns {
        levels += 1;
    }
    let size = settings.shadow_resolution;
    let row = size / columns.max(1);
    if row == 0 || levels.div_ceil(row) * columns > size {
        return None;
    }
    let local = Vec2::new(eye.x - terrain.origin.x, eye.z - terrain.origin.z);
    let nearest = Vec2::new(
        engine_math::clamp(local.x, 0.0, extent.x),
        engine_math::clamp(local.y, 0.0, extent.y),
    );
    let away = local.distance(nearest);
    let spacing = engine_math::min(extent.x, extent.y) / probes as f32;
    let reach = (columns as f32 * 0.5 - WINDOW_MARGIN) * spacing;
    let finest = (0..levels)
        .find(|&level| reach * (1_u32 << level) as f32 >= away)
        .unwrap_or(levels - 1);
    let mut first = [[0; 2]; MOST_LEVELS];
    for (level, first) in first.iter_mut().enumerate().take(levels as usize) {
        let along = probes >> level;
        let window = columns.min(along);
        let start = |position: f32, side: f32| {
            let start = (position / side * along as f32).floor() - (window / 2) as f32;
            let last = (along - window) as f32;
            (engine_math::clamp(start, 0.0, last) as u32) & !1
        };
        *first = [start(local.x, extent.x), start(local.y, extent.y)];
    }
    Some(BounceLevels {
        slot,
        probes,
        columns,
        levels,
        finest,
        first,
        reach,
        row,
    })
}

fn mean_offset(probes: u32, level: u32) -> u32 {
    (2..level).map(|finer| (probes >> finer).pow(2)).sum()
}

fn mean_bytes(probes: u32) -> usize {
    let levels = (2..u32::BITS).take_while(|&level| probes >> level > 0);
    levels
        .map(|level| (probes >> level).pow(2) as usize)
        .sum::<usize>()
        * PROBE_BYTES
}

fn means(bounce: &TerrainBounce) -> Vec<u8> {
    let records = bounce.probe_bytes();
    let record = |index: usize| &records[index * PROBE_BYTES..(index + 1) * PROBE_BYTES];
    let words = bounce.block_words();
    let mut level = Vec::with_capacity(words.len() * PROBE_BYTES);
    for word in words {
        let first = (word >> 1) as usize;
        if word & 1 == 0 {
            level.extend_from_slice(record(first));
        } else {
            level.extend(mean_of([
                record(first),
                record(first + 1),
                record(first + 2),
                record(first + 3),
            ]));
        }
    }
    let mut out = Vec::with_capacity(mean_bytes(bounce.probes));
    let mut side = bounce.texels() as usize;
    while side > 1 {
        let coarser = side / 2;
        let at = |x: usize, z: usize| {
            let index = z * side + x;
            &level[index * PROBE_BYTES..(index + 1) * PROBE_BYTES]
        };
        let mut next = Vec::with_capacity(coarser * coarser * PROBE_BYTES);
        for z in 0..coarser {
            for x in 0..coarser {
                next.extend(mean_of([
                    at(2 * x, 2 * z),
                    at(2 * x + 1, 2 * z),
                    at(2 * x, 2 * z + 1),
                    at(2 * x + 1, 2 * z + 1),
                ]));
            }
        }
        out.extend_from_slice(&next);
        level = next;
        side = coarser;
    }
    out
}

fn mean_of(four: [&[u8]; 4]) -> [u8; PROBE_BYTES] {
    let mut mean = [0_u8; PROBE_BYTES];
    for bin in 0..PROBE_BYTES / 2 {
        let share = |record: &[u8]| u32::from(record[2 * bin]);
        let filled: u32 = four.iter().map(|&record| share(record)).sum();
        let coded: u32 = four
            .iter()
            .map(|&record| share(record) * u32::from(record[2 * bin + 1]))
            .sum();
        mean[2 * bin] = ((filled + 2) / 4) as u8;
        mean[2 * bin + 1] = (coded + filled / 2).checked_div(filled).unwrap_or(0) as u8;
    }
    mean
}

fn write_ground(
    cx: &mut Prepare<'_, TerrainGpu>,
    ground: Texture<TerrainGpu>,
    index: usize,
    bounce: &TerrainBounce,
) {
    let side = bounce.texels();
    cx.write_texture(
        ground,
        wgpu::Origin3d {
            x: 0,
            y: 0,
            z: GROUND_LAYERS * index as u32,
        },
        &bounce.ground_layers(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(side * 4),
            rows_per_image: Some(side),
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: GROUND_LAYERS,
        },
    );
}
