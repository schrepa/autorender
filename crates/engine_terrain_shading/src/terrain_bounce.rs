use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Vec2, Vec3};
use engine_terrain::bounce::{PROBE_BYTES, TerrainBounce, bin_moments};
use engine_terrain::{MAX_BOUNCE_LEVELS, RenderTerrain, RenderTerrains, TerrainSettings};

use crate::bindings::BOUNCE;
use crate::gpu_layouts::bounce::{BounceFrame, BounceTerrain};
use crate::gpu_layouts::constants::BOUNCE_GROUP;
use crate::shaders::entry::terrain_bounce;
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_sky::SkyMaps;
use engine_render::buffers::Growable;
use engine_render::cached::{Bound, GroupCache};

pub(crate) const BOUNCE_LAYERS: usize = 4 * engine_terrain::bounce::LAYERS;

pub(crate) const MOST_LEVELS: usize = MAX_BOUNCE_LEVELS as usize;

const WINDOW_MARGIN: f32 = 2.5;

const SCRATCH_PROBE_BYTES: u64 = (BOUNCE_LAYERS * 16) as u64;

const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GROUND_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
const GROUND_LAYERS: u32 = 2;
const BINDINGS: usize = 17;
const BUFFERS: [&str; 3] = [
    "engine/terrain-bounce-blocks",
    "engine/terrain-bounce-probes",
    "engine/terrain-bounce-scratch",
];

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

pub(crate) struct SceneSources<'a> {
    pub(crate) pieces: Bound<'a>,
    pub(crate) cloud_shadows: (&'static str, Bound<'a>),
}

pub(crate) struct TerrainSources<'a> {
    pub(crate) horizons: &'a Horizons,
    pub(crate) sky: &'a SkyMaps,
    pub(crate) colors: &'a MacroMaps,
    pub(crate) palette: &'a wgpu::Buffer,
}

struct Maps {
    ground: wgpu::Texture,
    ground_view: wgpu::TextureView,
    light: wgpu::Texture,
    light_view: wgpu::TextureView,
    levels: Vec<wgpu::TextureView>,
    side: u32,
    slots: usize,
}

pub(crate) struct Bounce {
    layout: engine_render::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    frame: wgpu::Buffer,
    terrains: Vec<Growable>,
    blocks: Growable,
    probes: Growable,
    scratch: Growable,
    maps: Option<Maps>,
    maps_generation: u64,
    light_stand_in: wgpu::TextureView,
    out_stand_in: wgpu::TextureView,
    sampler: wgpu::Sampler,
    slots: Vec<Option<u32>>,
    windows: Vec<Option<BounceLevels>>,
    held: Vec<Option<Arc<TerrainBounce>>>,
    means: Vec<Vec<u8>>,
    bases: Vec<[u32; 3]>,
    records: Vec<Vec<BounceTerrain>>,
    ground_group: GroupCache<BINDINGS>,
    level_groups: Vec<GroupCache<BINDINGS>>,
    gather_groups: Vec<GroupCache<BINDINGS>>,
    widest: [u32; MOST_LEVELS],
    pub(crate) relit: u32,
    pub(crate) relit_probes: u32,
    pub(crate) levels: u32,
    pub(crate) dropped: u32,
}

impl Bounce {
    pub(crate) fn new(device: &wgpu::Device, queue: &engine_render::gpu_calls::Queue) -> Self {
        let frame = engine_render::buffers::uniform(
            device,
            "engine/terrain-bounce-frame",
            size_of::<BounceFrame>() as u64,
        );
        queue.write_buffer(
            &frame,
            0,
            bytemuck::bytes_of(&BounceFrame {
                bins: bin_moments(),
            }),
        );
        let stand_in = |label, usage| {
            engine_render::targets::texture_2d(device, label, (1, 1), 1, 1, LIGHT_FORMAT, usage)
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
        };
        let [blocks, probes, scratch] = BUFFERS.map(|label| growable(device, label, 16));
        Self {
            layout: engine_render::pipelines::layout(device, &BOUNCE),
            pipelines: None,
            frame,
            terrains: terrain_buffers(device),
            blocks,
            probes,
            scratch,
            maps: None,
            maps_generation: 0,
            light_stand_in: stand_in(
                "engine/terrain-bounce-light-stand-in",
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            out_stand_in: stand_in(
                "engine/terrain-bounce-out-stand-in",
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-bounce-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            slots: Vec::new(),
            windows: Vec::new(),
            held: Vec::new(),
            means: Vec::new(),
            bases: Vec::new(),
            records: Vec::new(),
            ground_group: GroupCache::default(),
            level_groups: Vec::new(),
            gather_groups: Vec::new(),
            widest: [0; MOST_LEVELS],
            relit: 0,
            relit_probes: 0,
            levels: 0,
            dropped: 0,
        }
    }

    pub(crate) fn relights(&self) -> bool {
        self.widest.iter().any(|&widest| widest > 0)
    }

    pub(crate) fn windows(&self) -> (&[Option<BounceLevels>], u32) {
        let slots = self.slots.iter().flatten().count();
        (&self.windows, slots as u32)
    }

    pub(crate) fn rest(&mut self, device: &wgpu::Device) {
        self.slots.clear();
        self.windows.clear();
        self.widest = [0; MOST_LEVELS];
        self.relit = 0;
        self.relit_probes = 0;
        self.levels = 0;
        self.dropped = 0;
        self.release(device);
    }

    fn release(&mut self, device: &wgpu::Device) {
        if self.maps.is_none() && self.pipelines.is_none() {
            return;
        }
        if let Some(maps) = self.maps.take() {
            engine_render::memory::release_any(&maps.ground);
            engine_render::memory::release_any(&maps.light);
        }
        self.pipelines = None;
        self.held.clear();
        self.means.clear();
        self.bases.clear();
        self.records.clear();
        self.ground_group.clear();
        self.level_groups.clear();
        self.gather_groups.clear();
        for buffer in &self.terrains {
            engine_render::memory::release_buffer(buffer);
        }
        self.terrains = terrain_buffers(device);
        for (buffer, label) in [&mut self.blocks, &mut self.probes, &mut self.scratch]
            .into_iter()
            .zip(BUFFERS)
        {
            engine_render::memory::release_buffer(buffer);
            *buffer = growable(device, label, 16);
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let maps = self.maps.as_ref().map_or(0, |maps| {
            let area = u64::from(maps.side) * u64::from(maps.side) * maps.slots as u64;
            area * 4 * u64::from(GROUND_LAYERS) + area * 8 * 4 / 3
        });
        let records: u64 = self.terrains.iter().map(|records| records.size()).sum();
        maps + records + self.blocks.size() + self.probes.size() + self.scratch.size()
    }

    pub(crate) fn place(
        &mut self,
        terrains: &RenderTerrains,
        (eye, settings): (Vec3, &TerrainSettings),
        (casting, slots): (usize, u32),
    ) {
        let on = settings.enabled && settings.sun_shadows && settings.bounce;
        let mut next = 0;
        self.slots.clear();
        self.slots.extend(
            terrains
                .terrains
                .iter()
                .enumerate()
                .map(|(index, terrain)| {
                    let relit = on && !terrain.surroundings && index < casting && next < slots;
                    let slot = relit.then_some(next);
                    next += u32::from(relit);
                    slot
                }),
        );
        self.windows.clear();
        self.dropped = 0;
        let mut room = [self.blocks.room(), self.probes.room(), self.scratch.room()];
        for (terrain, &slot) in terrains.terrains.iter().zip(&self.slots) {
            let Some((slot, bounce)) = slot.zip(terrain.bounce.as_ref()) else {
                self.windows.push(None);
                continue;
            };
            let placed = levels(terrain, (bounce, slot), eye, settings).filter(|levels| {
                let blocks = u64::from(bounce.texels());
                let records = bounce.probe_bytes().len() + mean_bytes(bounce.probes);
                let needs = [blocks * blocks * 4, records as u64, levels.scratch_bytes()];
                let fits = needs.iter().zip(&room).all(|(need, left)| need <= left);
                if fits {
                    room = [room[0] - needs[0], room[1] - needs[1], room[2] - needs[2]];
                }
                fits
            });
            self.dropped += u32::from(placed.is_none());
            self.windows.push(placed);
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        terrains: &RenderTerrains,
        (horizons, sky, colors, palette): (&Horizons, &SkyMaps, &MacroMaps, &[u32]),
        cooking: bool,
    ) {
        self.widest = [0; MOST_LEVELS];
        self.relit = 0;
        self.relit_probes = 0;
        self.levels = 0;
        let count = self.slots.iter().flatten().count();
        if count == 0 {
            self.release(device);
            return;
        }
        if cooking && self.pipelines.is_none() {
            self.pipelines = Some(pipelines(device));
        }
        let side = terrains
            .terrains
            .iter()
            .zip(&self.windows)
            .filter_map(|(terrain, window)| window.and(terrain.bounce.as_ref()))
            .map(|bounce| bounce.texels())
            .max();
        let Some(side) = side.filter(|_| horizons.bounces()) else {
            return;
        };
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.side < side || maps.slots != count)
        {
            self.maps = Some(maps(device, side, count));
            self.maps_generation += 1;
            self.held = vec![None; count];
            self.means = vec![Vec::new(); count];
        }
        let Some(maps) = &self.maps else {
            return;
        };
        let mut changed = self.bases.len() != count;
        let listed = terrains.terrains.iter().zip(&self.windows);
        for ((terrain, window), &slot) in listed.zip(&self.slots) {
            let Some(slot) = slot else {
                continue;
            };
            let wanted = terrain.bounce.as_ref().filter(|_| window.is_some());
            let held = &mut self.held[slot as usize];
            match (wanted, &*held) {
                (Some(bounce), Some(current)) if Arc::ptr_eq(bounce, current) => {}
                (Some(bounce), _) => {
                    write_ground(queue, &maps.ground, slot as usize, bounce);
                    self.means[slot as usize] = means(bounce);
                    *held = Some(bounce.clone());
                    changed = true;
                }
                (None, Some(_)) if terrain.bounce.is_some() => {}
                (None, current) => {
                    changed |= current.is_some();
                    *held = None;
                    self.means[slot as usize].clear();
                }
            }
        }
        if changed {
            self.upload(device, queue);
        }
        let levels = self.windows.iter().flatten().map(|window| window.levels);
        let levels = levels.max().unwrap_or(1) as usize;
        self.records.resize_with(levels, Vec::new);
        for records in &mut self.records {
            records.clear();
            records.resize(count, BounceTerrain::zeroed());
        }
        let mut scratch = 0_u32;
        let listed = terrains.terrains.iter().zip(&self.windows).enumerate();
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
                (window, index),
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
        for (records, buffer) in self.records.iter().zip(&mut self.terrains) {
            let bytes = (records.len() * size_of::<BounceTerrain>()) as u64;
            buffer.grow(device, bytes);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(records));
        }
        self.scratch
            .grow(device, u64::from(scratch) * SCRATCH_PROBE_BYTES);
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &engine_render::gpu_calls::Queue) {
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
        self.blocks.grow(device, (words.len() * 4) as u64);
        self.probes.grow(device, probes.len() as u64);
        if !words.is_empty() {
            queue.write_buffer(&self.blocks, 0, bytemuck::cast_slice(&words));
        }
        if !probes.is_empty() {
            queue.write_buffer(&self.probes, 0, &probes);
        }
    }

    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &engine_render::profiling::GpuProfiler,
        (terrain, scene): (TerrainSources<'_>, SceneSources<'_>),
    ) {
        if !self.relights() {
            return;
        }
        let (Some(maps), Some((horizons, written))) =
            (&self.maps, terrain.horizons.bounce_entries())
        else {
            return;
        };
        let pipelines = self.pipelines.get_or_insert_with(|| pipelines(device));
        let generation = self.maps_generation;
        let shared = Shared {
            frame: Bound::fixed(self.frame.as_entire_binding()),
            blocks: self.blocks.bound(),
            probes: self.probes.bound(),
            scratch: self.scratch.bound(),
            surface: Bound::counted(
                generation,
                engine_render::pipelines::view(&maps.ground_view),
            ),
            sky: Bound::counted(
                terrain.sky.generation(),
                engine_render::pipelines::view(terrain.sky.view()),
            ),
            colors: Bound::counted(
                terrain.colors.generation(),
                engine_render::pipelines::view(terrain.colors.view()),
            ),
            palette: Bound::fixed(terrain.palette.as_entire_binding()),
            sampler: Bound::fixed(engine_render::pipelines::sampler(&self.sampler)),
            horizons,
            written,
            scene: [("pieces", scene.pieces), scene.cloud_shadows],
        };
        let level = |index: usize| {
            Bound::counted(
                generation,
                engine_render::pipelines::view(&maps.levels[index]),
            )
        };
        let (layout, label) = ((&BOUNCE, &self.layout), "engine/terrain-bounce");
        let finest = self.terrains[0].bound();
        let ground = self.ground_group.bind(
            device,
            label,
            layout,
            shared.entries(
                finest.clone(),
                Bound::fixed(engine_render::pipelines::view(&self.light_stand_in)),
                level(0),
            ),
        );
        self.level_groups
            .resize_with(maps.levels.len() - 1, GroupCache::default);
        let mut downs = Vec::with_capacity(self.level_groups.len());
        for (index, group) in self.level_groups.iter_mut().enumerate() {
            downs.push(group.bind(
                device,
                label,
                layout,
                shared.entries(finest.clone(), level(index), level(index + 1)),
            ));
        }
        let levels = self.widest.iter().rposition(|&widest| widest > 0);
        let levels = levels.map_or(0, |last| last + 1);
        self.gather_groups.resize_with(levels, GroupCache::default);
        let mut gathers = Vec::with_capacity(levels);
        for (records, group) in self.terrains.iter().zip(&mut self.gather_groups) {
            gathers.push(group.bind(
                device,
                label,
                layout,
                shared.entries(
                    records.bound(),
                    Bound::counted(generation, engine_render::pipelines::view(&maps.light_view)),
                    Bound::fixed(engine_render::pipelines::view(&self.out_stand_in)),
                ),
            ));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/terrain-bounce"),
            timestamp_writes: profiler.span("bounce").map(|span| span.compute()),
        });
        let layers = maps.slots as u32;
        let groups = |texels: u32| texels.max(1).div_ceil(BOUNCE_GROUP);
        pass.set_pipeline(&pipelines[0]);
        pass.set_bind_group(0, ground, &[]);
        pass.dispatch_workgroups(groups(maps.side), groups(maps.side), layers);
        pass.set_pipeline(&pipelines[1]);
        for (index, group) in downs.iter().enumerate() {
            let side = maps.side >> (index + 1);
            pass.set_bind_group(0, *group, &[]);
            pass.dispatch_workgroups(groups(side), groups(side), layers);
        }
        pass.set_pipeline(&pipelines[2]);
        for (group, &widest) in gathers.iter().zip(&self.widest) {
            if widest == 0 {
                continue;
            }
            pass.set_bind_group(0, *group, &[]);
            pass.dispatch_workgroups(groups(widest), groups(widest), layers);
        }
    }
}

struct Shared<'a> {
    frame: Bound<'a>,
    blocks: Bound<'a>,
    probes: Bound<'a>,
    scratch: Bound<'a>,
    surface: Bound<'a>,
    sky: Bound<'a>,
    colors: Bound<'a>,
    palette: Bound<'a>,
    sampler: Bound<'a>,
    horizons: [(&'static str, Bound<'a>); 2],
    written: Bound<'a>,
    scene: [(&'static str, Bound<'a>); 2],
}

impl<'a> Shared<'a> {
    fn entries(
        &self,
        terrains: Bound<'a>,
        light_in: Bound<'a>,
        light_out: Bound<'a>,
    ) -> [(&'static str, Bound<'a>); BINDINGS] {
        let [pieces, cloud_shadows] = self.scene.clone();
        let [horizons, horizon_sampler] = self.horizons.clone();
        [
            pieces,
            horizons,
            horizon_sampler,
            cloud_shadows,
            ("bounce_frame", self.frame.clone()),
            ("bounce_terrains", terrains),
            ("bounce_sky", self.sky.clone()),
            ("bounce_surface", self.surface.clone()),
            ("bounce_blocks", self.blocks.clone()),
            ("bounce_records", self.probes.clone()),
            ("bounce_light_in", light_in),
            ("bounce_light_out", light_out),
            ("bounce_probes", self.written.clone()),
            ("bounce_sampler", self.sampler.clone()),
            ("bounce_palette", self.palette.clone()),
            ("bounce_macro", self.colors.clone()),
            ("bounce_scratch", self.scratch.clone()),
        ]
    }
}

fn terrain_record(
    terrain: &RenderTerrain,
    bounce: &TerrainBounce,
    (window, index): (&BounceLevels, usize),
    bases: [u32; 3],
    ((sky, colors), palette): ((u32, u32), u32),
) -> BounceTerrain {
    let extent = terrain.pyramid.extent();
    let slot = window.slot;
    BounceTerrain {
        place: [terrain.origin.x, terrain.origin.z, extent.x, extent.y],
        grid: [bounce.probes, bounce.texels(), palette, colors],
        layers: [GROUND_LAYERS * slot, slot, BOUNCE_LAYERS as u32 * slot, sky],
        sources: [bases[0], bases[1], bases[2], index as u32],
        level: [0; 4],
        kept: [0; 4],
        finer: [0; 4],
    }
}

fn levels(
    terrain: &RenderTerrain,
    (bounce, slot): (&TerrainBounce, u32),
    eye: Vec3,
    settings: &TerrainSettings,
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
        engine_core::math::clamp(local.x, 0.0, extent.x),
        engine_core::math::clamp(local.y, 0.0, extent.y),
    );
    let away = local.distance(nearest);
    let spacing = engine_core::math::min(extent.x, extent.y) / probes as f32;
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
            (engine_core::math::clamp(start, 0.0, last) as u32) & !1
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
    queue: &engine_render::gpu_calls::Queue,
    ground: &wgpu::Texture,
    index: usize,
    bounce: &TerrainBounce,
) {
    let side = bounce.texels();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: ground,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: GROUND_LAYERS * index as u32,
            },
            aspect: wgpu::TextureAspect::All,
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

fn maps(device: &wgpu::Device, side: u32, slots: usize) -> Maps {
    let layers = slots.max(1) as u32;
    let ground = engine_render::targets::texture_2d(
        device,
        "engine/terrain-bounce-ground",
        (side, side),
        GROUND_LAYERS * layers,
        1,
        GROUND_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let levels = 32 - side.leading_zeros();
    let light = engine_render::targets::texture_2d(
        device,
        "engine/terrain-bounce-light",
        (side, side),
        layers,
        levels,
        LIGHT_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
    );
    let array = |base: u32, count: Option<u32>| wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-bounce-light"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: base,
        mip_level_count: count,
        ..Default::default()
    };
    Maps {
        ground_view: ground.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/terrain-bounce-ground"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        }),
        ground,
        light_view: light.create_view(&array(0, None)),
        levels: (0..levels)
            .map(|level| light.create_view(&array(level, Some(1))))
            .collect(),
        light,
        side,
        slots,
    }
}

fn growable(device: &wgpu::Device, label: &'static str, bytes: usize) -> Growable {
    Growable::new(device, label, bytes as u64, wgpu::BufferUsages::empty())
}

fn terrain_buffers(device: &wgpu::Device) -> Vec<Growable> {
    (0..MOST_LEVELS)
        .map(|_| growable(device, "engine/terrain-bounce-terrains", size_of::<BounceTerrain>()))
        .collect()
}

fn pipelines(device: &wgpu::Device) -> [wgpu::ComputePipeline; 3] {
    let pipeline_layout =
        engine_render::pipelines::layout_of(device, &crate::pipeline_layouts::BOUNCE_PIPELINES);
    [
        terrain_bounce::bounce_ground,
        terrain_bounce::bounce_down,
        terrain_bounce::bounce_gather,
    ]
    .map(|entry| engine_render::pipelines::compute_entry(device, &pipeline_layout, entry))
}
