//! Terrain bounce light: each terrain's bounce transfer
//! ([`engine_core::TerrainBounce`]) relit every frame
//! (`terrain_bounce.wgsl`). First its ground's light: every texel of a map
//! over the terrain lit as terrain shades its own ground, the sun through
//! the horizon maps and the clouds' shadows, the sky through its sky map, in
//! its materials' colors as the palette holds them; then the map's levels,
//! each texel the mean of the four below it. Then its probes, in nested
//! levels around the detail eye: the finest a window of the cooked columns,
//! each coarser one twice their spacing and reach, the coarsest holding the
//! whole terrain. A probe gathers through its bins the light the ground
//! sends it, each bin's ground read at its distance from the level its
//! spread spans, into the low-order spherical harmonics of the light
//! reaching it and of the sky it sees; a coarser probe over four of the
//! finer level's is their mean instead, so a coarser level shows what the
//! finer one does where both reach, and the probes of the third level on
//! gather elsewhere through the means of the cooked probes they stand for.
//! Each level's window takes a region of the horizon maps' bounce layers,
//! which hold them for shading (`scene.wgsl`'s `bounce_light`). Nothing of
//! the sun or the sky is cooked: the hour moves and the light follows in the
//! next frame.
//!
//! Each terrain bounce light relights (none of surroundings, ground seen
//! from afar) holds a slot: its bounce layers of the horizon maps and its
//! layers of the ground's maps, while the horizon maps' array layers hold
//! them after every casting terrain's own (`terrain_horizons::room`). Its
//! ground takes the tint of its macro colors as terrain's own ground does
//! from afar. A terrain's ground, probes and their means upload as its
//! transfer arrives and again when it changes; the pipelines are made with
//! the first frame relighting, and in a cooking session as soon as a
//! terrain holds a slot, so the scene's pipeline list holds them wherever
//! the cook's camera stands.

use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Vec2, Vec3};
use engine_core::terrain_data::bounce::{PROBE_BYTES, bin_moments};
use engine_core::{MAX_BOUNCE_LEVELS, RenderTerrain, RenderTerrains, TerrainBounce, TerrainSettings};

use crate::bindings::BOUNCE;
use crate::buffers::Growable;
use crate::cached::{Bound, GroupCache};
use crate::gpu_layouts::bounce::{BounceFrame, BounceTerrain};
use crate::gpu_layouts::constants::BOUNCE_GROUP;
use crate::shaders::entry::terrain_bounce;
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_sky::SkyMaps;

/// Layers of the horizon maps each terrain's bounce light takes: per layer
/// of probes, the light's mean (red, green, blue) and the sky's share
/// (alpha), then their gradients along x, y and z; each level's window in a
/// region of them.
pub(crate) const BOUNCE_LAYERS: usize = 4 * engine_core::terrain_data::bounce::LAYERS;

/// Most levels of probes a terrain relights: as many as its horizon record
/// holds windows of.
pub(crate) const MOST_LEVELS: usize = MAX_BOUNCE_LEVELS as usize;

/// Columns of a level's window that its alignment and lookups give up
/// either way of the eye's: the window starts on an even column, the eye
/// stands anywhere in its own, and a lookup reads the next column too. A
/// window holds every point within its half side less these of the eye.
const WINDOW_MARGIN: f32 = 2.5;

/// Bytes a probe's light takes in the scratch: both layers' four vectors.
const SCRATCH_PROBE_BYTES: u64 = (BOUNCE_LAYERS * 16) as u64;

/// The ground's light: radiance, filtered and written.
const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The ground's surface: control values and normals, loaded.
const GROUND_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
/// Layers of the ground's surface each terrain takes.
const GROUND_LAYERS: u32 = 2;
/// Bindings of the bounce group (`bindings::BOUNCE`).
const BINDINGS: usize = 19;

/// Where a terrain's probes relight this frame: its slot, its finest
/// level's probe columns along a side, the columns along a side of each
/// level's window, its levels, the finest level relit (finer ones reach no
/// point of the terrain), each level's window's first column (x and z,
/// even), the meters around the eye within which the finest level's window
/// holds every point (each coarser one's twice as far), and the levels'
/// regions a row of the bounce layers holds.
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
    /// Level `level`'s window's columns along a side: all of the level's
    /// where they are fewer than the windows'.
    pub(crate) fn window(&self, level: u32) -> u32 {
        self.columns.min(self.probes >> level)
    }

    /// The first texel (x and y) of level `level`'s region of the bounce
    /// layers: regions of the windows' columns a side, in rows.
    fn region(&self, level: u32) -> [u32; 2] {
        [
            level % self.row * self.columns,
            level / self.row * self.columns,
        ]
    }

    /// Bytes the scratch holds of every level's probes.
    fn scratch_bytes(&self) -> u64 {
        (0..self.levels)
            .map(|level| u64::from(self.window(level)).pow(2) * SCRATCH_PROBE_BYTES)
            .sum()
    }
}

/// What bounce light reads of the scene's first group: the lighting
/// uniform, the shadow lookups' uniform (the horizon records' count) and
/// the clouds' shadows.
pub(crate) struct SceneSources<'a> {
    pub(crate) lighting: &'a wgpu::Buffer,
    pub(crate) shadow: (&'static str, Bound<'a>),
    pub(crate) cloud_shadows: &'a wgpu::Buffer,
}

/// What bounce light reads of the terrain pass: its horizon, sky and macro
/// color maps, and its palette.
pub(crate) struct TerrainSources<'a> {
    pub(crate) horizons: &'a Horizons,
    pub(crate) sky: &'a SkyMaps,
    pub(crate) colors: &'a MacroMaps,
    pub(crate) palette: &'a wgpu::Buffer,
}

/// Every slot's ground: its surface, two layers each, and its light, a
/// layer each, with every level and a view of each, `side` texels a side
/// (each terrain's in its first texels).
struct Maps {
    ground: wgpu::Texture,
    ground_view: wgpu::TextureView,
    light_view: wgpu::TextureView,
    levels: Vec<wgpu::TextureView>,
    side: u32,
    slots: usize,
}

pub(crate) struct Bounce {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    frame: wgpu::Buffer,
    // Each level's records, one per slot, which its gathers read; the
    // ground and its light's levels read the finest's.
    terrains: Vec<Growable>,
    blocks: Growable,
    probes: Growable,
    // Each level's probes this frame, which the next coarser one averages.
    scratch: Growable,
    maps: Option<Maps>,
    // Advances when the maps are made anew.
    maps_generation: u64,
    // Stand-ins for the light read and written where a pass writes or reads
    // the light itself, and the light's sampler.
    light_stand_in: wgpu::TextureView,
    out_stand_in: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // Each listed terrain's slot, by terrain index; this frame's levels, by
    // terrain index; and by slot, the transfer held and its probes' means,
    // where its blocks' words, probes and means start; and by level, each
    // slot's record.
    slots: Vec<Option<u32>>,
    windows: Vec<Option<BounceLevels>>,
    held: Vec<Option<Arc<TerrainBounce>>>,
    means: Vec<Vec<u8>>,
    bases: Vec<[u32; 3]>,
    records: Vec<Vec<BounceTerrain>>,
    ground_group: GroupCache<BINDINGS>,
    level_groups: Vec<GroupCache<BINDINGS>>,
    gather_groups: Vec<GroupCache<BINDINGS>>,
    // Each level's widest gather this frame, in invocations along a side
    // (the finest level's a block each); 0 where none relights.
    widest: [u32; MOST_LEVELS],
    /// Terrains relighting this frame, the probes they relight, the most
    /// levels a terrain relights, and the terrains the bounce layers or the
    /// buffers could not hold.
    pub(crate) relit: u32,
    pub(crate) relit_probes: u32,
    pub(crate) levels: u32,
    pub(crate) dropped: u32,
}

impl Bounce {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let frame = crate::buffers::uniform(
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
            crate::targets::texture_2d(device, label, (1, 1), 1, 1, LIGHT_FORMAT, usage)
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
        };
        let buffer = |label, bytes: usize| {
            Growable::new(device, label, bytes as u64, wgpu::BufferUsages::empty())
        };
        Self {
            layout: crate::pipelines::layout(device, &BOUNCE),
            pipelines: None,
            frame,
            terrains: (0..MOST_LEVELS)
                .map(|_| buffer("engine/terrain-bounce-terrains", size_of::<BounceTerrain>()))
                .collect(),
            blocks: buffer("engine/terrain-bounce-blocks", 16),
            probes: buffer("engine/terrain-bounce-probes", 16),
            scratch: buffer("engine/terrain-bounce-scratch", 16),
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

    /// Whether probes relight this frame.
    pub(crate) fn relights(&self) -> bool {
        self.widest.iter().any(|&widest| widest > 0)
    }

    /// This frame's levels, by terrain index: where each terrain's probes
    /// relight, none where they do not; and the slots, which the horizon
    /// maps hold bounce layers for.
    pub(crate) fn windows(&self) -> (&[Option<BounceLevels>], u32) {
        let slots = self.slots.iter().flatten().count();
        (&self.windows, slots as u32)
    }

    /// Relights nothing this frame: no terrain is listed.
    pub(crate) fn rest(&mut self) {
        self.slots.clear();
        self.windows.clear();
        self.widest = [0; MOST_LEVELS];
        self.relit = 0;
        self.relit_probes = 0;
        self.levels = 0;
        self.dropped = 0;
    }

    /// Bytes of the ground's maps and the probes' buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let maps = self.maps.as_ref().map_or(0, |maps| {
            let area = u64::from(maps.side) * u64::from(maps.side) * maps.slots as u64;
            // The surface's two layers of four bytes; the light's eight bytes
            // with a third more for its levels.
            area * 4 * u64::from(GROUND_LAYERS) + area * 8 * 4 / 3
        });
        let records: u64 = self.terrains.iter().map(|records| records.size()).sum();
        maps + records + self.blocks.size() + self.probes.size() + self.scratch.size()
    }

    /// Gives each terrain of `terrains` bounce light relights its slot, in
    /// order, with `settings` asking for bounce light: every one but
    /// surroundings among the first `casting`, while the horizon maps' layers
    /// hold `slots` of them. Places this frame's levels around the detail
    /// eye `eye`: for each terrain with a slot whose transfer is here, as
    /// many as the bounce layers and the probes' buffers hold in order; the
    /// others are dropped.
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

    /// Uploads the ground, probes and means of every terrain with levels
    /// whose transfer arrived or changed since, and writes each level's
    /// records: each slot's terrain's, relighting the levels it relights
    /// where its horizon map is, as the horizon maps' bounce layers hold
    /// them.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (horizons, sky, colors, palette): (&Horizons, &SkyMaps, &MacroMaps, &[u32]),
        cooking: bool,
    ) {
        self.widest = [0; MOST_LEVELS];
        self.relit = 0;
        self.relit_probes = 0;
        self.levels = 0;
        let count = self.slots.iter().flatten().count();
        // A cooking session's pipeline list holds bounce light's pipelines
        // wherever its camera stands.
        if cooking && count > 0 && self.pipelines.is_none() {
            self.pipelines = Some(pipelines(device, &self.layout));
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
                // A transfer whose levels are gone keeps its ground, probes
                // and means until another replaces it.
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
        // Where each relit level's probes start in the scratch.
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
            // The finer level relit: its window's first column, its columns
            // along a side and where its probes start in the scratch.
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
                    // The finest level gathers a block of two by two
                    // columns at a time.
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

    /// Writes every held transfer's blocks' words, probes and their means
    /// into their buffers, one after another in slots' order, noting where
    /// each slot's start.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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

    /// Relights the probes of this frame's levels: the ground's light, its
    /// levels, then each level's gathers, the finest first, into the horizon
    /// maps' bounce layers, timed by `profiler` (`bounce`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
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
        let pipelines = self
            .pipelines
            .get_or_insert_with(|| pipelines(device, &self.layout));
        let generation = self.maps_generation;
        let shared = Shared {
            frame: Bound::fixed(self.frame.as_entire_binding()),
            blocks: self.blocks.bound(),
            probes: self.probes.bound(),
            scratch: self.scratch.bound(),
            surface: Bound::counted(generation, crate::pipelines::view(&maps.ground_view)),
            sky: Bound::counted(
                terrain.sky.generation(),
                crate::pipelines::view(terrain.sky.view()),
            ),
            colors: Bound::counted(
                terrain.colors.generation(),
                crate::pipelines::view(terrain.colors.view()),
            ),
            palette: Bound::fixed(terrain.palette.as_entire_binding()),
            sampler: Bound::fixed(crate::pipelines::sampler(&self.sampler)),
            horizons,
            written,
            scene: [
                ("lighting", Bound::fixed(scene.lighting.as_entire_binding())),
                scene.shadow,
                (
                    "cloud_shadows",
                    Bound::fixed(scene.cloud_shadows.as_entire_binding()),
                ),
            ],
        };
        let level =
            |index: usize| Bound::counted(generation, crate::pipelines::view(&maps.levels[index]));
        let (layout, label) = ((&BOUNCE, &self.layout), "engine/terrain-bounce");
        // The ground and its light's levels read the finest level's records,
        // which every level's repeat the ground in.
        let finest = self.terrains[0].bound();
        let ground = self.ground_group.bind(
            device,
            label,
            layout,
            shared.entries(
                finest.clone(),
                Bound::fixed(crate::pipelines::view(&self.light_stand_in)),
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
        self.gather_groups
            .resize_with(levels, GroupCache::default);
        let mut gathers = Vec::with_capacity(levels);
        for (records, group) in self.terrains.iter().zip(&mut self.gather_groups) {
            gathers.push(group.bind(
                device,
                label,
                layout,
                shared.entries(
                    records.bound(),
                    Bound::counted(generation, crate::pipelines::view(&maps.light_view)),
                    Bound::fixed(crate::pipelines::view(&self.out_stand_in)),
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
        // Each level after the finer ones, whose probes it averages.
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

/// What every bounce group binds alike: the bins' moments, the terrains'
/// blocks, probes and the scratch, their ground, sky and macro color maps,
/// the palette and the light's sampler; the horizon maps' own layers with
/// their records and sampler, and their bounce layers written; the scene's
/// lighting, shadow lookups' uniform and clouds' shadows.
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
    horizons: [(&'static str, Bound<'a>); 3],
    written: Bound<'a>,
    scene: [(&'static str, Bound<'a>); 3],
}

impl<'a> Shared<'a> {
    /// A group's entries, reading the records of a level as `terrains`,
    /// the light as `light_in` and writing it as `light_out`.
    fn entries(
        &self,
        terrains: Bound<'a>,
        light_in: Bound<'a>,
        light_out: Bound<'a>,
    ) -> [(&'static str, Bound<'a>); BINDINGS] {
        let [lighting, shadow, cloud_shadows] = self.scene.clone();
        let [horizons, records, horizon_sampler] = self.horizons.clone();
        [
            lighting,
            shadow,
            horizons,
            records,
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

/// Terrain `index`'s record of `bounce`, at a level its `window` places,
/// relighting none until given the level: its place, its ground in its
/// slot's layers lit with its sky map's and macro colors' layers plus one
/// (0 without them) and its palette's first material, its blocks' words,
/// probes and means from `bases`.
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
        layers: [
            GROUND_LAYERS * slot,
            slot,
            BOUNCE_LAYERS as u32 * slot,
            sky,
        ],
        sources: [bases[0], bases[1], bases[2], index as u32],
        level: [0; 4],
        kept: [0; 4],
        finer: [0; 4],
    }
}

/// `terrain`'s levels of `bounce`'s probes around `eye` in its `slot`, as
/// `settings` ask for them: the windows' columns along a side (the
/// settings', more where the most levels would leave the coarsest short of
/// its level's columns, at most the finest level's), as many levels as it
/// takes for the coarsest to hold every column of its own, each level's
/// window around the eye's column (starting on an even one, held within the
/// terrain), and the finest relit, the first whose reach meets the terrain.
/// None where the levels' regions do not fit the bounce layers, the horizon
/// maps' texels a side.
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
    let mut levels = 1;
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
    // The finer of the columns' spacings along x and z.
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

/// The first of level `level`'s means (from the third level on) among a
/// terrain's means of `probes` columns along a side: after every finer
/// level's.
fn mean_offset(probes: u32, level: u32) -> u32 {
    (2..level).map(|finer| (probes >> finer).pow(2)).sum()
}

/// Bytes of the means of a terrain's probes of `probes` columns along a
/// side: a record for each probe of each level from the third on.
fn mean_bytes(probes: u32) -> usize {
    let levels = (2..u32::BITS).take_while(|&level| probes >> level > 0);
    levels.map(|level| (probes >> level).pow(2) as usize).sum::<usize>() * PROBE_BYTES
}

/// The means of `bounce`'s probes from its third level on, each level's
/// probes in rows along z after the finer level's: each one the mean of the
/// four under it of the level below; the second level's, which they start
/// from, each block's probe, or its four probes' mean.
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

/// The mean of four probes' records: each bin's share of the ground the
/// mean of theirs, and the code of its distance the mean of theirs by how
/// much ground each fills there.
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
        mean[2 * bin + 1] = if filled > 0 {
            ((coded + filled / 2) / filled) as u8
        } else {
            0
        };
    }
    mean
}

/// Writes slot `index`'s ground surface layers of `ground` from `bounce`.
fn write_ground(queue: &wgpu::Queue, ground: &wgpu::Texture, index: usize, bounce: &TerrainBounce) {
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

/// The ground's maps of `slots` slots, `side` texels a side.
fn maps(device: &wgpu::Device, side: u32, slots: usize) -> Maps {
    let layers = slots.max(1) as u32;
    let ground = crate::targets::texture_2d(
        device,
        "engine/terrain-bounce-ground",
        (side, side),
        GROUND_LAYERS * layers,
        1,
        GROUND_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let levels = 32 - side.leading_zeros();
    let light = crate::targets::texture_2d(
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
        side,
        slots,
    }
}

fn pipelines(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 3] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/terrain_bounce.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::BOUNCE_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::BOUNCE_PIPELINES,
        &[layout],
    );
    [
        terrain_bounce::bounce_ground,
        terrain_bounce::bounce_down,
        terrain_bounce::bounce_gather,
    ]
    .map(|entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry))
}
