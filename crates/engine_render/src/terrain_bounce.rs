//! Terrain bounce light: each terrain's bounce transfer
//! ([`engine_core::TerrainBounce`]) relit every frame
//! (`terrain_bounce.wgsl`). First its ground's light: every texel of a map
//! over the terrain lit as terrain shades its own ground, the sun through
//! the horizon maps and the clouds' shadows, the sky through its sky map, in
//! its materials' colors as the palette holds them; then the map's levels,
//! each texel the mean of the four below it. Then each probe within a window
//! of columns around the camera gathers through its bins the light the
//! ground sends it, each bin's ground read at its distance from the level
//! its spread spans, into the low-order spherical harmonics of the light
//! reaching it and of the sky it sees, which the horizon maps' bounce layers
//! hold for shading (`scene.wgsl`'s `bounce_light`). Nothing of the sun or
//! the sky is cooked: the hour moves and the light follows in the next
//! frame.
//!
//! A terrain's ground and probes upload as its transfer arrives and again
//! when it changes; the pipelines are made with the first frame relighting.

use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Vec2, Vec3};
use engine_core::terrain_data::bounce::bin_moments;
use engine_core::{RenderTerrain, RenderTerrains, TerrainBounce, TerrainSettings};

use crate::bindings::BOUNCE;
use crate::buffers::Growable;
use crate::cached::{Bound, GroupCache};
use crate::gpu_layouts::bounce::{BounceFrame, BounceTerrain};
use crate::gpu_layouts::constants::{BOUNCE_GROUP, BOUNCE_LAYERS};
use crate::shaders::entry::terrain_bounce;
use crate::terrain_horizons::Horizons;
use crate::terrain_sky::SkyMaps;

/// The ground's light: radiance, filtered and written.
const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The ground's surface: control values and normals, loaded.
const GROUND_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
/// Layers of the ground's surface each terrain takes.
const GROUND_LAYERS: u32 = 2;
/// Bindings of the bounce group (`bindings::BOUNCE`).
const BINDINGS: usize = 17;

/// Where a terrain's probes relight this frame: its probe columns along a
/// side, the first column of the window (x and z, even) and the window's
/// columns along a side (even).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BounceWindow {
    pub(crate) probes: u32,
    pub(crate) first: [u32; 2],
    pub(crate) columns: u32,
}

/// What bounce light reads of the scene's first group: the lighting
/// uniform, the shadow lookups' uniform (the horizon records' count) and
/// the clouds' shadows.
pub(crate) struct SceneSources<'a> {
    pub(crate) lighting: &'a wgpu::Buffer,
    pub(crate) shadow: (&'static str, Bound<'a>),
    pub(crate) cloud_shadows: &'a wgpu::Buffer,
}

/// What bounce light reads of the terrain pass: its horizon and sky maps,
/// and its palette.
pub(crate) struct TerrainSources<'a> {
    pub(crate) horizons: &'a Horizons,
    pub(crate) sky: &'a SkyMaps,
    pub(crate) palette: &'a wgpu::Buffer,
}

/// Every terrain's ground: its surface, two layers each, and its light, a
/// layer each, with every level and a view of each, `side` texels a side
/// (each terrain's in its first texels).
struct Maps {
    ground: wgpu::Texture,
    ground_view: wgpu::TextureView,
    light_view: wgpu::TextureView,
    levels: Vec<wgpu::TextureView>,
    side: u32,
    terrains: usize,
}

pub(crate) struct Bounce {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    frame: wgpu::Buffer,
    terrains: Growable,
    blocks: Growable,
    probes: Growable,
    maps: Option<Maps>,
    // Advances when the maps are made anew.
    maps_generation: u64,
    // Stand-ins for the light read and written where a pass writes or reads
    // the light itself, and the light's sampler.
    light_stand_in: wgpu::TextureView,
    out_stand_in: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // This frame's windows, each terrain's transfer held and whether its
    // ground uploaded, where its blocks' words and probes start, and each
    // terrain's record.
    windows: Vec<Option<BounceWindow>>,
    held: Vec<Option<Arc<TerrainBounce>>>,
    bases: Vec<[u32; 2]>,
    records: Vec<BounceTerrain>,
    ground_group: GroupCache<BINDINGS>,
    level_groups: Vec<GroupCache<BINDINGS>>,
    gather_group: GroupCache<BINDINGS>,
    // The widest window relighting this frame; 0 while none does.
    widest: u32,
    /// Terrains relighting this frame, and their probe columns.
    pub(crate) relit: u32,
    pub(crate) columns: u32,
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
            terrains: buffer("engine/terrain-bounce-terrains", size_of::<BounceTerrain>()),
            blocks: buffer("engine/terrain-bounce-blocks", 16),
            probes: buffer("engine/terrain-bounce-probes", 16),
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
            windows: Vec::new(),
            held: Vec::new(),
            bases: Vec::new(),
            records: Vec::new(),
            ground_group: GroupCache::default(),
            level_groups: Vec::new(),
            gather_group: GroupCache::default(),
            widest: 0,
            relit: 0,
            columns: 0,
        }
    }

    /// Whether probes relight this frame.
    pub(crate) fn relights(&self) -> bool {
        self.widest > 0
    }

    /// This frame's windows, by terrain index: where each terrain's probes
    /// relight, none where they do not.
    pub(crate) fn windows(&self) -> &[Option<BounceWindow>] {
        &self.windows
    }

    /// Relights nothing this frame: no terrain is listed.
    pub(crate) fn rest(&mut self) {
        self.windows.clear();
        self.widest = 0;
        self.relit = 0;
        self.columns = 0;
    }

    /// Bytes of the ground's maps and the probes' buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let maps = self.maps.as_ref().map_or(0, |maps| {
            let area = u64::from(maps.side) * u64::from(maps.side) * maps.terrains as u64;
            // The surface's two layers of four bytes; the light's eight bytes
            // with a third more for its levels.
            area * 4 * u64::from(GROUND_LAYERS) + area * 8 * 4 / 3
        });
        maps + self.terrains.size() + self.blocks.size() + self.probes.size()
    }

    /// Places this frame's windows around `eye`: for each terrain of
    /// `terrains` whose transfer is here, with `settings` asking for bounce
    /// light, within the settings' reach of the eye, as many as the probes'
    /// buffers hold in order. Surroundings take none.
    pub(crate) fn place(
        &mut self,
        terrains: &RenderTerrains,
        eye: Vec3,
        settings: &TerrainSettings,
    ) {
        self.windows.clear();
        let on = settings.enabled && settings.sun_shadows && settings.bounce;
        let mut room = [self.blocks.room(), self.probes.room()];
        for terrain in &terrains.terrains {
            let placed = terrain
                .bounce
                .as_ref()
                .filter(|_| on && !terrain.surroundings)
                .and_then(|bounce| {
                    let blocks = u64::from(bounce.texels());
                    let needs = [blocks * blocks * 4, bounce.probe_bytes().len() as u64];
                    if needs[0] > room[0] || needs[1] > room[1] {
                        return None;
                    }
                    room = [room[0] - needs[0], room[1] - needs[1]];
                    window(terrain, bounce, eye, settings)
                });
            self.windows.push(placed);
        }
    }

    /// Uploads the ground and probes of every terrain with a window whose
    /// transfer arrived or changed since, and writes each terrain's record:
    /// those with a window and their horizon map relight, as the horizon
    /// maps' bounce layers hold them.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (horizons, sky, palette): (&Horizons, &SkyMaps, &[u32]),
    ) {
        self.widest = 0;
        self.relit = 0;
        self.columns = 0;
        let count = terrains.terrains.len();
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
            .is_none_or(|maps| maps.side < side || maps.terrains != count)
        {
            self.maps = Some(maps(device, side, count));
            self.maps_generation += 1;
            self.held = vec![None; count];
        }
        let Some(maps) = &self.maps else {
            return;
        };
        let mut changed = self.bases.len() != count;
        let listed = terrains.terrains.iter().zip(&self.windows);
        for (index, (terrain, window)) in listed.enumerate() {
            let wanted = terrain.bounce.as_ref().filter(|_| window.is_some());
            let held = &mut self.held[index];
            match (wanted, &*held) {
                (Some(bounce), Some(current)) if Arc::ptr_eq(bounce, current) => {}
                (Some(bounce), _) => {
                    write_ground(queue, &maps.ground, index, bounce);
                    *held = Some(bounce.clone());
                    changed = true;
                }
                // A transfer whose window is gone keeps its ground and
                // probes until another replaces it.
                (None, Some(_)) if terrain.bounce.is_some() => {}
                (None, current) => {
                    changed |= current.is_some();
                    *held = None;
                }
            }
        }
        if changed {
            self.upload(device, queue);
        }
        self.records.clear();
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let relit = match (self.windows[index], &self.held[index]) {
                (Some(window), Some(bounce)) if horizons.held(index) => {
                    self.widest = self.widest.max(window.columns);
                    self.relit += 1;
                    self.columns += window.columns * window.columns;
                    let materials = palette.get(index).copied().unwrap_or(0);
                    terrain_record(terrain, bounce, window, index, self.bases[index])
                        .with_layers(sky.layer(index), materials)
                }
                _ => BounceTerrain::zeroed(),
            };
            self.records.push(relit);
        }
        let bytes = (self.records.len() * size_of::<BounceTerrain>()) as u64;
        self.terrains.grow(device, bytes);
        queue.write_buffer(&self.terrains, 0, bytemuck::cast_slice(&self.records));
    }

    /// Writes every held transfer's blocks' words and probes into their
    /// buffers, one after another in terrains' order, noting where each
    /// terrain's start.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut words: Vec<u32> = Vec::new();
        let mut probes: Vec<u8> = Vec::new();
        self.bases.clear();
        for held in &self.held {
            let probe_base = probes.len() / engine_core::terrain_data::bounce::PROBE_BYTES;
            self.bases.push([words.len() as u32, probe_base as u32]);
            if let Some(bounce) = held {
                words.extend(bounce.block_words());
                probes.extend_from_slice(bounce.probe_bytes());
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

    /// Relights the probes of this frame's windows: the ground's light,
    /// its levels, then the probes' gathers into the horizon maps' bounce
    /// layers, timed by `profiler` (`bounce`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
        (terrain, scene): (TerrainSources<'_>, SceneSources<'_>),
    ) {
        if self.widest == 0 {
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
            terrains: self.terrains.bound(),
            blocks: self.blocks.bound(),
            probes: self.probes.bound(),
            surface: Bound::counted(generation, crate::pipelines::view(&maps.ground_view)),
            sky: Bound::counted(
                terrain.sky.generation(),
                crate::pipelines::view(terrain.sky.view()),
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
        let level = |index: usize| {
            Bound::counted(generation, crate::pipelines::view(&maps.levels[index]))
        };
        let (layout, label) = ((&BOUNCE, &self.layout), "engine/terrain-bounce");
        let ground = self.ground_group.bind(
            device,
            label,
            layout,
            shared.entries(
                Bound::fixed(crate::pipelines::view(&self.light_stand_in)),
                level(0),
            ),
        );
        self.level_groups.resize_with(maps.levels.len() - 1, GroupCache::default);
        let mut downs = Vec::with_capacity(self.level_groups.len());
        for (index, group) in self.level_groups.iter_mut().enumerate() {
            downs.push(group.bind(
                device,
                label,
                layout,
                shared.entries(level(index), level(index + 1)),
            ));
        }
        let gather = self.gather_group.bind(
            device,
            label,
            layout,
            shared.entries(
                Bound::counted(generation, crate::pipelines::view(&maps.light_view)),
                Bound::fixed(crate::pipelines::view(&self.out_stand_in)),
            ),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/terrain-bounce"),
            timestamp_writes: profiler.span("bounce").map(|span| span.compute()),
        });
        let layers = maps.terrains as u32;
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
        pass.set_bind_group(0, gather, &[]);
        let blocks = groups(self.widest / 2);
        pass.dispatch_workgroups(blocks, blocks, layers);
    }
}

/// What every bounce group binds alike: the bins' moments, the terrains'
/// records, blocks and probes, their ground and sky maps, the palette and
/// the light's sampler; the horizon maps' own layers with their records
/// and sampler, and their bounce layers written; the scene's lighting,
/// shadow lookups' uniform and clouds' shadows.
struct Shared<'a> {
    frame: Bound<'a>,
    terrains: Bound<'a>,
    blocks: Bound<'a>,
    probes: Bound<'a>,
    surface: Bound<'a>,
    sky: Bound<'a>,
    palette: Bound<'a>,
    sampler: Bound<'a>,
    horizons: [(&'static str, Bound<'a>); 3],
    written: Bound<'a>,
    scene: [(&'static str, Bound<'a>); 3],
}

impl<'a> Shared<'a> {
    /// A group's entries, reading the light as `light_in` and writing it
    /// as `light_out`.
    fn entries(
        &self,
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
            ("bounce_terrains", self.terrains.clone()),
            ("bounce_sky", self.sky.clone()),
            ("bounce_surface", self.surface.clone()),
            ("bounce_blocks", self.blocks.clone()),
            ("bounce_records", self.probes.clone()),
            ("bounce_light_in", light_in),
            ("bounce_light_out", light_out),
            ("bounce_probes", self.written.clone()),
            ("bounce_sampler", self.sampler.clone()),
            ("bounce_palette", self.palette.clone()),
        ]
    }
}

impl BounceTerrain {
    /// The record with its sky map's first layer plus one (`sky`, 0 without
    /// one) and its palette's first material (`palette`).
    fn with_layers(mut self, sky: u32, palette: u32) -> Self {
        self.layers[3] = sky;
        self.materials[0] = palette;
        self
    }
}

/// Terrain `index`'s record relighting `window` of `bounce`, its blocks'
/// words and probes from `bases`.
fn terrain_record(
    terrain: &RenderTerrain,
    bounce: &TerrainBounce,
    window: BounceWindow,
    index: usize,
    bases: [u32; 2],
) -> BounceTerrain {
    let extent = terrain.pyramid.extent();
    let index = index as u32;
    BounceTerrain {
        place: [terrain.origin.x, terrain.origin.z, extent.x, extent.y],
        grid: [bounce.probes, bounce.texels(), window.first[0], window.first[1]],
        window: [window.columns, bases[0], bases[1], index],
        layers: [GROUND_LAYERS * index, index, BOUNCE_LAYERS as u32 * index, 0],
        materials: [0; 4],
    }
}

/// `terrain`'s window of `bounce`'s probes around `eye`, as `settings`
/// reach: the columns within the reach either way of the eye's (even, at
/// most the horizon maps' texels and the probes'), held within the terrain
/// and starting on a block; none where the eye lies farther than the reach
/// from the terrain.
fn window(
    terrain: &RenderTerrain,
    bounce: &TerrainBounce,
    eye: Vec3,
    settings: &TerrainSettings,
) -> Option<BounceWindow> {
    let probes = bounce.probes;
    let extent = terrain.pyramid.extent();
    let local = Vec2::new(eye.x - terrain.origin.x, eye.z - terrain.origin.z);
    let nearest = Vec2::new(
        engine_core::math::clamp(local.x, 0.0, extent.x),
        engine_core::math::clamp(local.y, 0.0, extent.y),
    );
    let away = local.distance(nearest);
    if away.is_nan() || away > settings.bounce_reach {
        return None;
    }
    let spacing = engine_core::math::max(extent.x, extent.y) / probes as f32;
    let most = probes.min(settings.shadow_resolution) & !1;
    let wanted = (2.0 * settings.bounce_reach / spacing).ceil();
    let columns = if wanted < most as f32 {
        (wanted as u32 + 1) & !1
    } else {
        most
    };
    let columns = columns.clamp(2, most.max(2));
    let first = |along: f32, side: f32| {
        let start = (along / side * probes as f32).floor() - (columns / 2) as f32;
        let last = probes.saturating_sub(columns) as f32;
        (engine_core::math::clamp(start, 0.0, last) as u32) & !1
    };
    Some(BounceWindow {
        probes,
        first: [first(local.x, extent.x), first(local.y, extent.y)],
        columns,
    })
}

/// Writes terrain `index`'s ground surface layers of `ground` from
/// `bounce`.
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

/// The ground's maps of `terrains` terrains, `side` texels a side.
fn maps(device: &wgpu::Device, side: u32, terrains: usize) -> Maps {
    let layers = terrains.max(1) as u32;
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
        terrains,
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
