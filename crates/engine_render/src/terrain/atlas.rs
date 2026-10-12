//! The tile atlas: every resident tile's heights, normals and control
//! values in three texture arrays, a layer each; which tile each layer
//! holds, when it arrived, and the layers free; the GPU's normal maps and
//! control values of tiles uploaded each frame.

use super::*;
use crate::shaders::entry::{terrain_control, terrain_normals};
use std::collections::BTreeMap;

/// The arrays: heights, control values and normals, a layer per tile.
struct Arrays {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    control: wgpu::Texture,
    control_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,
    layers: u32,
}

/// What a sync of the tiles found for the tables: whether a tile replaced
/// under its key changed its hole bits, and the resident tiles with holes,
/// sorted.
pub(super) struct Synced<'t> {
    pub(super) edited: bool,
    pub(super) holed: Vec<((u64, TileKey), &'t Arc<engine_core::Tile>)>,
}

pub(crate) struct TileAtlas {
    arrays: Option<Arrays>,
    /// Advances whenever the arrays are replaced.
    pub(super) generation: u64,
    pub(super) slots: HashMap<(u64, TileKey), u32>,
    // When each resident tile arrived, and when its surface
    // materials did, in seconds of presentation time; tiles
    // without them are absent; and this frame's time.
    pub(super) arrivals: HashMap<(u64, TileKey), f32>,
    pub(super) surfaced: HashMap<(u64, TileKey), f32>,
    pub(super) now: f32,
    // The tile each slot holds: edits replace tiles under the
    // same key, which upload again into their layers.
    pub(super) held: HashMap<(u64, TileKey), std::sync::Arc<engine_core::Tile>>,
    pub(super) free: Vec<u32>,
    pub(super) next_layer: u32,
    /// Tiles revision each terrain's layers were last synced to.
    synced: HashMap<u64, u64>,
    /// Tiles uploaded this frame, resident on the GPU, and dropped for want
    /// of layers.
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    /// Tiles the GPU gave control values this frame.
    pub(crate) controls: u32,
    // Normal maps of tiles uploaded this frame (layer, spacing), the pass
    // computing them, and their sampler.
    normal_jobs: Vec<(u32, f32)>,
    normal_pass: Option<(crate::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    // Control values the GPU gives tiles uploaded this frame,
    // the pass giving them, and each terrain's surface data there with
    // the data it holds, which changes when the data does.
    control_jobs: Vec<ControlJob>,
    control_pass: Option<(crate::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    surfaces: HashMap<u64, (Arc<SurfaceData>, wgpu::Buffer)>,
    pub(super) normal_sampler: wgpu::Sampler,
    pub(super) control_sampler: wgpu::Sampler,
    /// The tiles that arrived, left, were replaced or finished fading in
    /// this frame, whose pages the runtime virtual texture drops, and the
    /// tiles fading in last frame.
    pub(super) changed: Vec<(u64, TileKey)>,
    fading: engine_core::collections::HashSet<(u64, TileKey)>,
}

impl TileAtlas {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            arrays: None,
            generation: 0,
            slots: HashMap::default(),
            arrivals: HashMap::default(),
            surfaced: HashMap::default(),
            now: 0.0,
            held: HashMap::default(),
            free: Vec::new(),
            next_layer: 0,
            synced: HashMap::default(),
            uploads: 0,
            dropped: 0,
            controls: 0,
            normal_jobs: Vec::new(),
            normal_pass: None,
            control_jobs: Vec::new(),
            control_pass: None,
            surfaces: HashMap::default(),
            // Nearest texels: control values gather as stored.
            control_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-control-sampler"),
                ..Default::default()
            }),
            normal_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-normal-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            changed: Vec::new(),
            fading: engine_core::collections::HashSet::default(),
        }
    }

    /// The height array's view, once tiles arrived.
    pub(super) fn heights(&self) -> Option<&wgpu::TextureView> {
        self.arrays.as_ref().map(|arrays| &arrays.view)
    }

    /// The height, normal and control arrays' views, once tiles arrived.
    pub(super) fn views(
        &self,
    ) -> Option<(&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.view, &arrays.normals_view, &arrays.control_view))
    }

    /// Bytes of the height, normal and control arrays.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers) * u64::from(LAYER_SIZE * LAYER_SIZE) * 12
        })
    }

    /// Layers the arrays hold.
    pub(super) fn layers(&self) -> u32 {
        self.arrays.as_ref().map_or(0, |arrays| arrays.layers)
    }

    /// Uploads tiles the GPU lacks and releases those no terrain lists;
    /// `None` when every terrain's tiles stand as last synced.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn sync<'t>(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        terrains: &'t RenderTerrains,
        max_layers: u32,
    ) -> Option<Synced<'t>> {
        let ids: Vec<u64> = terrains.terrains.iter().map(|terrain| terrain.id).collect();
        let current = terrains
            .terrains
            .iter()
            .all(|terrain| self.synced.get(&terrain.id) == Some(&terrain.tiles_revision))
            && self.synced.keys().all(|id| ids.contains(id));
        if current {
            return None;
        }
        self.dropped = 0;
        // Release what no terrain lists any more.
        let mut listed: HashMap<(u64, TileKey), &'t std::sync::Arc<engine_core::Tile>> =
            HashMap::default();
        for terrain in &terrains.terrains {
            for tile in terrain.tiles.iter() {
                listed.insert((terrain.id, tile.key()), tile);
            }
        }
        let mut released = Vec::new();
        self.slots.retain(|key, layer| {
            let keep = listed.contains_key(key);
            if !keep {
                released.push((*key, *layer));
            }
            keep
        });
        // In key order, so which layers tiles take next follows no hash:
        // layers free last are taken first.
        released.sort_unstable();
        for (key, layer) in released {
            self.free.push(layer);
            self.changed.push(key);
        }
        self.held.retain(|key, _| listed.contains_key(key));
        self.arrivals.retain(|key, _| listed.contains_key(key));
        self.surfaced.retain(|key, _| listed.contains_key(key));
        // Terrains whose tiles' control values the GPU gives.
        let surfaces: BTreeMap<u64, &Arc<SurfaceData>> = terrains
            .terrains
            .iter()
            .filter_map(|terrain| Some((terrain.id, terrain.gpu_surface.as_ref()?)))
            .collect();
        self.sync_surfaces(device, queue, &surfaces);
        // Tiles replaced under their keys upload again into their layers; one
        // whose surface materials arrive with it takes them over gradually,
        // and edits change them in place.
        for (key, tile) in terrains.terrains.iter().flat_map(|terrain| {
            terrain
                .tiles
                .iter()
                .map(|tile| ((terrain.id, tile.key()), tile))
        }) {
            let key = &key;
            let had = self.held.get(key).map(|held| held.control().is_some());
            match (had, tile.control().is_some()) {
                (_, false) if !surfaces.contains_key(&key.0) => {
                    self.surfaced.remove(key);
                }
                (Some(false), true) => {
                    self.surfaced.insert(*key, self.now);
                }
                _ => {}
            }
        }
        let replaced: Vec<((u64, TileKey), u32)> = self
            .slots
            .iter()
            .filter(|(key, _)| {
                self.held
                    .get(*key)
                    .is_none_or(|held| !std::sync::Arc::ptr_eq(held, listed[*key]))
            })
            .map(|(key, layer)| (*key, *layer))
            .collect();
        // What they replace, so arrays keep what did not change:
        // a tile arriving whole after its shape, or with its objects
        // standing far away, has the same heights.
        let mut previous: HashMap<(u64, TileKey), std::sync::Arc<engine_core::Tile>> =
            HashMap::default();
        for (key, _) in &replaced {
            self.slots.remove(key);
            if let Some(held) = self.held.remove(key) {
                previous.insert(*key, held);
            }
        }
        let mut reused: HashMap<(u64, TileKey), u32> = replaced.into_iter().collect();
        let pyramids: HashMap<u64, engine_core::terrain_data::Pyramid> = terrains
            .terrains
            .iter()
            .map(|terrain| (terrain.id, terrain.pyramid))
            .collect();
        // Upload the rest, coarse levels first.
        let mut missing: Vec<(u64, TileKey)> = listed
            .keys()
            .filter(|key| !self.slots.contains_key(key))
            .copied()
            .collect();
        missing.sort_unstable_by_key(|(id, key)| (std::cmp::Reverse(key.level), *id, key.x, key.z));
        let needed = self.slots.len() as u32 + missing.len() as u32;
        self.ensure_arrays(device, queue, needed.min(max_layers), max_layers);
        let capacity = self.layers();
        let mut edited = false;
        for key in missing {
            let arriving = !reused.contains_key(&key);
            let layer = match reused.remove(&key).or_else(|| self.free.pop()) {
                Some(layer) => layer,
                None if self.next_layer < capacity => {
                    self.next_layer += 1;
                    self.next_layer - 1
                }
                None => {
                    self.dropped += 1;
                    continue;
                }
            };
            // Tiles arriving grow out of their parents; edited ones
            // change in place.
            if arriving {
                self.arrivals.insert(key, self.now);
                if listed[&key].control().is_some() {
                    self.surfaced.insert(key, self.now);
                }
            }
            // What draws over it changes only as it uploads; tiles finding
            // no layer leave their parents standing in.
            self.changed.push(key);
            let arrays = self.arrays.as_ref().expect("allocated above");
            let tile = listed[&key];
            let before = previous.remove(&key);
            edited |= before
                .as_ref()
                .is_some_and(|held| held.hole_bits() != tile.hole_bits());
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LAYER_SIZE * 4),
                rows_per_image: Some(LAYER_SIZE),
            };
            let extent = wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: 1,
            };
            let place = |texture| wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            };
            let heights = before
                .as_ref()
                .is_none_or(|held| held.heights() != tile.heights());
            if heights {
                queue.write_texture(
                    place(&arrays.texture),
                    bytemuck::cast_slice(tile.heights()),
                    layout,
                    extent,
                );
                self.normal_jobs.push((layer, tile.spacing()));
            }
            // Without control values on a terrain whose surface the GPU
            // reads, the GPU gives them as its heights
            // change, its materials then taking over from the coarser
            // level's.
            let gpu = tile.control().is_none() && surfaces.contains_key(&key.0);
            if gpu && heights {
                let surface = surfaces[&key.0];
                let spacing = tile.spacing();
                let first = surface.first_sample(key.1, spacing);
                self.control_jobs.push(ControlJob {
                    layer,
                    terrain: key.0,
                    first: [first[0] as i32, first[1] as i32],
                    spacing,
                    level: surface.level(spacing),
                });
                self.surfaced.insert(key, self.now);
            }
            // Its control values; a tile without its own yet
            // takes its nearest ancestor's for ground cover, and without a
            // surface, the first material everywhere.
            let control = !gpu
                && (tile.control().is_none()
                    || before
                        .as_ref()
                        .is_none_or(|held| held.control() != tile.control()));
            if control {
                let borrowed;
                let values = match tile.control() {
                    Some(values) => values,
                    None => {
                        borrowed = ancestor_control(&listed, key, pyramids[&key.0]);
                        &borrowed
                    }
                };
                queue.write_texture(
                    place(&arrays.control),
                    bytemuck::cast_slice(values),
                    layout,
                    extent,
                );
            }
            self.slots.insert(key, layer);
            self.held.insert(key, tile.clone());
            self.uploads += u32::from(heights || control);
        }
        self.compute_controls(device, queue);
        // Layers of replaced tiles that found no upload are free again.
        self.free.extend(reused.into_values());
        self.compute_normals(device, queue);
        self.synced = terrains
            .terrains
            .iter()
            .map(|terrain| (terrain.id, terrain.tiles_revision))
            .collect();
        // The tiles with holes the tables hold.
        let mut holed: Vec<((u64, TileKey), &'t Arc<engine_core::Tile>)> = listed
            .into_iter()
            .filter(|(key, tile)| tile.hole_bits().is_some() && self.slots.contains_key(key))
            .collect();
        holed.sort_unstable_by_key(|&(key, _)| key);
        Some(Synced { edited, holed })
    }

    /// How much of a coarser level still shows of what arrived at `at`:
    /// 1 then, 0 [`ARRIVAL_SECONDS`] later.
    pub(super) fn fade_since(&self, at: f32) -> f32 {
        let since = engine_core::math::max(self.now - at, 0.0);
        1.0 - engine_core::math::min(since / ARRIVAL_SECONDS, 1.0)
    }

    /// Each layer's fades, `count` words: how much of the next coarser
    /// level's normals and surface materials still show over its tile's,
    /// 0 to 255 in two bytes, two layers a word. On exact frames (without
    /// `fades`) tiles show as they are, but one without surface materials
    /// yet always shows the coarser level's. Tiles that finished fading in
    /// count as changed.
    pub(super) fn fade_words(&mut self, fades: bool, count: u32) -> Vec<u32> {
        let mut words = vec![0_u32; count as usize];
        let byte = |fade: f32| (engine_core::math::clamp(fade, 0.0, 1.0) * 255.0 + 0.5) as u32;
        // Tiles that finished fading in change what pages over them hold.
        let mut fading = engine_core::collections::HashSet::default();
        #[expect(
            clippy::iter_over_hash_type,
            reason = "each tile's fades land in its own layer's word, and the fading set"
        )]
        for (key, &layer) in &self.slots {
            let normals = match self.arrivals.get(key) {
                Some(&at) if fades => self.fade_since(at),
                _ => 0.0,
            };
            let materials = match self.surfaced.get(key) {
                None => 1.0,
                Some(&at) if fades => self.fade_since(at),
                Some(_) => 0.0,
            };
            if normals > 0.0 || materials > 0.0 {
                fading.insert(*key);
            }
            let Some(word) = words.get_mut(layer as usize / 2) else {
                continue;
            };
            *word |= (byte(normals) | (byte(materials) << 8)) << ((layer % 2) * 16);
        }
        // Sorted: the order pages drop in follows no hash.
        let mut faded: Vec<(u64, TileKey)> = self.fading.difference(&fading).copied().collect();
        faded.sort_unstable();
        self.changed.extend(faded);
        self.fading = fading;
        words
    }

    /// Creates the compute pipelines tiles' normal maps and control values
    /// build with unless they exist.
    pub(super) fn warm(&mut self, device: &wgpu::Device) {
        self.normal_pass(device);
        self.control_pass(device);
    }

    /// Creates the pipeline giving tiles control values and
    /// its group's layout unless they exist.
    fn control_pass(&mut self, device: &wgpu::Device) {
        self.control_pass.get_or_insert_with(|| {
            let layout = crate::pipelines::layout(device, &TERRAIN_CONTROL);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/terrain-control.wgsl"),
                source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_CONTROL_SHADER.into()),
            });
            let pipeline_layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_CONTROL_PIPELINES,
                &[&layout],
            );
            let pipeline = crate::pipelines::compute_entry(
                device,
                &pipeline_layout,
                &module,
                terrain_control::tile_control,
            );
            (layout, pipeline)
        });
    }

    /// Uploads each terrain's surface data for the GPU when
    /// it changes, and releases those of terrains without.
    fn sync_surfaces(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surfaces: &BTreeMap<u64, &Arc<SurfaceData>>,
    ) {
        self.surfaces.retain(|id, _| surfaces.contains_key(id));
        for (&id, &surface) in surfaces {
            if self
                .surfaces
                .get(&id)
                .is_some_and(|(held, _)| Arc::ptr_eq(held, surface))
            {
                continue;
            }
            let bytes = (surface.words.len().max(1) * 4) as u64;
            let buffer = crate::buffers::storage(
                device,
                "engine/terrain-surface",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&surface.words));
            self.surfaces.insert(id, (Arc::clone(surface), buffer));
        }
    }

    /// Gives the tiles uploaded this frame without control values theirs,
    /// after their heights, in a submission of its own.
    fn compute_controls(&mut self, device: &wgpu::Device, queue: &Queue) {
        if self.arrays.is_none() || self.control_jobs.is_empty() {
            self.control_jobs.clear();
            return;
        }
        self.control_pass(device);
        let (Some(arrays), Some((layout, pipeline))) = (&self.arrays, &self.control_pass) else {
            return;
        };
        // One uniform slot per job.
        let mut words = vec![0_u32; self.control_jobs.len() * SLOT as usize / 4];
        for (index, job) in self.control_jobs.iter().enumerate() {
            let at = index * SLOT as usize / 4;
            words[at] = job.layer;
            words[at + 1] = job.level;
            words[at + 2] = job.first[0] as u32;
            words[at + 3] = job.first[1] as u32;
            words[at + 4] = job.spacing.to_bits();
        }
        let jobs = crate::buffers::uniform(
            device,
            "engine/terrain-control-jobs",
            (words.len() * 4) as u64,
        );
        queue.write_buffer(&jobs, 0, bytemuck::cast_slice(&words));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-control"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-control"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (index, job) in self.control_jobs.iter().enumerate() {
                let Some((_, data)) = self.surfaces.get(&job.terrain) else {
                    continue;
                };
                let target = arrays.control.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: job.layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let group = crate::pipelines::group(
                    device,
                    "engine/terrain-control-group",
                    &TERRAIN_CONTROL,
                    layout,
                    &[
                        ("heights", crate::pipelines::view(&arrays.view)),
                        (
                            "job",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &jobs,
                                offset: index as u64 * SLOT,
                                size: wgpu::BufferSize::new(32),
                            }),
                        ),
                        ("data", data.as_entire_binding()),
                        ("control", crate::pipelines::view(&target)),
                    ],
                );
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(
                    LAYER_SIZE.div_ceil(CONTROL_GROUP),
                    LAYER_SIZE.div_ceil(CONTROL_GROUP),
                    1,
                );
            }
        }
        queue.submit_now([encoder.finish()]);
        self.controls += self.control_jobs.len() as u32;
        self.control_jobs.clear();
    }

    /// Creates the pipeline computing tiles' normal maps and its group's
    /// layout unless they exist.
    fn normal_pass(&mut self, device: &wgpu::Device) {
        self.normal_pass.get_or_insert_with(|| {
            let layout = crate::pipelines::layout(device, &TERRAIN_NORMALS);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/terrain-normals.wgsl"),
                source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_NORMALS_SHADER.into()),
            });
            let pipeline_layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_NORMAL_PIPELINES,
                &[&layout],
            );
            let pipeline = crate::pipelines::compute_entry(
                device,
                &pipeline_layout,
                &module,
                terrain_normals::tile_normals,
            );
            (layout, pipeline)
        });
    }

    /// Computes the normal maps of the tiles uploaded this frame, after
    /// their heights, in a submission of its own.
    fn compute_normals(&mut self, device: &wgpu::Device, queue: &Queue) {
        if self.arrays.is_none() {
            self.normal_jobs.clear();
            return;
        }
        if self.normal_jobs.is_empty() {
            return;
        }
        self.normal_pass(device);
        let (Some(arrays), Some((layout, pipeline))) = (&self.arrays, &self.normal_pass) else {
            return;
        };
        // One uniform slot per job.
        let mut words = vec![0_u32; self.normal_jobs.len() * SLOT as usize / 4];
        for (index, &(layer, spacing)) in self.normal_jobs.iter().enumerate() {
            let at = index * SLOT as usize / 4;
            words[at] = layer;
            words[at + 1] = spacing.to_bits();
        }
        let jobs = crate::buffers::uniform(
            device,
            "engine/terrain-normal-jobs",
            (words.len() * 4) as u64,
        );
        queue.write_buffer(&jobs, 0, bytemuck::cast_slice(&words));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-normals"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-normals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (index, &(layer, _)) in self.normal_jobs.iter().enumerate() {
                let target = arrays.normals.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let group = crate::pipelines::group(
                    device,
                    "engine/terrain-normals-group",
                    &TERRAIN_NORMALS,
                    layout,
                    &[
                        ("heights", crate::pipelines::view(&arrays.view)),
                        (
                            "job",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &jobs,
                                offset: index as u64 * SLOT,
                                size: wgpu::BufferSize::new(16),
                            }),
                        ),
                        ("normals", crate::pipelines::view(&target)),
                    ],
                );
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(
                    LAYER_SIZE.div_ceil(NORMALS_GROUP),
                    LAYER_SIZE.div_ceil(NORMALS_GROUP),
                    1,
                );
            }
        }
        queue.submit_now([encoder.finish()]);
        self.normal_jobs.clear();
    }

    /// Grows the arrays to hold `needed` layers, doubling, up to
    /// `max_layers`; existing layers are copied over.
    fn ensure_arrays(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        needed: u32,
        max_layers: u32,
    ) {
        let current = self.layers();
        if needed <= current {
            return;
        }
        let layers = needed.next_power_of_two().max(16).min(max_layers.max(1));
        let array = |label: &'static str, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | usage,
                view_formats: &[],
            })
        };
        let texture = array(
            "engine/terrain-heights",
            HEIGHT_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        let normals = array(
            "engine/terrain-normals",
            NORMAL_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        // The GPU writes control values.
        let control = array(
            "engine/terrain-control",
            CONTROL_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        if let Some(old) = &self.arrays
            && self.next_layer > 0
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/terrain-grow"),
            });
            let extent = wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: self.next_layer,
            };
            encoder.copy_texture_to_texture(
                old.texture.as_image_copy(),
                texture.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(
                old.normals.as_image_copy(),
                normals.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(
                old.control.as_image_copy(),
                control.as_image_copy(),
                extent,
            );
            // Before this frame's tile writes, which the queue runs in order.
            queue.submit_now([encoder.finish()]);
        }
        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        self.arrays = Some(Arrays {
            view: array_view(&texture),
            normals_view: array_view(&normals),
            control_view: array_view(&control),
            texture,
            control,
            normals,
            layers,
        });
        self.generation += 1;
    }
}

/// Control values for tile `key` (of terrain `key.0`, over `pyramid`)
/// without its own yet, for ground cover to grow from: its
/// nearest listed ancestor's holding some, each sample its nearest sample's
/// there; the first material everywhere without one.
fn ancestor_control(
    listed: &HashMap<(u64, TileKey), &std::sync::Arc<engine_core::Tile>>,
    (id, key): (u64, TileKey),
    pyramid: engine_core::terrain_data::Pyramid,
) -> Vec<u32> {
    let size = LAYER_SIZE as usize;
    let mut out = vec![0_u32; size * size];
    let (mut at, mut shift) = (key, 0_u32);
    let control = loop {
        let Some(parent) = pyramid.parent(at) else {
            return out;
        };
        (at, shift) = (parent, shift + 1);
        if let Some(control) = listed.get(&(id, at)).and_then(|tile| tile.control()) {
            break control;
        }
    };
    // Layers keep a border sample before the tile's first: texel `i` holds
    // sample `i - 1`, counted from the level's first.
    let quads = i64::from(engine_core::terrain_data::TILE_QUADS);
    let half = 1_i64 << (shift - 1);
    let nearest = |texel: usize, tile: u16, ancestor: u16| {
        let sample = i64::from(tile) * quads + texel as i64 - 1;
        let there = (sample + half).div_euclid(1 << shift) - i64::from(ancestor) * quads + 1;
        there.clamp(0, size as i64 - 1) as usize
    };
    let columns: Vec<usize> = (0..size).map(|x| nearest(x, key.x, at.x)).collect();
    for (z, row) in out.chunks_exact_mut(size).enumerate() {
        let from = nearest(z, key.z, at.z) * size;
        for (value, &x) in row.iter_mut().zip(&columns) {
            *value = control[from + x];
        }
    }
    out
}

/// A tile the GPU gives control values: its layer, its
/// terrain, its first sample, spacing and the hydrology images' level its
/// masks read.
struct ControlJob {
    layer: u32,
    terrain: u64,
    first: [i32; 2],
    spacing: f32,
    level: u32,
}
