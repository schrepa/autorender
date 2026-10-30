#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before tiles upload"
)]

use super::*;
use crate::gpu_layouts::constants::ERROR_MIPS;
use crate::shaders::entry::{terrain_control, terrain_normals};
use engine_core::collections::HashSet;
use engine_terrain::{ERROR_CELLS, ERROR_TEXELS, ErrorPyramid, Pyramid, TileShading};
use std::collections::{BTreeMap, BTreeSet};

const ERROR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

struct Arrays {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    control: wgpu::Texture,
    control_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,
    rivers: Option<(wgpu::Texture, wgpu::TextureView)>,
    ground_errors: wgpu::Texture,
    ground_errors_view: wgpu::TextureView,
    water_errors: wgpu::Texture,
    water_errors_view: wgpu::TextureView,
    layers: u32,
}

const HEIGHT_MIPS: u32 = 1 + ERROR_MIPS;

const _: () = assert!(
    LAYER_SIZE >= 2 * ERROR_CELLS && LAYER_SIZE >> ERROR_MIPS >= 1,
    "height_mips_hold_peak_mips"
);

struct Listed {
    revision: u64,
    tiles: Arc<[Arc<engine_terrain::Tile>]>,
    surface: bool,
}

#[derive(Default)]
struct Changes {
    left: Vec<(u64, TileKey)>,
    arrived: Vec<(u64, TileKey)>,
    swapped: Vec<(u64, TileKey)>,
}

impl Changes {
    fn compare(
        &mut self,
        id: u64,
        before: &[Arc<engine_terrain::Tile>],
        now: &[Arc<engine_terrain::Tile>],
    ) {
        let (mut old, mut new) = (0, 0);
        loop {
            match (before.get(old), now.get(new)) {
                (Some(was), Some(is)) => match was.key().cmp(&is.key()) {
                    std::cmp::Ordering::Less => {
                        self.left.push((id, was.key()));
                        old += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        self.arrived.push((id, is.key()));
                        new += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        if !Arc::ptr_eq(was, is) {
                            self.swapped.push((id, is.key()));
                        }
                        (old, new) = (old + 1, new + 1);
                    }
                },
                (Some(was), None) => {
                    self.left.push((id, was.key()));
                    old += 1;
                }
                (None, Some(is)) => {
                    self.arrived.push((id, is.key()));
                    new += 1;
                }
                (None, None) => return,
            }
        }
    }
}

#[derive(Default)]
struct Replaced {
    layers: HashMap<(u64, TileKey), u32>,
    tiles: HashMap<(u64, TileKey), Arc<engine_terrain::Tile>>,
}

pub(crate) struct TileAtlas {
    arrays: Option<Arrays>,
    pub(super) generation: u64,
    pub(super) slots: HashMap<(u64, TileKey), u32>,
    pub(super) arrivals: HashMap<(u64, TileKey), f32>,
    pub(super) surfaced: HashMap<(u64, TileKey), f32>,
    pub(super) now: f32,
    pub(super) shaded: HashSet<(u64, TileKey)>,
    pub(super) held: HashMap<(u64, TileKey), Arc<engine_terrain::Tile>>,
    pub(super) tops: HashMap<(u64, TileKey), f32>,
    pub(super) free: Vec<u32>,
    pub(super) next_layer: u32,
    synced: HashMap<u64, Listed>,
    pub(super) unplaced: HashSet<(u64, TileKey)>,
    pub(super) holed: BTreeSet<(u64, TileKey)>,
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    pub(crate) controls: u32,
    normal_jobs: Vec<(u32, f32)>,
    normal_pass: Option<(engine_render::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    control_jobs: Vec<ControlJob>,
    control_pass: Option<(engine_render::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    gpu_materials: bool,
    surfaces: HashMap<u64, (Arc<SurfaceData>, wgpu::Buffer)>,
    pub(super) sky_sampler: wgpu::Sampler,
    pub(super) control_sampler: wgpu::Sampler,
    pub(super) changed: Vec<(u64, TileKey)>,
    pub(super) held_changes: Vec<(u64, TileKey)>,
    pub(super) fades: Vec<u32>,
    pub(super) fades_dirty: Range<usize>,
    pub(super) faded_with: Option<(bool, f32)>,
    moving: HashSet<(u64, TileKey)>,
    touched: Vec<(u64, TileKey)>,
    vacated: Vec<u32>,
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
            shaded: HashSet::default(),
            held: HashMap::default(),
            tops: HashMap::default(),
            free: Vec::new(),
            next_layer: 0,
            synced: HashMap::default(),
            unplaced: HashSet::default(),
            holed: BTreeSet::new(),
            uploads: 0,
            dropped: 0,
            controls: 0,
            normal_jobs: Vec::new(),
            normal_pass: None,
            control_jobs: Vec::new(),
            control_pass: None,
            gpu_materials: false,
            surfaces: HashMap::default(),
            control_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-control-sampler"),
                ..Default::default()
            }),
            sky_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-sky-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            changed: Vec::new(),
            held_changes: Vec::new(),
            fades: Vec::new(),
            fades_dirty: 0..0,
            faded_with: None,
            moving: HashSet::default(),
            touched: Vec::new(),
            vacated: Vec::new(),
        }
    }

    pub(super) fn views(
        &self,
    ) -> Option<(&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.view, &arrays.normals_view, &arrays.control_view))
    }

    pub(super) fn errors(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.ground_errors_view, &arrays.water_errors_view))
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            let pyramids = 2 * 2 * ERROR_TEXELS as u64;
            let peaks: u64 = (1..HEIGHT_MIPS)
                .map(|mip| u64::from(LAYER_SIZE >> mip).pow(2) * 4)
                .sum();
            let texel = if arrays.rivers.is_some() { 20 } else { 12 };
            u64::from(arrays.layers)
                * (u64::from(LAYER_SIZE * LAYER_SIZE) * texel + peaks + pyramids)
        })
    }

    pub(super) fn follow_materials(&mut self, device: &wgpu::Device, terrains: &RenderTerrains) {
        let gpu = (terrains.terrains.iter()).any(|terrain| terrain.gpu_surface.is_some());
        self.gpu_materials = gpu;
        if !gpu {
            self.control_pass = None;
        }
        let Some(arrays) = &mut self.arrays else {
            return;
        };
        if gpu {
            if arrays.rivers.is_none() {
                arrays.rivers = Some(rivers_array(device, arrays.layers));
            }
        } else if let Some((rivers, _)) = arrays.rivers.take() {
            engine_render::memory::release_any(&rivers);
        }
    }

    pub(super) fn layers(&self) -> u32 {
        self.arrays.as_ref().map_or(0, |arrays| arrays.layers)
    }

    pub(super) fn begin(&mut self, now: f32) {
        (self.uploads, self.controls, self.now) = (0, 0, now);
        self.changed.clear();
        self.held_changes.clear();
    }

    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        terrains: &RenderTerrains,
        max_layers: u32,
    ) -> Option<bool> {
        let listed = |id: &u64| terrains.terrains.iter().any(|terrain| terrain.id == *id);
        let current = terrains.terrains.iter().all(|terrain| {
            self.synced
                .get(&terrain.id)
                .is_some_and(|synced| synced.revision == terrain.tiles_revision)
        }) && self.synced.keys().all(listed);
        if current {
            return None;
        }
        self.dropped = 0;
        let mut changes = self.changes(terrains);
        self.release(&mut changes.left);
        let surfaces: BTreeMap<u64, &Arc<SurfaceData>> = terrains
            .terrains
            .iter()
            .filter_map(|terrain| Some((terrain.id, terrain.gpu_surface.as_ref()?)))
            .collect();
        self.sync_surfaces(device, queue, &surfaces);
        self.surface_times(terrains, &surfaces, &changes);
        let mut replaced = self.replace(queue, terrains, &changes.swapped);
        let mut missing: Vec<(u64, TileKey)> = (self.unplaced.iter().copied())
            .chain(changes.arrived)
            .chain(replaced.layers.keys().copied())
            .collect();
        missing.sort_unstable_by_key(|(id, key)| (std::cmp::Reverse(key.level), *id, key.x, key.z));
        missing.dedup();
        let needed = self.slots.len() as u32 + missing.len() as u32;
        self.ensure_arrays(device, queue, needed.min(max_layers), max_layers);
        let capacity = self.layers();
        let mut edited = false;
        for key in missing {
            let terrain = terrains.terrains.iter().find(|terrain| terrain.id == key.0);
            let Some(terrain) = terrain else {
                continue;
            };
            let Some(tile) = listed_tile(&terrain.tiles, key.1) else {
                continue;
            };
            let room = (&surfaces, capacity);
            edited |= self.upload(queue, (key, tile), terrain, &mut replaced, room);
        }
        self.compute_controls(device, queue);
        self.free.extend(replaced.layers.into_values());
        self.compute_normals(device, queue);
        self.synced.retain(|id, _| listed(id));
        for terrain in &terrains.terrains {
            let synced = Listed {
                revision: terrain.tiles_revision,
                tiles: Arc::clone(&terrain.tiles),
                surface: terrain.gpu_surface.is_some(),
            };
            self.synced.insert(terrain.id, synced);
        }
        Some(edited)
    }

    fn changes(&self, terrains: &RenderTerrains) -> Changes {
        let mut changes = Changes::default();
        for terrain in &terrains.terrains {
            let synced = self.synced.get(&terrain.id);
            if synced.is_some_and(|synced| synced.revision == terrain.tiles_revision) {
                continue;
            }
            let before: &[Arc<engine_terrain::Tile>] = match synced {
                Some(synced) => &synced.tiles[..],
                None => &[],
            };
            changes.compare(terrain.id, before, &terrain.tiles);
        }
        let mut gone: Vec<u64> = (self.synced.keys())
            .filter(|id| !terrains.terrains.iter().any(|terrain| terrain.id == **id))
            .copied()
            .collect();
        gone.sort_unstable();
        for id in gone {
            if let Some(synced) = self.synced.get(&id) {
                (changes.left).extend(synced.tiles.iter().map(|tile| (id, tile.key())));
            }
        }
        changes
    }

    fn release(&mut self, left: &mut [(u64, TileKey)]) {
        left.sort_unstable();
        for &key in left.iter() {
            self.unplaced.remove(&key);
            let Some(layer) = self.slots.remove(&key) else {
                continue;
            };
            self.free.push(layer);
            self.changed.push(key);
            self.held_changes.push(key);
            self.touched.push(key);
            self.vacated.push(layer);
            self.held.remove(&key);
            self.tops.remove(&key);
            self.arrivals.remove(&key);
            self.surfaced.remove(&key);
            self.shaded.remove(&key);
            self.holed.remove(&key);
        }
    }

    fn surface_times(
        &mut self,
        terrains: &RenderTerrains,
        surfaces: &BTreeMap<u64, &Arc<SurfaceData>>,
        changes: &Changes,
    ) {
        for terrain in &terrains.terrains {
            let stopped = !surfaces.contains_key(&terrain.id)
                && (self.synced.get(&terrain.id)).is_some_and(|synced| synced.surface);
            if !stopped {
                continue;
            }
            let unsurfaced = |tile: &&Arc<engine_terrain::Tile>| {
                tile.control().is_none()
                    && !TileShading::shades(tile.key().level, terrain.surroundings)
            };
            for tile in terrain.tiles.iter().filter(unsurfaced) {
                self.unsurface((terrain.id, tile.key()));
            }
        }
        for &key in changes.arrived.iter().chain(&changes.swapped) {
            let terrain = terrains.terrains.iter().find(|terrain| terrain.id == key.0);
            let Some(terrain) = terrain else {
                continue;
            };
            let Some(tile) = listed_tile(&terrain.tiles, key.1) else {
                continue;
            };
            if TileShading::shades(key.1.level, terrain.surroundings) {
                continue;
            }
            let had = self.held.get(&key).map(|held| held.control().is_some());
            match (had, tile.control().is_some()) {
                (_, false) if !surfaces.contains_key(&key.0) => self.unsurface(key),
                (Some(false), true) => {
                    self.surfaced.insert(key, self.now);
                    self.touched.push(key);
                }
                _ => {}
            }
        }
    }

    fn unsurface(&mut self, key: (u64, TileKey)) {
        if self.surfaced.remove(&key).is_some() {
            self.touched.push(key);
        }
    }

    fn replace(
        &mut self,
        queue: &Queue,
        terrains: &RenderTerrains,
        swapped: &[(u64, TileKey)],
    ) -> Replaced {
        let mut replaced = Replaced::default();
        for &key in swapped {
            let (Some(&layer), Some(tile)) = (self.slots.get(&key), listed_in(terrains, key))
            else {
                continue;
            };
            let same_ground = self.held.get(&key).is_some_and(|held| {
                held.heights() == tile.heights()
                    && held.hole_bits() == tile.hole_bits()
                    && held.control() == tile.control()
                    && held.river_edges() == tile.river_edges()
                    && held.river_beds() == tile.river_beds()
                    && same_shading(held.shading(), tile.shading())
            });
            if same_ground {
                let watered = (self.held.get(&key))
                    .is_some_and(|held| !same_errors(held.water_errors(), tile.water_errors()));
                if let Some(arrays) = &self.arrays
                    && watered
                {
                    write_errors(queue, &arrays.water_errors, layer, tile.water_errors());
                }
                self.held.insert(key, Arc::clone(tile));
                self.held_changes.push(key);
                continue;
            }
            self.slots.remove(&key);
            replaced.layers.insert(key, layer);
            if let Some(held) = self.held.remove(&key) {
                replaced.tiles.insert(key, held);
            }
        }
        replaced
    }

    fn upload(
        &mut self,
        queue: &Queue,
        (key, tile): ((u64, TileKey), &Arc<engine_terrain::Tile>),
        terrain: &RenderTerrain,
        replaced: &mut Replaced,
        (surfaces, capacity): (&BTreeMap<u64, &Arc<SurfaceData>>, u32),
    ) -> bool {
        let arriving = !replaced.layers.contains_key(&key);
        let layer = match replaced.layers.remove(&key).or_else(|| self.free.pop()) {
            Some(layer) => layer,
            None if self.next_layer < capacity => {
                self.next_layer += 1;
                self.next_layer - 1
            }
            None => {
                self.dropped += 1;
                self.unplaced.insert(key);
                return false;
            }
        };
        self.unplaced.remove(&key);
        let shades = TileShading::shades(key.1.level, terrain.surroundings);
        if arriving {
            self.arrivals.insert(key, self.now);
            if tile.control().is_some() && !shades {
                self.surfaced.insert(key, self.now);
            }
        }
        self.changed.push(key);
        self.held_changes.push(key);
        self.touched.push(key);
        let arrays = self.arrays.as_ref().expect("arrays_allocated");
        let before = replaced.tiles.remove(&key);
        let edited = (before.as_ref()).is_some_and(|held| held.hole_bits() != tile.hole_bits());
        let heights = (before.as_ref()).is_none_or(|held| held.heights() != tile.heights());
        if heights {
            let texels = bytemuck::cast_slice(tile.heights());
            write_layer(queue, &arrays.texture, layer, (texels, 4));
            write_errors(queue, &arrays.ground_errors, layer, tile.ground_errors());
            let peaks = peaks(tile.heights());
            write_peaks(queue, &arrays.texture, layer, &peaks);
            self.tops.insert(key, peaks_top(&peaks));
            if !shades {
                self.normal_jobs.push((layer, tile.spacing()));
            }
        }
        let water = (before.as_ref())
            .is_none_or(|held| !same_errors(held.water_errors(), tile.water_errors()));
        if water {
            write_errors(queue, &arrays.water_errors, layer, tile.water_errors());
        }
        let placed = ((key, layer), &**tile, before.as_ref());
        let control = if shades {
            self.shaded.insert(key);
            self.shade(queue, placed, terrain)
        } else {
            self.give_control(queue, placed, (terrain, surfaces), heights)
        };
        self.slots.insert(key, layer);
        self.held.insert(key, Arc::clone(tile));
        if tile.hole_bits().is_some() {
            self.holed.insert(key);
        } else {
            self.holed.remove(&key);
        }
        self.uploads += u32::from(heights || control);
        edited
    }

    fn give_control(
        &mut self,
        queue: &Queue,
        ((key, layer), tile, before): Uploading<'_>,
        (terrain, surfaces): (&RenderTerrain, &BTreeMap<u64, &Arc<SurfaceData>>),
        heights: bool,
    ) -> bool {
        let arrays = self.arrays.as_ref().expect("arrays_allocated");
        let gpu = tile.control().is_none() && surfaces.contains_key(&key.0);
        let rivers = before.is_none_or(|held| {
            held.river_edges() != tile.river_edges() || held.river_beds() != tile.river_beds()
        });
        if gpu && (heights || rivers) {
            let texels = tile.river_texels();
            if let (Some(texels), Some((target, _))) = (&texels, &arrays.rivers) {
                write_layer(queue, target, layer, (bytemuck::cast_slice(&texels[..]), 8));
            }
            let surface = surfaces[&key.0];
            let spacing = tile.spacing();
            let first = surface.first_sample(key.1, spacing);
            self.control_jobs.push(ControlJob {
                layer,
                terrain: key.0,
                first: [first[0] as i32, first[1] as i32],
                spacing,
                level: surface.level(spacing),
                rivers: texels.is_some(),
            });
            self.surfaced.insert(key, self.now);
        }
        let control = !gpu
            && (tile.control().is_none()
                || before.is_none_or(|held| held.control() != tile.control()));
        if control {
            let borrowed;
            let values = match tile.control() {
                Some(values) => values,
                None => {
                    let listing = (&terrain.tiles[..], terrain.pyramid);
                    borrowed = ancestor_texels(listing, key.1, held_control);
                    &borrowed
                }
            };
            write_layer(
                queue,
                &arrays.control,
                layer,
                (bytemuck::cast_slice(values), 4),
            );
        }
        control
    }

    fn shade(
        &mut self,
        queue: &Queue,
        ((key, layer), tile, before): Uploading<'_>,
        terrain: &RenderTerrain,
    ) -> bool {
        let shading = tile.shading();
        if before.is_some_and(|held| same_shading(held.shading(), shading)) {
            return false;
        }
        let (control, gradients) = match shading {
            Some(shading) => {
                self.surfaced.insert(key, self.now);
                (shading.control_texels(), shading.gradient_texels())
            }
            None => {
                self.surfaced.remove(&key);
                let listing = (&terrain.tiles[..], terrain.pyramid);
                (
                    ancestor_texels(listing, key.1, held_control),
                    ancestor_texels(listing, key.1, held_gradients),
                )
            }
        };
        let arrays = self.arrays.as_ref().expect("arrays_allocated");
        write_layer(
            queue,
            &arrays.control,
            layer,
            (bytemuck::cast_slice(&control[..]), 4),
        );
        write_layer(
            queue,
            &arrays.normals,
            layer,
            (bytemuck::cast_slice(&gradients[..]), 4),
        );
        true
    }

    pub(super) fn fade_since(&self, at: f32) -> f32 {
        let since = engine_core::math::max(self.now - at, 0.0);
        1.0 - engine_core::math::min(since / ARRIVAL_SECONDS, 1.0)
    }

    pub(super) fn fades_of(&self, key: &(u64, TileKey), fades: bool) -> (f32, f32) {
        let materials = match self.surfaced.get(key) {
            None => 1.0,
            Some(&at) if fades => self.fade_since(at),
            Some(_) => 0.0,
        };
        if self.shaded.contains(key) {
            return (materials, materials);
        }
        let normals = match self.arrivals.get(key) {
            Some(&at) if fades => self.fade_since(at),
            _ => 0.0,
        };
        (normals, materials)
    }

    pub(super) fn fade(&mut self, fades: bool, count: u32) {
        let whole = self.fades.len() != count as usize
            || (self.faded_with).is_none_or(|(faded, then)| faded != fades || self.now < then);
        self.faded_with = Some((fades, self.now));
        self.fades_dirty = 0..0;
        let mut touched = std::mem::take(&mut self.touched);
        if whole {
            self.vacated.clear();
            self.fades.clear();
            self.fades.resize(count as usize, 0);
            self.fades_dirty = 0..self.fades.len();
            self.moving.clear();
            touched.clear();
            touched.extend(self.slots.keys().copied());
        } else {
            while let Some(layer) = self.vacated.pop() {
                self.set_fade(layer, 0);
            }
            touched.extend(self.moving.iter().copied());
            touched.sort_unstable();
            touched.dedup();
        }
        for &key in &touched {
            self.refade(key, fades);
        }
        touched.clear();
        self.touched = touched;
    }

    fn refade(&mut self, key: (u64, TileKey), fades: bool) {
        let Some(&layer) = self.slots.get(&key) else {
            self.moving.remove(&key);
            return;
        };
        let (normals, materials) = self.fades_of(&key, fades);
        let timed = self.surfaced.contains_key(&key);
        let normals_move = normals > 0.0 && (timed || !self.shaded.contains(&key));
        if normals_move || (materials > 0.0 && timed) {
            self.moving.insert(key);
        } else {
            self.moving.remove(&key);
        }
        self.set_fade(layer, fade_half(normals, materials));
    }

    fn set_fade(&mut self, layer: u32, half: u32) {
        let at = layer as usize / 2;
        let shift = (layer % 2) * 16;
        let Some(word) = self.fades.get_mut(at) else {
            return;
        };
        let value = (*word & !(0xFFFF << shift)) | (half << shift);
        if value != *word {
            *word = value;
            self.fades_dirty = widen(self.fades_dirty.clone(), at);
        }
    }

    pub(super) fn warm(&mut self, device: &wgpu::Device) {
        self.normal_pass(device);
        if self.gpu_materials {
            self.control_pass(device);
        }
    }

    fn control_pass(&mut self, device: &wgpu::Device) {
        self.control_pass.get_or_insert_with(|| {
            let layout = engine_render::pipelines::layout(device, &TERRAIN_CONTROL);
            let pipeline_layout = engine_render::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::TERRAIN_CONTROL_PIPELINES,
            );
            let pipeline = engine_render::pipelines::compute_entry(
                device,
                &pipeline_layout,
                terrain_control::tile_control,
            );
            (layout, pipeline)
        });
    }

    fn sync_surfaces(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
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
            let buffer = engine_render::buffers::storage(
                device,
                "engine/terrain-surface",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&surface.words));
            self.surfaces.insert(id, (Arc::clone(surface), buffer));
        }
    }

    fn compute_controls(&mut self, device: &wgpu::Device, queue: &Queue) {
        let ready = (self.arrays.as_ref()).is_some_and(|arrays| arrays.rivers.is_some());
        if !ready || self.control_jobs.is_empty() {
            self.control_jobs.clear();
            return;
        }
        self.control_pass(device);
        let (Some(arrays), Some((layout, pipeline))) = (&self.arrays, &self.control_pass) else {
            return;
        };
        let Some((_, rivers)) = &arrays.rivers else {
            return;
        };
        let mut words = vec![0_u32; self.control_jobs.len() * SLOT as usize / 4];
        for (index, job) in self.control_jobs.iter().enumerate() {
            let at = index * SLOT as usize / 4;
            words[at] = job.layer;
            words[at + 1] = job.level;
            words[at + 2] = job.first[0] as u32;
            words[at + 3] = job.first[1] as u32;
            words[at + 4] = job.spacing.to_bits();
            words[at + 5] = u32::from(job.rivers);
        }
        let jobs = engine_render::buffers::uniform(
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
                let group = engine_render::pipelines::group(
                    device,
                    "engine/terrain-control-group",
                    &TERRAIN_CONTROL,
                    layout,
                    &[
                        ("heights", engine_render::pipelines::view(&arrays.view)),
                        (
                            "job",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &jobs,
                                offset: index as u64 * SLOT,
                                size: wgpu::BufferSize::new(32),
                            }),
                        ),
                        ("data", data.as_entire_binding()),
                        ("control", engine_render::pipelines::view(&target)),
                        ("rivers", engine_render::pipelines::view(rivers)),
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

    fn normal_pass(&mut self, device: &wgpu::Device) {
        self.normal_pass.get_or_insert_with(|| {
            let layout = engine_render::pipelines::layout(device, &TERRAIN_NORMALS);
            let pipeline_layout = engine_render::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::TERRAIN_NORMAL_PIPELINES,
            );
            let pipeline = engine_render::pipelines::compute_entry(
                device,
                &pipeline_layout,
                terrain_normals::tile_normals,
            );
            (layout, pipeline)
        });
    }

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
        let mut words = vec![0_u32; self.normal_jobs.len() * SLOT as usize / 4];
        for (index, &(layer, spacing)) in self.normal_jobs.iter().enumerate() {
            let at = index * SLOT as usize / 4;
            words[at] = layer;
            words[at + 1] = spacing.to_bits();
        }
        let jobs = engine_render::buffers::uniform(
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
                let group = engine_render::pipelines::group(
                    device,
                    "engine/terrain-normals-group",
                    &TERRAIN_NORMALS,
                    layout,
                    &[
                        ("heights", engine_render::pipelines::view(&arrays.view)),
                        (
                            "job",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &jobs,
                                offset: index as u64 * SLOT,
                                size: wgpu::BufferSize::new(16),
                            }),
                        ),
                        ("normals", engine_render::pipelines::view(&target)),
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

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
        let array = |label, format, usage, mips| {
            layer_array(device, (label, format, usage, mips), layers)
        };
        let texture = array(
            "engine/terrain-heights",
            HEIGHT_FORMAT,
            wgpu::TextureUsages::empty(),
            HEIGHT_MIPS,
        );
        let normals = array(
            "engine/terrain-normals",
            NORMAL_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
            1,
        );
        let control = array(
            "engine/terrain-control",
            CONTROL_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
            1,
        );
        let rivers = self.gpu_materials.then(|| rivers_array(device, layers));
        let pyramid = |label: &'static str| {
            engine_render::memory::texture(
                device,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: ERROR_CELLS,
                        height: ERROR_CELLS,
                        depth_or_array_layers: layers,
                    },
                    mip_level_count: ERROR_MIPS,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: ERROR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        };
        let ground_errors = pyramid("engine/terrain-ground-errors");
        let water_errors = pyramid("engine/terrain-water-errors");
        if let Some(old) = &self.arrays
            && self.next_layer > 0
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/terrain-grow"),
            });
            for mip in 0..ERROR_MIPS {
                let side = ERROR_CELLS >> mip;
                let extent = wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: self.next_layer,
                };
                for (from, to) in [
                    (&old.ground_errors, &ground_errors),
                    (&old.water_errors, &water_errors),
                ] {
                    encoder.copy_texture_to_texture(mip_copy(from, mip), mip_copy(to, mip), extent);
                }
            }
            for mip in 0..HEIGHT_MIPS {
                let side = LAYER_SIZE >> mip;
                let extent = wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: self.next_layer,
                };
                encoder.copy_texture_to_texture(
                    mip_copy(&old.texture, mip),
                    mip_copy(&texture, mip),
                    extent,
                );
            }
            let extent = wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: self.next_layer,
            };
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
            if let (Some((old_rivers, _)), Some((new_rivers, _))) = (&old.rivers, &rivers) {
                encoder.copy_texture_to_texture(
                    old_rivers.as_image_copy(),
                    new_rivers.as_image_copy(),
                    extent,
                );
            }
            queue.submit_now([encoder.finish()]);
        }
        self.arrays = Some(Arrays {
            view: array_view(&texture),
            normals_view: array_view(&normals),
            control_view: array_view(&control),
            ground_errors_view: array_view(&ground_errors),
            water_errors_view: array_view(&water_errors),
            texture,
            control,
            normals,
            rivers,
            ground_errors,
            water_errors,
            layers,
        });
        self.generation += 1;
    }
}

fn layer_array(
    device: &wgpu::Device,
    (label, format, usage, mips): (&'static str, wgpu::TextureFormat, wgpu::TextureUsages, u32),
    layers: u32,
) -> wgpu::Texture {
    engine_render::memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | usage,
            view_formats: &[],
        },
    )
}

fn rivers_array(device: &wgpu::Device, layers: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let rivers = layer_array(
        device,
        (
            "engine/terrain-rivers",
            RIVER_FORMAT,
            wgpu::TextureUsages::empty(),
            1,
        ),
        layers,
    );
    let view = array_view(&rivers);
    (rivers, view)
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

pub(super) fn fade_half(normals: f32, materials: f32) -> u32 {
    let byte = |fade: f32| (engine_core::math::clamp(fade, 0.0, 1.0) * 255.0 + 0.5) as u32;
    byte(normals) | (byte(materials) << 8)
}

fn write_layer(queue: &Queue, texture: &wgpu::Texture, layer: u32, (texels, bytes): (&[u8], u32)) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LAYER_SIZE * bytes),
            rows_per_image: Some(LAYER_SIZE),
        },
        wgpu::Extent3d {
            width: LAYER_SIZE,
            height: LAYER_SIZE,
            depth_or_array_layers: 1,
        },
    );
}

fn write_errors(queue: &Queue, texture: &wgpu::Texture, layer: u32, pyramid: &ErrorPyramid) {
    write_mips(queue, texture, layer, |mip| pyramid.mip(mip));
}

fn write_mips<'a>(
    queue: &Queue,
    texture: &wgpu::Texture,
    layer: u32,
    mips: impl Fn(u32) -> &'a [u16],
) {
    for mip in 0..ERROR_MIPS {
        let side = ERROR_CELLS >> mip;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(mips(mip)),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 2),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn write_peaks(queue: &Queue, texture: &wgpu::Texture, layer: u32, peaks: &[f32]) {
    for mip in 0..ERROR_MIPS {
        let side = ERROR_CELLS >> mip;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip + 1,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(peaks_mip(peaks, mip)),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 4),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn mip_copy(texture: &wgpu::Texture, mip: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: mip,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
}

fn same_errors(a: &ErrorPyramid, b: &ErrorPyramid) -> bool {
    std::ptr::eq(a, b) || a == b
}

fn listed_tile(
    tiles: &[Arc<engine_terrain::Tile>],
    key: TileKey,
) -> Option<&Arc<engine_terrain::Tile>> {
    let at = tiles.binary_search_by_key(&key, |tile| tile.key()).ok()?;
    tiles.get(at)
}

fn listed_in(
    terrains: &RenderTerrains,
    (id, key): (u64, TileKey),
) -> Option<&Arc<engine_terrain::Tile>> {
    let terrain = terrains.terrains.iter().find(|terrain| terrain.id == id)?;
    listed_tile(&terrain.tiles, key)
}

type Uploading<'a> = (
    ((u64, TileKey), u32),
    &'a engine_terrain::Tile,
    Option<&'a Arc<engine_terrain::Tile>>,
);

fn same_shading(a: Option<&Arc<TileShading>>, b: Option<&Arc<TileShading>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

fn held_control(tile: &engine_terrain::Tile) -> Option<Box<[u32]>> {
    (tile.control().map(Box::from)).or_else(|| tile.shading().map(|held| held.control_texels()))
}

fn held_gradients(tile: &engine_terrain::Tile) -> Option<Box<[u32]>> {
    tile.shading().map(|held| held.gradient_texels())
}

fn ancestor_texels(
    (tiles, pyramid): (&[Arc<engine_terrain::Tile>], Pyramid),
    key: TileKey,
    held: fn(&engine_terrain::Tile) -> Option<Box<[u32]>>,
) -> Box<[u32]> {
    let size = LAYER_SIZE as usize;
    let mut out = vec![0_u32; size * size].into_boxed_slice();
    let (mut at, mut shift) = (key, 0_u32);
    let values = loop {
        let Some(parent) = pyramid.parent(at) else {
            return out;
        };
        (at, shift) = (parent, shift + 1);
        if let Some(values) = listed_tile(tiles, at).and_then(|tile| held(tile.as_ref())) {
            break values;
        }
    };
    let quads = i64::from(engine_terrain::TILE_QUADS);
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
            *value = values[from + x];
        }
    }
    out
}

struct ControlJob {
    layer: u32,
    terrain: u64,
    first: [i32; 2],
    spacing: f32,
    level: u32,
    rivers: bool,
}

fn peaks(heights: &[f32]) -> Box<[f32]> {
    let stride = engine_terrain::TILE_STRIDE as usize;
    let height = |x: usize, z: usize| heights.get((z + 1) * stride + x + 1).copied();
    let mut side = ERROR_CELLS as usize;
    let mut level: Vec<f32> = Vec::with_capacity(side * side);
    for cell_z in 0..side {
        for cell_x in 0..side {
            let mut peak = -f32::MAX;
            for z in 2 * cell_z..=2 * cell_z + 2 {
                for x in 2 * cell_x..=2 * cell_x + 2 {
                    peak = engine_core::math::max(peak, height(x, z).unwrap_or(peak));
                }
            }
            level.push(peak);
        }
    }
    let mut all = Vec::with_capacity(ERROR_TEXELS);
    all.extend_from_slice(&level);
    while side > 1 {
        let half = side / 2;
        let peak = |x: usize, z: usize| level[z * side + x];
        let coarser: Vec<f32> = (0..half * half)
            .map(|at| {
                let (x, z) = (2 * (at % half), 2 * (at / half));
                engine_core::math::max(
                    engine_core::math::max(peak(x, z), peak(x + 1, z)),
                    engine_core::math::max(peak(x, z + 1), peak(x + 1, z + 1)),
                )
            })
            .collect();
        all.extend_from_slice(&coarser);
        level = coarser;
        side = half;
    }
    all.into_boxed_slice()
}

fn peaks_top(peaks: &[f32]) -> f32 {
    peaks.last().copied().unwrap_or(-f32::MAX)
}

fn peaks_mip(peaks: &[f32], mip: u32) -> &[f32] {
    let side = |level: u32| (ERROR_CELLS >> level) as usize;
    let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
    &peaks[start..start + side(mip) * side(mip)]
}
