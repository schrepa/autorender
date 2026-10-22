//! The tile atlas: every resident tile's heights, normals (the smooth
//! surface's gradient at each sample), control values and meters to rivers'
//! edges in four texture arrays, a layer each; which tile each layer holds,
//! when it arrived, and the layers free; the GPU's normal maps and control
//! values of tiles uploaded each frame; and each layer's fades.
//!
//! A sync reads what each terrain's list gained, lost or replaced since the
//! list it last synced (both in key order), beside the listed tiles still
//! waiting for a layer, so its work follows what changed, not how many
//! tiles are resident. Fades are worked out again only for the tiles whose
//! fades can have changed: those fading in as time passes, and those whose
//! layer, arrival or surface materials changed.

#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before tiles upload"
)]

use super::*;
use crate::shaders::entry::{terrain_control, terrain_normals};
use engine_core::collections::HashSet;
use engine_core::terrain_data::Pyramid;
use std::collections::{BTreeMap, BTreeSet};

/// The arrays: heights, control values, normals and meters to rivers'
/// edges, a layer per tile.
struct Arrays {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    control: wgpu::Texture,
    control_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,
    rivers: wgpu::Texture,
    rivers_view: wgpu::TextureView,
    layers: u32,
}

/// A terrain's tiles as last synced: their revision and list, in key
/// order, and whether the GPU gave them control values.
struct Listed {
    revision: u64,
    tiles: Arc<[Arc<engine_core::Tile>]>,
    surface: bool,
}

/// What changed in the terrains' lists since they were last synced: the
/// tiles that left them, those that arrived in them and those replaced
/// under their keys.
#[derive(Default)]
struct Changes {
    left: Vec<(u64, TileKey)>,
    arrived: Vec<(u64, TileKey)>,
    swapped: Vec<(u64, TileKey)>,
}

impl Changes {
    /// Notes what changed from `before` to `now`, terrain `id`'s lists in
    /// key order.
    fn compare(
        &mut self,
        id: u64,
        before: &[Arc<engine_core::Tile>],
        now: &[Arc<engine_core::Tile>],
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

/// Resident tiles replaced under their keys that upload again: the layers
/// they leave, which they take again, and the tiles they replace, so the
/// arrays keep what did not change.
#[derive(Default)]
struct Replaced {
    layers: HashMap<(u64, TileKey), u32>,
    tiles: HashMap<(u64, TileKey), Arc<engine_core::Tile>>,
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
    pub(super) held: HashMap<(u64, TileKey), Arc<engine_core::Tile>>,
    pub(super) free: Vec<u32>,
    pub(super) next_layer: u32,
    /// Each terrain's tiles as last synced, by id.
    synced: HashMap<u64, Listed>,
    /// Listed tiles that found no layer, which every sync tries again, and
    /// the resident tiles with holes, in key order.
    pub(super) unplaced: HashSet<(u64, TileKey)>,
    pub(super) holed: BTreeSet<(u64, TileKey)>,
    /// Tiles uploaded this frame, resident on the GPU, and dropped for want
    /// of layers.
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    /// Tiles the GPU gave control values this frame.
    pub(crate) controls: u32,
    // Normal maps of tiles uploaded this frame (layer, spacing), and the
    // pass computing them.
    normal_jobs: Vec<(u32, f32)>,
    normal_pass: Option<(crate::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    // Control values the GPU gives tiles uploaded this frame,
    // the pass giving them, and each terrain's surface data there with
    // the data it holds, which changes when the data does.
    control_jobs: Vec<ControlJob>,
    control_pass: Option<(crate::pipelines::GroupLayout, wgpu::ComputePipeline)>,
    surfaces: HashMap<u64, (Arc<SurfaceData>, wgpu::Buffer)>,
    // The sky maps' sampler, linear; and the nearest one the control
    // values and normal maps gather through.
    pub(super) sky_sampler: wgpu::Sampler,
    pub(super) control_sampler: wgpu::Sampler,
    /// The tiles that arrived, left, were replaced or finished fading in
    /// this frame, whose pages the runtime virtual texture drops; those
    /// that began fading in; and those whose held tile changed (it arrived,
    /// left or was replaced), each once, which their water follows.
    pub(super) changed: Vec<(u64, TileKey)>,
    pub(super) entered: Vec<(u64, TileKey)>,
    pub(super) held_changes: Vec<(u64, TileKey)>,
    /// Each layer's fades as shaders read them, the words this frame
    /// changed, and whether frames faded and the time when they were last
    /// worked out.
    pub(super) fades: Vec<u32>,
    pub(super) fades_dirty: Range<usize>,
    pub(super) faded_with: Option<(bool, f32)>,
    /// The resident tiles a coarser level still shows over, and those of
    /// them whose fades move as time passes.
    pub(super) fading: HashSet<(u64, TileKey)>,
    moving: HashSet<(u64, TileKey)>,
    // The tiles whose layer, arrival or surface materials' time changed
    // since fades were last worked out, and the layers tiles left.
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
            held: HashMap::default(),
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
            surfaces: HashMap::default(),
            // Nearest texels: control values and normal maps gather as
            // stored.
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
            entered: Vec::new(),
            held_changes: Vec::new(),
            fades: Vec::new(),
            fades_dirty: 0..0,
            faded_with: None,
            fading: HashSet::default(),
            moving: HashSet::default(),
            touched: Vec::new(),
            vacated: Vec::new(),
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

    /// Bytes of the height, normal, control and river arrays.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers) * u64::from(LAYER_SIZE * LAYER_SIZE) * 14
        })
    }

    /// Layers the arrays hold.
    pub(super) fn layers(&self) -> u32 {
        self.arrays.as_ref().map_or(0, |arrays| arrays.layers)
    }

    /// Starts a frame at `now`, in seconds of presentation time: no tile
    /// uploaded, given control values or changed yet.
    pub(super) fn begin(&mut self, now: f32) {
        (self.uploads, self.controls, self.now) = (0, 0, now);
        self.changed.clear();
        self.entered.clear();
        self.held_changes.clear();
    }

    /// Uploads tiles the GPU lacks and releases those no terrain lists, from
    /// what each terrain's list gained, lost or replaced since it was last
    /// synced and the listed tiles still waiting for a layer: `None` when
    /// every terrain's tiles stand as last synced, else whether a tile
    /// replaced under its key changed its hole bits.
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
        // Terrains whose tiles' control values the GPU gives.
        let surfaces: BTreeMap<u64, &Arc<SurfaceData>> = terrains
            .terrains
            .iter()
            .filter_map(|terrain| Some((terrain.id, terrain.gpu_surface.as_ref()?)))
            .collect();
        self.sync_surfaces(device, queue, &surfaces);
        self.surface_times(terrains, &surfaces, &changes);
        let mut replaced = self.replace(terrains, &changes.swapped);
        // Upload the rest, coarse levels first: the tiles that arrived,
        // those replaced and those still waiting for a layer.
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
            let listing = (&terrain.tiles[..], terrain.pyramid);
            let room = (&surfaces, capacity);
            edited |= self.upload(queue, (key, tile), listing, &mut replaced, room);
        }
        self.compute_controls(device, queue);
        // Layers of replaced tiles that found no upload are free again.
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

    /// What changed in `terrains`' lists since they were last synced: each
    /// list against the one before where its revision advanced, and every
    /// tile of the terrains no longer listed, left.
    fn changes(&self, terrains: &RenderTerrains) -> Changes {
        let mut changes = Changes::default();
        for terrain in &terrains.terrains {
            let synced = self.synced.get(&terrain.id);
            if synced.is_some_and(|synced| synced.revision == terrain.tiles_revision) {
                continue;
            }
            let before: &[Arc<engine_core::Tile>] = match synced {
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

    /// Releases the tiles of `left` holding layers, in key order, so which
    /// layers tiles take next follows no hash: layers free last are taken
    /// first.
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
            self.arrivals.remove(&key);
            self.surfaced.remove(&key);
            self.holed.remove(&key);
        }
    }

    /// Gives tiles their surface materials' times as the lists changed: a
    /// resident tile replaced without control values by one with them
    /// takes them over from now, and a tile without them on a terrain whose
    /// surface the GPU does not read has none, every such tile of a terrain
    /// whose surface the GPU stopped reading included.
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
            for tile in terrain.tiles.iter().filter(|tile| tile.control().is_none()) {
                self.unsurface((terrain.id, tile.key()));
            }
        }
        for &key in changes.arrived.iter().chain(&changes.swapped) {
            let Some(tile) = listed_in(terrains, key) else {
                continue;
            };
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

    /// Takes resident tile `key`'s surface materials' time away, if it had
    /// one.
    fn unsurface(&mut self, key: (u64, TileKey)) {
        if self.surfaced.remove(&key).is_some() {
            self.touched.push(key);
        }
    }

    /// Of the resident tiles `swapped` replaced under their keys, keeps
    /// those over the same ground and materials in their layers as they
    /// are (their water made again as their rivers run now, which the
    /// water's blocks follow), and takes the rest out of their layers to
    /// upload again, keeping what they replace: a tile arriving whole after
    /// its shape, or with its objects standing far away, has the same
    /// heights.
    fn replace(&mut self, terrains: &RenderTerrains, swapped: &[(u64, TileKey)]) -> Replaced {
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
            });
            if same_ground {
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

    /// Uploads listed tile `key`, `tile` (of a terrain listing `tiles` over
    /// `pyramid`), into a layer: the one it left as it was replaced
    /// (`replaced`), else a free one or a new one within `capacity`; a tile
    /// finding none leaves its parent standing in and waits for the next
    /// sync, dropped. The GPU gives tiles without control values on
    /// terrains it reads the surface of (`surfaces`) theirs. Returns
    /// whether it changed the hole bits of the tile it replaced.
    fn upload(
        &mut self,
        queue: &Queue,
        (key, tile): ((u64, TileKey), &Arc<engine_core::Tile>),
        (tiles, pyramid): (&[Arc<engine_core::Tile>], Pyramid),
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
        // Tiles arriving grow out of their parents; edited ones
        // change in place.
        if arriving {
            self.arrivals.insert(key, self.now);
            if tile.control().is_some() {
                self.surfaced.insert(key, self.now);
            }
        }
        // What draws over it changes only as it uploads; tiles finding
        // no layer leave their parents standing in.
        self.changed.push(key);
        self.held_changes.push(key);
        self.touched.push(key);
        let arrays = self.arrays.as_ref().expect("allocated above");
        let before = replaced.tiles.remove(&key);
        let edited = (before.as_ref()).is_some_and(|held| held.hole_bits() != tile.hole_bits());
        let heights = (before.as_ref()).is_none_or(|held| held.heights() != tile.heights());
        if heights {
            let texels = bytemuck::cast_slice(tile.heights());
            write_layer(queue, &arrays.texture, layer, (texels, 4));
            self.normal_jobs.push((layer, tile.spacing()));
        }
        // Without control values on a terrain whose surface the GPU
        // reads, the GPU gives them as its heights or meters to
        // rivers' edges change, from those meters it uploads, its
        // materials then taking over from the coarser level's.
        let gpu = tile.control().is_none() && surfaces.contains_key(&key.0);
        let rivers = (before.as_ref()).is_none_or(|held| held.river_edges() != tile.river_edges());
        if gpu && (heights || rivers) {
            let edges = tile.river_edges();
            if let Some(edges) = edges {
                write_layer(queue, &arrays.rivers, layer, (bytemuck::cast_slice(edges), 2));
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
                rivers: edges.is_some(),
            });
            self.surfaced.insert(key, self.now);
        }
        // Its control values; a tile without its own yet
        // takes its nearest ancestor's for ground cover, and without a
        // surface, the first material everywhere.
        let control = !gpu
            && (tile.control().is_none()
                || (before.as_ref()).is_none_or(|held| held.control() != tile.control()));
        if control {
            let borrowed;
            let values = match tile.control() {
                Some(values) => values,
                None => {
                    borrowed = ancestor_control(tiles, key.1, pyramid);
                    &borrowed
                }
            };
            write_layer(queue, &arrays.control, layer, (bytemuck::cast_slice(values), 4));
        }
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

    /// How much of a coarser level still shows of what arrived at `at`:
    /// 1 then, 0 [`ARRIVAL_SECONDS`] later.
    pub(super) fn fade_since(&self, at: f32) -> f32 {
        let since = engine_core::math::max(self.now - at, 0.0);
        1.0 - engine_core::math::min(since / ARRIVAL_SECONDS, 1.0)
    }

    /// How much of the next coarser level's normals and surface materials
    /// shows over resident tile `key` now: as it grows out of its parent on
    /// ordinary frames (with `fades`) and none on exact ones, but the whole
    /// of the coarser level's materials over a tile without its own yet.
    pub(super) fn fades_of(&self, key: &(u64, TileKey), fades: bool) -> (f32, f32) {
        let normals = match self.arrivals.get(key) {
            Some(&at) if fades => self.fade_since(at),
            _ => 0.0,
        };
        let materials = match self.surfaced.get(key) {
            None => 1.0,
            Some(&at) if fades => self.fade_since(at),
            Some(_) => 0.0,
        };
        (normals, materials)
    }

    /// Works each layer's fades out for `count` words, noting the words that
    /// changed: how much of the next coarser level's normals and surface
    /// materials still show over its tile's ([`Self::fades_of`]), two
    /// layers a word ([`fade_half`]). Only the tiles whose fades can have
    /// changed are worked out again: those fading in as time passes and
    /// those whose layer, arrival or surface materials changed; every tile
    /// when the words or `fades` changed or time ran back. Tiles that
    /// finished fading in count as changed, and those that began are noted.
    pub(super) fn fade(&mut self, fades: bool, count: u32) {
        let whole = self.fades.len() != count as usize
            || (self.faded_with).is_none_or(|(faded, then)| faded != fades || self.now < then);
        self.faded_with = Some((fades, self.now));
        self.fades_dirty = 0..0;
        let mut touched = std::mem::take(&mut self.touched);
        let mut faded = Vec::new();
        if whole {
            self.vacated.clear();
            self.fades.clear();
            self.fades.resize(count as usize, 0);
            self.fades_dirty = 0..self.fades.len();
            let before = std::mem::take(&mut self.fading);
            self.moving.clear();
            touched.clear();
            touched.extend(self.slots.keys().copied());
            for &key in &touched {
                self.refade(key, fades);
            }
            faded.extend(before.iter().filter(|key| !self.fading.contains(*key)).copied());
            let began = self.fading.iter().filter(|key| !before.contains(*key));
            self.entered.extend(began.copied());
        } else {
            while let Some(layer) = self.vacated.pop() {
                self.set_fade(layer, 0);
            }
            touched.extend(self.moving.iter().copied());
            touched.sort_unstable();
            touched.dedup();
            for &key in &touched {
                let was = self.fading.contains(&key);
                self.refade(key, fades);
                match (was, self.fading.contains(&key)) {
                    (true, false) => faded.push(key),
                    (false, true) => self.entered.push(key),
                    (true, true) | (false, false) => {}
                }
            }
        }
        touched.clear();
        self.touched = touched;
        // Sorted: the order pages drop in follows no hash.
        faded.sort_unstable();
        self.entered.sort_unstable();
        self.changed.extend(faded);
    }

    /// Works tile `key`'s fades out again: its half of the word of the layer
    /// it holds, whether a coarser level shows over it, and whether that
    /// changes as time passes; a tile holding no layer fades in no more.
    fn refade(&mut self, key: (u64, TileKey), fades: bool) {
        let Some(&layer) = self.slots.get(&key) else {
            self.fading.remove(&key);
            self.moving.remove(&key);
            return;
        };
        let (normals, materials) = self.fades_of(&key, fades);
        if normals > 0.0 || materials > 0.0 {
            self.fading.insert(key);
        } else {
            self.fading.remove(&key);
        }
        // Materials fading in from their own time move; a tile without its
        // own shows the coarser level's whole until they arrive.
        if normals > 0.0 || (materials > 0.0 && self.surfaced.contains_key(&key)) {
            self.moving.insert(key);
        } else {
            self.moving.remove(&key);
        }
        self.set_fade(layer, fade_half(normals, materials));
    }

    /// Sets `layer`'s half of its word of fades to `half`, noting the word
    /// where it changed; layers past the words are left out.
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

    /// Whether terrain `id`'s tile `key` is resident and stands as it is:
    /// no coarser level's normals or surface materials show over it this
    /// frame.
    pub(super) fn settled(&self, key: (u64, TileKey)) -> bool {
        self.slots.contains_key(&key) && !self.fading.contains(&key)
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
            words[at + 5] = u32::from(job.rivers);
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
                        ("rivers", crate::pipelines::view(&arrays.rivers_view)),
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
        let rivers = array(
            "engine/terrain-rivers",
            RIVER_FORMAT,
            wgpu::TextureUsages::empty(),
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
            encoder.copy_texture_to_texture(
                old.rivers.as_image_copy(),
                rivers.as_image_copy(),
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
            rivers_view: array_view(&rivers),
            texture,
            control,
            normals,
            rivers,
            layers,
        });
        self.generation += 1;
    }
}

/// A layer's half of its word of fades: how much of the coarser level's
/// normals and surface materials show, 0 to 255 in a byte each.
pub(super) fn fade_half(normals: f32, materials: f32) -> u32 {
    let byte = |fade: f32| (engine_core::math::clamp(fade, 0.0, 1.0) * 255.0 + 0.5) as u32;
    byte(normals) | (byte(materials) << 8)
}

/// Writes `texels` of `bytes` bytes each, a tile's samples with their
/// border, into `layer` of `texture`.
fn write_layer(
    queue: &Queue,
    texture: &wgpu::Texture,
    layer: u32,
    (texels, bytes): (&[u8], u32),
) {
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

/// The tile `tiles`, a terrain's list in key order, lists under `key`.
fn listed_tile(
    tiles: &[Arc<engine_core::Tile>],
    key: TileKey,
) -> Option<&Arc<engine_core::Tile>> {
    let at = tiles.binary_search_by_key(&key, |tile| tile.key()).ok()?;
    tiles.get(at)
}

/// The tile `terrains` list under terrain `id`'s `key`.
fn listed_in(
    terrains: &RenderTerrains,
    (id, key): (u64, TileKey),
) -> Option<&Arc<engine_core::Tile>> {
    let terrain = terrains.terrains.iter().find(|terrain| terrain.id == id)?;
    listed_tile(&terrain.tiles, key)
}

/// Control values for tile `key` without its own yet, of a terrain over
/// `pyramid` listing `tiles`, for ground cover to grow from: its nearest
/// listed ancestor's holding some, each sample its nearest sample's there;
/// the first material everywhere without one.
fn ancestor_control(
    tiles: &[Arc<engine_core::Tile>],
    key: TileKey,
    pyramid: Pyramid,
) -> Vec<u32> {
    let size = LAYER_SIZE as usize;
    let mut out = vec![0_u32; size * size];
    let (mut at, mut shift) = (key, 0_u32);
    let control = loop {
        let Some(parent) = pyramid.parent(at) else {
            return out;
        };
        (at, shift) = (parent, shift + 1);
        if let Some(control) = listed_tile(tiles, at).and_then(|tile| tile.control()) {
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
/// terrain, its first sample, spacing, the hydrology images' level its
/// masks read, and whether its layer of the river array holds its meters
/// to rivers' edges.
struct ControlJob {
    layer: u32,
    terrain: u64,
    first: [i32; 2],
    spacing: f32,
    level: u32,
    rivers: bool,
}
