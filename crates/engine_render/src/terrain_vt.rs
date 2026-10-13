//! The runtime virtual texture of terrain's ground.
//!
//! Terrain's deferred resolves spend most of their time on
//! its surface materials: control values, height blending, every
//! material's maps over its projection planes and lattices, detail noise
//! and variety. All of it holds still while the light and the view move,
//! so pages of it, baked once by the exact path's own code (the ground as
//! seen where a pixel's footprint spans a texel), stand in for it at the
//! pixels that ask: a page-table read and three bilinear texel fetches.
//!
//! Mip m's texels span [`TEXEL`] · 2^m meters, its pages [`PAGE_TEXELS`] of
//! them, kept with a border of
//! [`VT_BORDER_TEXELS`](crate::gpu_layouts::constants::VT_BORDER_TEXELS) so
//! filtering never reaches
//! a neighbor. Pages lie in slots of sheets past the materials' layers of
//! the material arrays (`terrain_maps.rs`), which grow as the views ask for
//! more pages than they hold, up to `TerrainSettings::virtual_texture_pages`
//! where set and the device's texture array layers; pages asked for beyond
//! those drop and count. Each textured terrain keeps, per mip, a window of
//! [`WINDOW`]² pages around the eye, its table entries in the directory
//! buffer after the directory's words (the resolve's fragments are at their
//! device's limits of textures and storage buffers, so the texture adds no
//! binding to them).
//!
//! Pixels ask for pages through a compute pass after the resolves, which
//! marks the pages of the mips around each 8x8 block's footprint (a
//! different pixel of the block each frame) in a bitmap read back a few
//! frames late. Pages asked for and absent bake, up to
//! `virtual_texture_bakes` a frame, coarsest first, into free slots or the
//! least recently asked for. Until a pixel's two pages are baked it
//! computes its ground as before, so pages arriving never change what
//! shows, only what it costs. A terrain's sky map rebuilt drops its pages.
//!
//! Pages go stale and bake again with the budget absent pages leave, those
//! still asked for only: pages over tiles that arrived, left or were edited
//! first, once the tiles have held still [`SETTLE_FRAMES`] (they show their
//! old ground until then, rather than none), then
//! pages whose shading levels (`terrain_shading_level`, which the exact
//! path takes from each point's distance to the eye they bake from) the
//! eye has turned by more than [`STALE_SHADING`] of a level.

#![expect(
    clippy::expect_used,
    reason = "pages are asked for and textured before they are read"
)]

use crate::shaders::entry::meshes;
use engine_core::collections::HashMap;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use engine_core::glam::Vec3;
use engine_core::math::{max, min};
use engine_core::{RenderTerrains, TerrainSettings, TileKey};

use crate::bindings::{
    TERRAIN_VT_BAKE, TERRAIN_VT_CAMERA, TERRAIN_VT_FEEDBACK, TERRAIN_VT_TERRAIN,
};
use crate::buffers::{READBACKS, ReadbackRing};
use crate::gpu_layouts::constants::VT_MIPS as MIPS;
pub(crate) use crate::gpu_layouts::constants::VT_PAGE_TEXELS as PAGE_TEXELS;
pub(crate) use crate::gpu_layouts::constants::VT_SLOT_TEXELS as SLOT_TEXELS;
use crate::gpu_layouts::constants::VT_WINDOW as WINDOW;
use crate::gpu_layouts::constants::{VT_FEEDBACK_BLOCK, VT_FEEDBACK_GROUP};

const TEXEL: f32 = 1.0 / 256.0;
/// Words a textured terrain's windows and tables take in the directory
/// buffer, and its bits of pages wanted.
pub(crate) const TERRAIN_WORDS: usize = 2 * MIPS + MIPS * (WINDOW * WINDOW) as usize;
const WANTED_WORDS: usize = MIPS * (WINDOW * WINDOW) as usize / 32;
/// Frames a page goes unasked for before another may take its slot: a
/// round of the feedback's pixels, so every pixel still showing it has
/// asked.
const KEEP_FRAMES: u64 = 64;
/// Pages the sheets first hold, which grow by a quarter past what the views
/// ask for once they run out.
const FIRST_PAGES: u32 = 64;
/// Samples around a changed tile whose pages it drops: its border, and the
/// neighbors its normals and bilinear heights read.
const TILE_MARGIN: f32 = 2.0;
/// The share of a level the exact path's shading levels may turn from a
/// page's before it bakes again.
const STALE_SHADING: f32 = 0.1;
/// Frames the tiles under a page hold still before it bakes again: tiles
/// coming and going frame by frame cost no bakes.
const SETTLE_FRAMES: u64 = 4;

/// A page: its terrain, mip, and page along x and z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PageKey {
    terrain: u64,
    mip: u8,
    page: [i32; 2],
}

use crate::gpu_layouts::terrain_vt::BakeRecord;

/// A textured terrain: its index this frame (its windows' and tables'
/// place, and its feedback bits'), its windows' first pages, its table
/// entries (slot plus one, 0 while absent) and which mips' tables changed
/// since written, and the sky map build its pages follow.
struct Textured {
    index: u32,
    firsts: [[i32; 2]; MIPS],
    entries: Vec<u32>,
    dirty: [bool; MIPS],
    sky: u64,
}

impl Textured {
    fn new() -> Self {
        Self {
            index: 0,
            firsts: [[i32::MIN / 2; 2]; MIPS],
            entries: vec![0; MIPS * (WINDOW * WINDOW) as usize],
            dirty: [true; MIPS],
            sky: u64::MAX,
        }
    }
}

/// The table entry of `page` in a window of mip `mip`: page modulo the
/// window.
fn entry(mip: usize, page: [i32; 2]) -> usize {
    let [x, z] = page.map(|axis| axis.rem_euclid(WINDOW) as usize);
    mip * (WINDOW * WINDOW) as usize + z * WINDOW as usize + x
}

/// Whether `page` lies in the window starting at `first`.
fn inside(first: [i32; 2], page: [i32; 2]) -> bool {
    (0..2).all(|axis| (first[axis]..first[axis] + WINDOW).contains(&page[axis]))
}

/// The page holding entry `index` of a window starting at `first`.
fn page_of(first: [i32; 2], index: usize) -> [i32; 2] {
    let cell = [
        (index % WINDOW as usize) as i32,
        (index / WINDOW as usize) as i32,
    ];
    [0, 1].map(|axis| first[axis] + (cell[axis] - first[axis]).rem_euclid(WINDOW))
}

/// Meters a texel of mip `mip` spans.
fn texel_size(mip: usize) -> f32 {
    TEXEL * (1_u32 << mip) as f32
}

/// What a terrain's shading levels turn with: its origin, its levels'
/// reaches (`RenderTerrain::reach`) and the heights its drawn patches span
/// from its origin.
struct Levels<'a> {
    origin: Vec3,
    reach: &'a [f32],
    heights: [f32; 2],
}

impl Levels<'_> {
    /// The nearest and farthest a point of `page` can lie from `eye`.
    fn distances(&self, page: &PageKey, eye: Vec3) -> (f32, f32) {
        let span = texel_size(usize::from(page.mip)) * PAGE_TEXELS as f32;
        let local = eye - self.origin;
        let mut near = 0.0;
        let mut far = 0.0;
        for (axis, at) in [local.x, local.z].into_iter().enumerate() {
            let low = page.page[axis] as f32 * span;
            let out = max(max(low - at, at - low - span), 0.0);
            let across = max((at - low).abs(), (low + span - at).abs());
            near += out * out;
            far += across * across;
        }
        let [bottom, top] = self.heights;
        let out = max(max(bottom - local.y, local.y - top), 0.0);
        let across = max((local.y - bottom).abs(), (top - local.y).abs());
        ((near + out * out).sqrt(), (far + across * across).sqrt())
    }

    /// The most the exact path's shading levels anywhere on `page` turn, as
    /// a share of a level, between eyes `from` and `to`: none while all of
    /// it lies within the first half of the finest level's reach, which no
    /// blend reaches, and otherwise at most the distance moved over half the
    /// reach of the level shading its nearest point, where levels blend
    /// fastest.
    fn turn(&self, page: &PageKey, from: Vec3, to: Vec3) -> f32 {
        let (near_from, far_from) = self.distances(page, from);
        let (near_to, far_to) = self.distances(page, to);
        let first = self.reach.first().copied().unwrap_or(f32::MAX);
        if max(far_from, far_to) < 0.5 * first {
            return 0.0;
        }
        let nearest = min(near_from, near_to);
        let reach = self
            .reach
            .iter()
            .copied()
            .find(|&reach| nearest < reach)
            .unwrap_or(f32::MAX);
        from.distance(to) * 2.0 / reach
    }
}

/// What the texture did this frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VtStats {
    /// Slots the sheets hold, pages resident, baked this frame, wanted but
    /// absent, wanted but with no slot left under the budget or the
    /// device's layers, and evicted.
    pub slots: u32,
    pub pages: u32,
    pub bakes: u32,
    pub missing: u32,
    pub dropped: u32,
    pub evictions: u32,
}

/// The windows a feedback read back through: each textured terrain's id,
/// index and windows' first pages.
type Snapshot = Vec<(u64, u32, [[i32; 2]; MIPS])>;

/// Feedback words read back, tagged with the frame that recorded them.
type Arrived = Rc<RefCell<Vec<(u64, Vec<u32>)>>>;

/// What a frame's uniform tells a terrain's shaders of its texture: `vt`,
/// `vt_scale` and `vt_layout` of `TerrainUniform`.
pub(crate) type VtUniform = ([u32; 4], [f32; 4], [u32; 4]);

/// A frame's page bakes: by sheet and terrain, their records as the bakes'
/// buffer takes them (kept from frame to frame), and that buffer.
struct Bakes {
    list: Vec<(u32, u64, BakeRecord)>,
    records: Vec<BakeRecord>,
    buffer: crate::buffers::Growable,
}

pub(crate) struct VirtualTexture {
    enabled: bool,
    bakes_per_frame: u32,
    // The pages it asks the sheets to hold, and the most they may.
    want: u32,
    limit: u32,
    // The sheets' slots along a side and per sheet, and the material maps
    // generation, layers of materials and texels along a side they lie in.
    per_row: u32,
    per_sheet: u32,
    capacity: u32,
    maps_generation: u64,
    materials: u32,
    sheet_texels: u32,
    // Each textured terrain by terrain ID, the order its feedback
    // dispatches and directory writes run in.
    terrains: BTreeMap<u64, Textured>,
    resident: HashMap<PageKey, u32>,
    owners: Vec<Option<PageKey>>,
    asked: Vec<u64>,
    // The eye each slot's page baked from, none once tiles under it change,
    // and the frame they last did.
    baked_from: Vec<Option<Vec3>>,
    changed: Vec<u64>,
    free: Vec<u32>,
    frame: u64,
    // The directory buffer's words where the windows and tables start, and
    // the buffer's generation, which a new buffer writes whole.
    base: u32,
    directory_generation: u64,
    // Feedback: the bits pixels set, their read-backs (tagged with their
    // frame) and the windows each frame read them through, and the words
    // read back but not yet taken.
    wanted: wgpu::Buffer,
    wanted_words: usize,
    readbacks: ReadbackRing<u64>,
    snapshots: HashMap<u64, Snapshot>,
    arrived: Arrived,
    // This frame's bakes by sheet and terrain, their records and buffer.
    bakes: Bakes,
    // Bakes' and feedback's layouts and pipelines.
    bake_layout: crate::pipelines::GroupLayout,
    bake_pipeline: Option<wgpu::RenderPipeline>,
    feedback_layouts: [crate::pipelines::GroupLayout; 3],
    feedback_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) stats: VtStats,
}

impl VirtualTexture {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            enabled: false,
            bakes_per_frame: 0,
            want: FIRST_PAGES,
            limit: u32::MAX,
            per_row: 0,
            per_sheet: 0,
            capacity: 0,
            maps_generation: u64::MAX,
            materials: 0,
            sheet_texels: 1,
            terrains: BTreeMap::new(),
            resident: HashMap::default(),
            owners: Vec::new(),
            asked: Vec::new(),
            baked_from: Vec::new(),
            changed: Vec::new(),
            free: Vec::new(),
            frame: 0,
            base: 0,
            directory_generation: u64::MAX,
            wanted: crate::buffers::storage(
                device,
                "engine/terrain-vt-wanted",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            wanted_words: 0,
            readbacks: ReadbackRing::new("engine/terrain-vt-readback"),
            snapshots: HashMap::default(),
            arrived: Rc::new(RefCell::new(Vec::new())),
            bakes: Bakes {
                list: Vec::new(),
                records: Vec::new(),
                buffer: crate::buffers::Growable::new(
                    device,
                    "engine/terrain-vt-bakes",
                    16,
                    wgpu::BufferUsages::empty(),
                ),
            },
            bake_layout: crate::pipelines::layout(device, &TERRAIN_VT_BAKE),
            bake_pipeline: None,
            feedback_layouts: [
                crate::pipelines::layout(device, &TERRAIN_VT_CAMERA),
                crate::pipelines::layout(device, &TERRAIN_VT_TERRAIN),
                crate::pipelines::layout(device, &TERRAIN_VT_FEEDBACK),
            ],
            feedback_pipeline: None,
            stats: VtStats::default(),
        }
    }

    /// Pages the material maps should hold sheets for: as many as the views
    /// asked for while the texture runs, within `settings`' budget; none
    /// otherwise.
    pub(crate) fn wanted(&self, settings: &TerrainSettings) -> u32 {
        if settings.enabled && settings.virtual_texture {
            self.want
                .min(settings.virtual_texture_pages.unwrap_or(u32::MAX))
        } else {
            0
        }
    }

    /// Words the windows and tables of `terrains` take after the
    /// directory's, at the directory buffer's word `base`.
    pub(crate) fn words(terrains: usize) -> usize {
        terrains * TERRAIN_WORDS
    }

    /// Drops every page and slot.
    fn reset(&mut self) {
        self.terrains.clear();
        self.resident.clear();
        self.owners = vec![None; self.capacity as usize];
        self.asked = vec![0; self.capacity as usize];
        self.baked_from = vec![None; self.capacity as usize];
        self.changed = vec![0; self.capacity as usize];
        self.free = (0..self.capacity).rev().collect();
        self.readbacks.clear();
        self.snapshots.clear();
        self.arrived.borrow_mut().clear();
    }

    /// Frees `slot`, clearing its page's entry where its terrain keeps it.
    fn release(&mut self, slot: u32) {
        if let Some(key) = self.owners[slot as usize].take() {
            self.resident.remove(&key);
            if let Some(terrain) = self.terrains.get_mut(&key.terrain) {
                let mip = usize::from(key.mip);
                if inside(terrain.firsts[mip], key.page) {
                    terrain.entries[entry(mip, key.page)] = 0;
                    terrain.dirty[mip] = true;
                }
            }
            self.free.push(slot);
        }
    }

    /// Lays out this frame's texture: the slots the material maps' sheets
    /// hold (new ones free; new materials empty them), each terrain's
    /// windows around `eye`, the pages tiles that changed (`changed`) or
    /// rebuilt sky maps drop, the pages pixels asked for, and the bakes this
    /// frame draws; writes the windows and tables into the directory buffer
    /// at word `base` and the bakes into their buffer. Returns each
    /// terrain's uniform words, by its index in `terrains`.
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (eye, settings): (Vec3, &TerrainSettings),
        maps: &crate::terrain_maps::TerrainMaps,
        sky: &crate::terrain_sky::SkyMaps,
        changed: &[(u64, TileKey)],
        (directory, base, generation): (&wgpu::Buffer, u32, u64),
    ) -> Vec<VtUniform> {
        self.stats = VtStats::default();
        self.bakes.list.clear();
        self.frame += 1;
        self.bakes_per_frame = settings.virtual_texture_bakes;
        let per_row = maps.size.0 / SLOT_TEXELS;
        let budget = settings.virtual_texture_pages.unwrap_or(u32::MAX);
        let capacity = (maps.sheets * per_row * per_row).min(budget);
        self.limit = (maps.sheet_room(device) * maps.per_sheet()).min(budget);
        self.enabled = settings.enabled && settings.virtual_texture && capacity > 0;
        if maps.generation != self.maps_generation || capacity < self.capacity {
            self.maps_generation = maps.generation;
            self.per_row = per_row;
            self.per_sheet = per_row * per_row;
            self.capacity = capacity;
            self.materials = maps.materials;
            self.sheet_texels = maps.size.0.max(1);
            self.reset();
        } else if capacity > self.capacity {
            // Sheets grown, their pages copied: the new slots free.
            self.owners.resize(capacity as usize, None);
            self.asked.resize(capacity as usize, 0);
            self.baked_from.resize(capacity as usize, None);
            self.changed.resize(capacity as usize, 0);
            self.free.splice(0..0, (self.capacity..capacity).rev());
            self.capacity = capacity;
        }
        if !self.enabled {
            if !self.terrains.is_empty() {
                self.reset();
            }
            return vec![([0; 4], [0.0; 4], [0; 4]); terrains.terrains.len()];
        }
        // Textured terrains: those drawing whose sky maps hold still.
        let mut uniforms = Vec::with_capacity(terrains.terrains.len());
        let mut kept = Vec::new();
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let Some(sky_build) = sky.settled(index) else {
                uniforms.push(([0; 4], [0.0; 4], [0; 4]));
                continue;
            };
            if terrain.patches.is_empty() {
                uniforms.push(([0; 4], [0.0; 4], [0; 4]));
                continue;
            }
            let vt_index = kept.len() as u32;
            kept.push(terrain.id);
            let textured = self
                .terrains
                .entry(terrain.id)
                .or_insert_with(Textured::new);
            textured.index = vt_index;
            if textured.sky != sky_build {
                textured.sky = sky_build;
                let drop: Vec<u32> = self
                    .resident
                    .iter()
                    .filter(|(key, _)| key.terrain == terrain.id)
                    .map(|(_, &slot)| slot)
                    .collect();
                for slot in drop {
                    self.release(slot);
                }
            }
            uniforms.push((
                [
                    vt_index,
                    MIPS as u32,
                    self.per_row,
                    (self.frame % 64) as u32,
                ],
                [TEXEL, 1.0 / self.sheet_texels as f32, 0.0, 0.0],
                [
                    base + vt_index * TERRAIN_WORDS as u32,
                    self.materials,
                    self.materials,
                    0,
                ],
            ));
        }
        // Terrains no longer textured leave.
        let gone: Vec<u64> = self
            .terrains
            .keys()
            .filter(|id| !kept.contains(id))
            .copied()
            .collect();
        for id in gone {
            let drop: Vec<u32> = self
                .resident
                .iter()
                .filter(|(key, _)| key.terrain == id)
                .map(|(_, &slot)| slot)
                .collect();
            for slot in drop {
                self.release(slot);
            }
            self.terrains.remove(&id);
        }
        self.stale_changed(terrains, changed);
        self.move_windows(terrains, eye);
        self.ask(terrains, eye);
        self.write(device, queue, (directory, base, generation), kept.len());
        self.stats.slots = self.capacity;
        self.stats.pages = self.resident.len() as u32;
        uniforms
    }

    /// Marks stale the pages over tiles that arrived, left or were edited.
    fn stale_changed(&mut self, terrains: &RenderTerrains, changed: &[(u64, TileKey)]) {
        for &(id, key) in changed {
            if !self.terrains.contains_key(&id) {
                continue;
            }
            let Some(terrain) = terrains.terrains.iter().find(|terrain| terrain.id == id) else {
                continue;
            };
            let spacing = terrain.pyramid.spacing(u32::from(key.level));
            let side = spacing * engine_core::terrain_data::TILE_QUADS as f32;
            let margin = spacing * TILE_MARGIN;
            let low = [
                f32::from(key.x) * side - margin,
                f32::from(key.z) * side - margin,
            ];
            let high = [low[0] + side + 2.0 * margin, low[1] + side + 2.0 * margin];
            #[expect(
                clippy::iter_over_hash_type,
                reason = "marks every resident page over the tile, in any order"
            )]
            for (page, &slot) in &self.resident {
                let span = texel_size(usize::from(page.mip)) * PAGE_TEXELS as f32;
                let over = (0..2).all(|axis| {
                    let start = page.page[axis] as f32 * span;
                    start < high[axis] && start + span > low[axis]
                });
                if page.terrain == id && over {
                    self.baked_from[slot as usize] = None;
                    self.changed[slot as usize] = self.frame;
                }
            }
        }
    }

    /// Moves each terrain's windows to center on `eye`, releasing the pages
    /// left outside.
    fn move_windows(&mut self, terrains: &RenderTerrains, eye: Vec3) {
        for terrain in &terrains.terrains {
            let Some(textured) = self.terrains.get_mut(&terrain.id) else {
                continue;
            };
            let local = [eye.x - terrain.origin.x, eye.z - terrain.origin.z];
            let mut moved = Vec::new();
            for mip in 0..MIPS {
                let span = texel_size(mip) * PAGE_TEXELS as f32;
                let first = local.map(|axis| (axis / span).floor() as i32 - WINDOW / 2);
                if first != textured.firsts[mip] {
                    textured.firsts[mip] = first;
                    textured.dirty[mip] = true;
                    moved.push(mip);
                }
            }
            if moved.is_empty() {
                continue;
            }
            let drop: Vec<u32> = self
                .resident
                .iter()
                .filter(|(key, _)| {
                    key.terrain == terrain.id
                        && moved.contains(&usize::from(key.mip))
                        && !inside(textured.firsts[usize::from(key.mip)], key.page)
                })
                .map(|(_, &slot)| slot)
                .collect();
            for slot in drop {
                self.release(slot);
            }
            // Entries left from pages outside the window clear.
            let textured = self.terrains.get_mut(&terrain.id).expect("textured above");
            for &mip in &moved {
                let first = textured.firsts[mip];
                for index in 0..(WINDOW * WINDOW) as usize {
                    let at = mip * (WINDOW * WINDOW) as usize + index;
                    if textured.entries[at] != 0 {
                        let page = page_of(first, index);
                        let key = PageKey {
                            terrain: terrain.id,
                            mip: mip as u8,
                            page,
                        };
                        if !self.resident.contains_key(&key) {
                            textured.entries[at] = 0;
                        }
                    }
                }
            }
        }
    }

    /// Takes the pages pixels asked for, keeping those resident and baking
    /// up to the frame's budget of the absent, coarsest and nearest first,
    /// then of the stale, stalest first.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn ask(&mut self, terrains: &RenderTerrains, eye: Vec3) {
        let arrived = std::mem::take(&mut *self.arrived.borrow_mut());
        let mut wanted: Vec<PageKey> = Vec::new();
        for (frame, words) in arrived {
            let Some(snapshot) = self.snapshots.remove(&frame) else {
                continue;
            };
            for (id, index, firsts) in snapshot {
                let offset = index as usize * WANTED_WORDS;
                for word in 0..WANTED_WORDS {
                    let mut bits = words.get(offset + word).copied().unwrap_or(0);
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let flat = word * 32 + bit;
                        let mip = flat / (WINDOW * WINDOW) as usize;
                        let page = page_of(firsts[mip], flat % (WINDOW * WINDOW) as usize);
                        wanted.push(PageKey {
                            terrain: id,
                            mip: mip as u8,
                            page,
                        });
                    }
                }
            }
        }
        // Snapshots whose read-backs never arrive (failed maps) give way:
        // no more are in flight than the ring holds, however many frames
        // their maps take (paused renders may draw many before one lands).
        while self.snapshots.len() > READBACKS {
            let oldest = self.snapshots.keys().min().copied();
            if let Some(oldest) = oldest {
                self.snapshots.remove(&oldest);
            }
        }
        wanted.sort_unstable_by_key(|key| (key.terrain, key.mip, key.page));
        wanted.dedup();
        let mut missing: Vec<(PageKey, f32)> = Vec::new();
        for key in wanted {
            let Some(textured) = self.terrains.get(&key.terrain) else {
                continue;
            };
            if !inside(textured.firsts[usize::from(key.mip)], key.page) {
                continue;
            }
            if let Some(&slot) = self.resident.get(&key) {
                self.asked[slot as usize] = self.frame;
                continue;
            }
            let Some(terrain) = terrains
                .terrains
                .iter()
                .find(|terrain| terrain.id == key.terrain)
            else {
                continue;
            };
            let span = texel_size(usize::from(key.mip)) * PAGE_TEXELS as f32;
            let center = [
                (key.page[0] as f32 + 0.5) * span + terrain.origin.x - eye.x,
                (key.page[1] as f32 + 0.5) * span + terrain.origin.z - eye.z,
            ];
            missing.push((
                key,
                (center[0] * center[0] + center[1] * center[1]).sqrt() / span,
            ));
        }
        self.stats.missing = missing.len() as u32;
        // Coarse pages serve more pixels, and near ones first.
        missing.sort_by(|a, b| {
            b.0.mip
                .cmp(&a.0.mip)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let absent = missing.len() as u32;
        for (placed, (key, _)) in missing
            .into_iter()
            .take(self.bakes_per_frame as usize)
            .enumerate()
        {
            let Some(slot) = self.slot() else {
                // Out of slots: the sheets grow to a quarter past the pages
                // kept and absent, and pages beyond the most they may hold
                // drop.
                let demand = self.resident.len() as u32 + absent;
                self.want = self.want.max(demand + demand / 4).min(self.limit);
                if self.want <= self.capacity {
                    self.stats.dropped = absent - placed as u32;
                }
                break;
            };
            self.owners[slot as usize] = Some(key);
            self.asked[slot as usize] = self.frame;
            self.resident.insert(key, slot);
            let textured = self.terrains.get_mut(&key.terrain).expect("asked above");
            let mip = usize::from(key.mip);
            textured.entries[entry(mip, key.page)] = slot + 1;
            textured.dirty[mip] = true;
            self.bake(slot, key, eye);
        }
        let room = (self.bakes_per_frame as usize).saturating_sub(self.bakes.list.len());
        if room > 0 {
            let levels: HashMap<u64, Levels> = terrains
                .terrains
                .iter()
                .map(|terrain| {
                    let heights = terrain
                        .patches
                        .iter()
                        .fold([f32::MAX, f32::MIN], |[bottom, top], patch| {
                            [min(bottom, patch.min.y), max(top, patch.max.y)]
                        });
                    let levels = Levels {
                        origin: terrain.origin,
                        reach: &terrain.reach,
                        heights,
                    };
                    (terrain.id, levels)
                })
                .collect();
            let frame = self.frame;
            let mut stale: Vec<(u32, PageKey, f32)> = self
                .owners
                .iter()
                .enumerate()
                .filter_map(|(slot, key)| {
                    let key = (*key)?;
                    if self.asked[slot] + KEEP_FRAMES <= frame {
                        return None;
                    }
                    let levels = levels.get(&key.terrain)?;
                    let turn = match self.baked_from[slot] {
                        Some(from) => levels.turn(&key, from, eye),
                        None if self.changed[slot] + SETTLE_FRAMES <= frame => f32::INFINITY,
                        None => return None,
                    };
                    (turn > STALE_SHADING).then_some((slot as u32, key, turn))
                })
                .collect();
            stale.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for (slot, key, _) in stale.into_iter().take(room) {
                self.bake(slot, key, eye);
            }
        }
        self.stats.bakes = self.bakes.list.len() as u32;
    }

    /// Bakes `key`'s page into `slot` from `eye` this frame.
    fn bake(&mut self, slot: u32, key: PageKey, eye: Vec3) {
        self.baked_from[slot as usize] = Some(eye);
        self.bakes.list.push((
            slot / self.per_sheet,
            key.terrain,
            BakeRecord {
                slot,
                mip: u32::from(key.mip),
                page: key.page,
            },
        ));
    }

    /// A free slot, or the least recently asked for once it has gone
    /// unasked for `KEEP_FRAMES`.
    fn slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let (slot, &asked) = self
            .asked
            .iter()
            .enumerate()
            .min_by_key(|(_, asked)| **asked)?;
        if asked + KEEP_FRAMES > self.frame {
            return None;
        }
        self.release(slot as u32);
        self.stats.evictions += 1;
        self.free.pop()
    }

    /// Writes the windows and changed tables into the directory buffer at
    /// word `base` (all of them into a new buffer or at a new place), and
    /// this frame's bakes, sorted by sheet and terrain, into their buffer.
    fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (directory, base, generation): (&wgpu::Buffer, u32, u64),
        textured: usize,
    ) {
        let whole = generation != self.directory_generation || base != self.base;
        self.directory_generation = generation;
        self.base = base;
        let room = directory.size() / 4;
        for terrain in self.terrains.values_mut() {
            let start = u64::from(base) + u64::from(terrain.index) * TERRAIN_WORDS as u64;
            if start + TERRAIN_WORDS as u64 > room || (terrain.index as usize) >= textured {
                continue;
            }
            let firsts: Vec<u32> = terrain
                .firsts
                .iter()
                .flat_map(|first| first.map(|axis| axis as u32))
                .collect();
            queue.write_buffer(directory, start * 4, bytemuck::cast_slice(&firsts));
            for mip in 0..MIPS {
                if !(whole || terrain.dirty[mip]) {
                    continue;
                }
                let table = mip * (WINDOW * WINDOW) as usize;
                let offset = (start + (2 * MIPS + table) as u64) * 4;
                queue.write_buffer(
                    directory,
                    offset,
                    bytemuck::cast_slice(
                        &terrain.entries[table..table + (WINDOW * WINDOW) as usize],
                    ),
                );
                terrain.dirty[mip] = false;
            }
        }
        let bakes = &mut self.bakes;
        bakes
            .list
            .sort_unstable_by_key(|(sheet, terrain, record)| (*sheet, *terrain, record.slot));
        bakes.records.clear();
        bakes
            .records
            .extend(bakes.list.iter().map(|(_, _, record)| *record));
        if bakes.records.is_empty() {
            return;
        }
        let bytes = std::mem::size_of_val(bakes.records.as_slice()) as u64;
        bakes.buffer.grow(device, bytes);
        queue.write_buffer(&bakes.buffer, 0, bytemuck::cast_slice(&bakes.records));
    }

    /// Whether this frame bakes pages.
    pub(crate) fn baking(&self) -> bool {
        !self.bakes.list.is_empty()
    }

    /// Whether any terrain is textured this frame, which asks for pages.
    pub(crate) fn active(&self) -> bool {
        self.enabled && !self.terrains.is_empty()
    }

    /// Records this frame's bakes: a pass per sheet drawing its pages, each
    /// terrain's with its uniform (`uniform_of` gives a terrain's offset by
    /// its id), through the camera's terrain group, the terrain group whose
    /// material maps leave the sheets out, and the bakes' own group.
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    pub(crate) fn encode_bakes(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        (camera_layout, camera_group): (&crate::pipelines::GroupLayout, &wgpu::BindGroup),
        (terrain_layout, terrain_group): (&crate::pipelines::GroupLayout, &wgpu::BindGroup),
        heights: &wgpu::TextureView,
        maps: &crate::terrain_maps::TerrainMaps,
        uniform_of: impl Fn(u64) -> Option<u32>,
    ) {
        if self.bakes.list.is_empty() {
            return;
        }
        let pipeline = self.bake_pipeline.get_or_insert_with(|| {
            let layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_VT_BAKE_PIPELINES,
                &[camera_layout, terrain_layout, &self.bake_layout],
            );
            let formats = [
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm,
            ];
            let targets = formats.map(|format| Some(wgpu::ColorTargetState::from(format)));
            crate::pipelines::fullscreen(
                device,
                &crate::pipelines::Fullscreen {
                    label: "engine/terrain-vt-bake",
                    layout: &layout,
                    module: shader,
                    vertex: meshes::vs_terrain_vt_bake,
                    fragment: meshes::fs_terrain_vt_bake,
                    constants: &[],
                    targets: &targets,
                    depth: None,
                },
            )
        });
        let group = crate::pipelines::group(
            device,
            "engine/terrain-vt-bake",
            &TERRAIN_VT_BAKE,
            &self.bake_layout,
            &[
                crate::buffers::entry("terrain_vt_bakes", &self.bakes.buffer),
                ("terrain_vt_heights", crate::pipelines::view(heights)),
            ],
        );
        let mut start = 0;
        while start < self.bakes.list.len() {
            let sheet = self.bakes.list[start].0;
            let end = start
                + self.bakes.list[start..]
                    .iter()
                    .take_while(|(other, _, _)| *other == sheet)
                    .count();
            let views = maps.sheet_views(sheet);
            let attachments = views.each_ref().map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/terrain-vt-bake"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera_group, &[]);
            pass.set_bind_group(2, &group, &[]);
            let mut first = start;
            while first < end {
                let terrain = self.bakes.list[first].1;
                let last = first
                    + self.bakes.list[first..end]
                        .iter()
                        .take_while(|(_, other, _)| *other == terrain)
                        .count();
                if let Some(uniform) = uniform_of(terrain) {
                    pass.set_bind_group(1, terrain_group, &[0, uniform]);
                    pass.draw(0..6, first as u32..last as u32);
                }
                first = last;
            }
            drop(pass);
            start = end;
        }
    }

    /// Records the feedback pass over terrain's visibility target `ids`
    /// (its size `size`) for each textured terrain (`uniform_of` gives its
    /// uniform's offset), then copies the bits to a read-back and clears
    /// them.
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    pub(crate) fn encode_feedback(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::Buffer,
        (views, uniforms, directory): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        (ids, size): (&wgpu::TextureView, (u32, u32)),
        uniform_of: impl Fn(u64) -> Option<u32>,
    ) {
        if !self.active() {
            return;
        }
        let words = self.terrains.len() * WANTED_WORDS;
        if words != self.wanted_words {
            self.wanted = crate::buffers::storage(
                device,
                "engine/terrain-vt-wanted",
                (words * 4) as u64,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.wanted_words = words;
            self.readbacks.clear();
            encoder.clear_buffer(&self.wanted, 0, None);
        }
        let layouts = &self.feedback_layouts;
        let pipeline = self.feedback_pipeline.get_or_insert_with(|| {
            let layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_VT_FEEDBACK_PIPELINES,
                &[&layouts[0], &layouts[1], &layouts[2]],
            );
            crate::pipelines::compute_entry(device, &layout, shader, meshes::cs_terrain_vt_feedback)
        });
        let camera_group = crate::pipelines::group(
            device,
            "engine/terrain-vt-camera",
            &TERRAIN_VT_CAMERA,
            &layouts[0],
            &[crate::buffers::entry("camera", camera)],
        );
        let terrain_group = crate::pipelines::group(
            device,
            "engine/terrain-vt-terrain",
            &TERRAIN_VT_TERRAIN,
            &layouts[1],
            &[
                (
                    "terrain_view",
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: views,
                        offset: 0,
                        size: wgpu::BufferSize::new(144),
                    }),
                ),
                (
                    "terrain",
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(crate::terrain::TERRAIN_UNIFORM_BYTES as u64),
                    }),
                ),
                crate::buffers::entry("terrain_directory", directory),
            ],
        );
        let feedback_group = crate::pipelines::group(
            device,
            "engine/terrain-vt-feedback",
            &TERRAIN_VT_FEEDBACK,
            &layouts[2],
            &[
                ("terrain_ids_texture", crate::pipelines::view(ids)),
                crate::buffers::entry("terrain_vt_wanted", &self.wanted),
            ],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-vt-feedback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &camera_group, &[]);
            pass.set_bind_group(2, &feedback_group, &[]);
            for &id in self.terrains.keys() {
                let Some(uniform) = uniform_of(id) else {
                    continue;
                };
                pass.set_bind_group(1, &terrain_group, &[0, uniform]);
                // Each invocation tests one pixel of an 8 by 8 block.
                let span = VT_FEEDBACK_GROUP * VT_FEEDBACK_BLOCK;
                pass.dispatch_workgroups(size.0.div_ceil(span), size.1.div_ceil(span), 1);
            }
        }
        let snapshot: Snapshot = self
            .terrains
            .iter()
            .map(|(&id, textured)| (id, textured.index, textured.firsts))
            .collect();
        if self.readbacks.record(
            device,
            encoder,
            &self.wanted,
            (words * 4) as u64,
            self.frame,
        ) {
            encoder.clear_buffer(&self.wanted, 0, None);
            self.snapshots.insert(self.frame, snapshot);
        }
    }

    /// Maps the frame's read-back once submitted; its words wait for the
    /// next frame's [`Self::prepare`].
    pub(crate) fn after_submit(&self) {
        let arrived = Rc::clone(&self.arrived);
        let words = self.wanted_words;
        self.readbacks.after_submit(move |read, frame| {
            let copied: Vec<u32> = (0..words).map(|index| read.get(index)).collect();
            arrived.borrow_mut().push((frame, copied));
        });
    }

    /// Bytes of its buffers (the sheets count with the material maps).
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.wanted.size() * 4 + self.bakes.buffer.size()
    }
}
