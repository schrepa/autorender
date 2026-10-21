//! Far pages: each textured terrain's pages at its coarse mips, the same
//! whatever the eye (their bake reads the terrain's finest pinned level
//! alone, every material at its maps' means), as one record per terrain
//! that the host keeps and the cook's pack ships. Pages of every mip from
//! the finest whose texels span eight meters or more and whose pages along
//! the terrain's longer side number [`FAR_ACROSS`] at most pin their slots
//! (where they take at most half the slots the sheets may hold): they stay
//! while the terrain is textured, never go stale and never give way to
//! other pages, so far ground shows textured from its first frames and only
//! near pages, which follow the eye, bake as the view moves.
//!
//! A record's key holds the hash of the renderer's code and of everything
//! its pages read ([`FarSource`], `terrain/far.rs`). A host holding it hands
//! it back and its pages upload as they are; one that does not has them
//! baked in place by the bake near pages take, [`FAR_BAKES`] a frame, read
//! back once all are and kept: the cook's session makes them so. Pages are
//! made on the GPU, so the same record made on another differs within its
//! rounding: records are cooked once, never compared bit for bit across
//! GPUs, and near and far pages blend over a mip as the ground's own levels
//! do. A verifying host has those it handed back baked again and kept,
//! which it compares within the records' parity ([`FAR_PARITY`], `alike`)
//! as its one verify path compares every record.
//!
//! A record holds its pages' count, each page's mip and page along x and z,
//! then their texels: each page's three layers (base color, canopy, normal
//! turn), each layer's channels as planes of [`SLOT_TEXELS`]² bytes,
//! written as differences from what their neighbours predict
//! (`codec::put_planes`) for the cook's compression.

use super::*;
use engine_core::RecordKey;
use engine_core::terrain_data::codec::{self, Reader, Writer};

/// The finest mip far pages begin at: texels of eight meters.
const FAR_FINEST: usize = 11;
/// Pages along a terrain's longer side its finest far mip holds at most.
const FAR_ACROSS: u32 = 8;
/// Far pages baked in a frame at most, beside near pages.
const FAR_BAKES: usize = 32;
/// The parity pages baked again keep with a record handed back: a peak
/// signal-to-noise ratio of 40 dB over every byte, the mean squared
/// difference at most 255² over this.
const FAR_PARITY: u64 = 10_000;
/// The kind far page records' keys name.
const KIND: &str = "terrain_far_pages";
/// Layers a page takes: its sheet's base color, canopy and normal turn.
const LAYERS: usize = crate::terrain_maps::SHEET_LAYERS as usize;
/// Bytes of a layer of a slot, and of its rows as read back (whole multiples
/// of what copies to buffers align rows to).
const LAYER_BYTES: usize = (SLOT_TEXELS * SLOT_TEXELS * 4) as usize;
const READ_ROW: u32 = (SLOT_TEXELS * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

/// What a terrain's far pages are baked from: the level they read, its
/// finest pinned, and the hash of everything their texels read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FarSource {
    pub(crate) level: u32,
    pub(crate) identity: u128,
}

/// How the host's records stand this frame: whether it verifies the records
/// it hands back, and whether textures still load (sets wait until they
/// show their own).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RecordMode {
    pub(crate) verifying: bool,
    pub(crate) loading: bool,
}

/// Where a terrain's far pages stand: asked for from the host; handed back,
/// their texels uploading from the page given; missing, baking from the page
/// given; baked, their slots read back; or resident.
enum State {
    Asking,
    Uploading(Vec<u8>, usize),
    Baking(usize),
    Reading,
    Here,
}

/// A terrain's far pages: their record's key, the level their bakes read,
/// the pages in the record's order, where they stand, and whether their
/// bakes are kept once read back (a missing record's, and those a verifying
/// host handed back).
struct FarSet {
    key: RecordKey,
    level: u32,
    pages: Vec<PageKey>,
    state: State,
    keep: bool,
}

/// A set's slots read back: the record's key and pages, the slots in their
/// order, the buffer, whether a frame copied the slots into it and asked
/// for its map, and what the map gave.
struct Reading {
    key: RecordKey,
    pages: Vec<PageKey>,
    slots: Vec<u32>,
    buffer: wgpu::Buffer,
    copied: bool,
    mapping: bool,
    mapped: Rc<RefCell<Option<Result<Vec<u8>, String>>>>,
}

/// Every textured terrain's far pages, by terrain id; which slots they pin;
/// the host's records as they stand; the records to fetch and those made,
/// to keep, until the host takes them, and the answers it gave (`None`
/// where it holds none); the read-back under way; and the textured
/// terrains whose sets wait for what they are baked from.
#[derive(Default)]
pub(crate) struct FarPages {
    sets: BTreeMap<u64, FarSet>,
    pinned: Vec<bool>,
    mode: RecordMode,
    fetches: Vec<RecordKey>,
    keeps: Vec<(RecordKey, Vec<u8>)>,
    answers: Vec<(RecordKey, Option<Vec<u8>>)>,
    reading: Option<Reading>,
    waiting: u64,
}

impl FarPages {
    /// Whether `slot` holds a far page.
    pub(super) fn pinned(&self, slot: u32) -> bool {
        self.pinned.get(slot as usize).copied().unwrap_or(false)
    }

    /// `slot` holds a far page no longer.
    pub(super) fn unpin(&mut self, slot: u32) {
        if let Some(pinned) = self.pinned.get_mut(slot as usize) {
            *pinned = false;
        }
    }

    /// Slots holding far pages.
    pub(super) fn resident(&self) -> u32 {
        self.pinned.iter().filter(|pinned| **pinned).count() as u32
    }

    /// Every slot dropped: the sets ask again, and none waits.
    pub(super) fn reset(&mut self) {
        self.sets.clear();
        self.pinned.clear();
        self.reading = None;
        self.waiting = 0;
    }

    /// Takes the host's records as they stand.
    pub(super) fn set_mode(&mut self, mode: RecordMode) {
        self.mode = mode;
    }

    /// Whether a set asked for the record `key`.
    fn asks(&self, key: RecordKey) -> bool {
        self.sets
            .values()
            .any(|set| set.key == key && matches!(set.state, State::Asking))
    }

    /// Far page sets a cooking session has left to make: those not
    /// resident yet, and the textured terrains' waiting for what they are
    /// baked from.
    pub(super) fn left(&self) -> u64 {
        let unmade = self
            .sets
            .values()
            .filter(|set| !matches!(set.state, State::Here))
            .count() as u64;
        unmade + self.waiting
    }

    /// Whether a set waits for the host's answer: far ground shows textured
    /// once its record uploads, which preparation waits for.
    pub(super) fn loading(&self) -> bool {
        self.sets
            .values()
            .any(|set| matches!(set.state, State::Asking))
    }

    /// Whether this frame copies a set's slots to read back, which runs the
    /// bakes' pass.
    pub(super) fn copying(&self) -> bool {
        self.reading.as_ref().is_some_and(|reading| !reading.copied)
    }

    /// Records the copies of the read-back's slots, after this frame's bakes,
    /// from `sheets`, `per_row` slots along a sheet's side and `per_sheet` a
    /// sheet.
    pub(super) fn copy_slots(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sheets: &wgpu::Texture,
        (per_row, per_sheet): (u32, u32),
    ) {
        let Some(reading) = self.reading.as_mut().filter(|reading| !reading.copied) else {
            return;
        };
        for (index, &slot) in reading.slots.iter().enumerate() {
            let within = slot % per_sheet;
            let corner = [
                within % per_row * SLOT_TEXELS,
                within / per_row * SLOT_TEXELS,
            ];
            for layer in 0..LAYERS {
                let at = (index * LAYERS + layer) as u64 * u64::from(READ_ROW * SLOT_TEXELS);
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: sheets,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: corner[0],
                            y: corner[1],
                            z: slot / per_sheet * LAYERS as u32 + layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &reading.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: at,
                            bytes_per_row: Some(READ_ROW),
                            rows_per_image: Some(SLOT_TEXELS),
                        },
                    },
                    wgpu::Extent3d {
                        width: SLOT_TEXELS,
                        height: SLOT_TEXELS,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        reading.copied = true;
    }

    /// Maps the read-back copied in the frame just submitted; its bytes
    /// wait for a later frame.
    pub(super) fn after_submit(&mut self) {
        let Some(reading) = self
            .reading
            .as_mut()
            .filter(|reading| reading.copied && !reading.mapping)
        else {
            return;
        };
        reading.mapping = true;
        let (buffer, mapped) = (reading.buffer.clone(), Rc::clone(&reading.mapped));
        reading.buffer.slice(..).map_async(
            wgpu::MapMode::Read,
            crate::platform::map_callback(move |result| {
                let bytes = result.map_err(|error| error.to_string()).and_then(|()| {
                    let view = buffer
                        .slice(..)
                        .get_mapped_range()
                        .map_err(|error| error.to_string())?;
                    let bytes = view.to_vec();
                    drop(view);
                    buffer.unmap();
                    Ok(bytes)
                });
                *mapped.borrow_mut() = Some(bytes);
            }),
        );
    }
}

/// The pages along x and z of mip `mip` over a terrain of `extent`.
fn far_across(extent: engine_core::glam::Vec2, mip: usize) -> [i32; 2] {
    let span = texel_size(mip) * PAGE_TEXELS as f32;
    [extent.x, extent.y].map(|side| ((side / span).ceil() as i32).max(1))
}

/// The finest far mip of a terrain of `extent`: from [`FAR_FINEST`], the
/// first whose pages along the longer side number [`FAR_ACROSS`] at most;
/// none where no mip's do.
fn far_mip(extent: engine_core::glam::Vec2) -> Option<usize> {
    (FAR_FINEST..MIPS).find(|&mip| {
        far_across(extent, mip)
            .iter()
            .all(|&pages| pages <= FAR_ACROSS as i32)
    })
}

/// How many far pages a terrain of `extent` takes from mip `first`.
fn far_count(extent: engine_core::glam::Vec2, first: usize) -> usize {
    (first..MIPS)
        .map(|mip| {
            let [x, z] = far_across(extent, mip);
            (x * z) as usize
        })
        .sum()
}

/// Terrain `id`'s far pages over `extent` from mip `first`, coarsest mip
/// first, rows along z.
fn far_pages(id: u64, extent: engine_core::glam::Vec2, first: usize) -> Vec<PageKey> {
    let mut pages = Vec::with_capacity(far_count(extent, first));
    for mip in (first..MIPS).rev() {
        let across = far_across(extent, mip);
        for z in 0..across[1] {
            for x in 0..across[0] {
                pages.push(PageKey {
                    terrain: id,
                    mip: mip as u8,
                    page: [x, z],
                });
            }
        }
    }
    pages
}

/// The key of the record of far pages baked from `source`.
fn far_key(source: FarSource) -> RecordKey {
    let mut identity = source.identity.to_le_bytes().to_vec();
    identity.extend_from_slice(&source.level.to_le_bytes());
    RecordKey::new(&crate::SOURCE, KIND, &identity)
}

/// A far page record of `pages` whose slots' layers `texels` holds, page by
/// page, layer by layer, as RGBA8 rows.
fn encode(pages: &[PageKey], texels: &[u8]) -> Vec<u8> {
    let mut out = Writer::new();
    out.put(&(pages.len() as u32));
    for page in pages {
        out.put(&page.mip)
            .put(&(page.page[0] as u16))
            .put(&(page.page[1] as u16));
    }
    // Each layer's channels apart, as planes.
    let area = (SLOT_TEXELS * SLOT_TEXELS) as usize;
    let mut planes = vec![0_u8; texels.len()];
    for (layer, texels) in texels.chunks_exact(LAYER_BYTES).enumerate() {
        for (at, texel) in texels.chunks_exact(4).enumerate() {
            for (channel, &value) in texel.iter().enumerate() {
                planes[(layer * 4 + channel) * area + at] = value;
            }
        }
    }
    codec::put_planes(&mut out, &planes, SLOT_TEXELS as usize);
    out.into_bytes()
}

/// Whether texels `baked` keep [`FAR_PARITY`] with texels `was`, as many.
fn within_parity(was: &[u8], baked: &[u8]) -> bool {
    let squared: u64 = was
        .iter()
        .zip(baked)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
        .sum();
    was.len() == baked.len() && squared * FAR_PARITY <= 255 * 255 * was.len() as u64
}

/// Whether far page records `handed` and `made` hold the same pages, their
/// texels within [`FAR_PARITY`]: a verifying host's comparison of pages
/// baked again with the record it handed back
/// (`engine_core::RecordParity::Within`).
fn alike(handed: &[u8], made: &[u8]) -> bool {
    match (read(handed), read(made)) {
        (Ok((pages, was)), Ok((again, baked))) => pages == again && within_parity(&was, &baked),
        _ => false,
    }
}

/// The texels of a far page record of `pages`, as [`encode`] took them; why
/// it holds none where its pages are others.
fn decode(bytes: &[u8], pages: &[PageKey]) -> Result<Vec<u8>, String> {
    let (held, texels) = read(bytes)?;
    let same = held.len() == pages.len()
        && held.iter().zip(pages).all(|(&(mip, x, z), page)| {
            mip == page.mip && i32::from(x) == page.page[0] && i32::from(z) == page.page[1]
        });
    if !same {
        return Err("a far page record of other pages".into());
    }
    Ok(texels)
}

/// A far page record's pages (each one's mip and page along x and z) and
/// its texels, as [`encode`] took them.
fn read(bytes: &[u8]) -> Result<(Vec<(u8, u16, u16)>, Vec<u8>), String> {
    let invalid = |error: codec::CodecError| error.to_string();
    let mut input = Reader::new(bytes);
    let count: u32 = input.get().map_err(invalid)?;
    let mut pages = Vec::with_capacity((count as usize).min(bytes.len()));
    for _ in 0..count {
        let mip: u8 = input.get().map_err(invalid)?;
        let x: u16 = input.get().map_err(invalid)?;
        let z: u16 = input.get().map_err(invalid)?;
        pages.push((mip, x, z));
    }
    let planes = codec::get_planes(&mut input, SLOT_TEXELS as usize).map_err(invalid)?;
    input.finish().map_err(invalid)?;
    if planes.len() != pages.len() * LAYERS * LAYER_BYTES {
        return Err("a far page record of another size".into());
    }
    let area = (SLOT_TEXELS * SLOT_TEXELS) as usize;
    let mut texels = vec![0_u8; planes.len()];
    for (layer, texels) in texels.chunks_exact_mut(LAYER_BYTES).enumerate() {
        for (at, texel) in texels.chunks_exact_mut(4).enumerate() {
            for (channel, value) in texel.iter_mut().enumerate() {
                *value = planes[(layer * 4 + channel) * area + at];
            }
        }
    }
    Ok((pages, texels))
}

impl VirtualTexture {
    /// The host's records as they stand this frame.
    pub(crate) fn set_records(&mut self, mode: RecordMode) {
        self.far.set_mode(mode);
    }

    /// Moves the far pages' records to fetch into `out`.
    pub(crate) fn take_wants(&mut self, out: &mut Vec<RecordKey>) {
        out.append(&mut self.far.fetches);
    }

    /// Whether far pages asked for the record `key`.
    pub(crate) fn asks(&self, key: RecordKey) -> bool {
        self.far.asks(key)
    }

    /// The host's answer for far pages' record `key`: its bytes, or `None`
    /// where it holds none, which bakes the pages.
    pub(crate) fn answer(&mut self, key: RecordKey, bytes: Option<Vec<u8>>) {
        self.far.answers.push((key, bytes));
    }

    /// Moves the far pages' records made, for the host to keep, into
    /// `out`.
    pub(crate) fn take_made(
        &mut self,
        out: &mut Vec<(RecordKey, Vec<u8>, engine_core::RecordParity)>,
    ) {
        out.extend(
            self.far
                .keeps
                .drain(..)
                .map(|(key, bytes)| (key, bytes, engine_core::RecordParity::Within(alike))),
        );
    }

    /// Far page sets a cooking session has left to make.
    pub(crate) fn records_left(&self) -> u64 {
        self.far.left()
    }

    /// Whether far pages still load.
    pub(crate) fn records_loading(&self) -> bool {
        self.far.loading()
    }

    /// This frame's far pages, after the near pages' asking: sets of
    /// textured `terrains` whose `sources` changed ask for their records
    /// anew (once textures show their own), the host's answers upload or
    /// start their bakes, up to [`FAR_BAKES`] pages of sets baking bake
    /// into slots they pin, and a set baked whole is read back and kept;
    /// every far page resident has its table entry where its window holds
    /// it. `maps` holds the sheets they upload into.
    pub(super) fn far_pages(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        terrains: &RenderTerrains,
        sources: &[Option<FarSource>],
        maps: &crate::terrain_maps::TerrainMaps,
    ) {
        // Sets of terrains textured no longer went with their slots.
        let textured = &self.terrains;
        self.far.sets.retain(|id, _| textured.contains_key(id));
        self.far.waiting = 0;
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let extent = terrain.pyramid.extent();
            let (true, Some(first)) = (self.terrains.contains_key(&terrain.id), far_mip(extent))
            else {
                continue;
            };
            // Far pages take half the slots the sheets may hold at most,
            // leaving near pages the rest.
            if 2 * far_count(extent, first) > self.limit as usize {
                continue;
            }
            let source = sources.get(index).copied().flatten();
            let Some(source) = source.filter(|_| !self.far.mode.loading) else {
                // Surroundings may leave their pinned tiles to the ground
                // they surround: they never count as waiting.
                if !terrain.surroundings && !self.far.sets.contains_key(&terrain.id) {
                    self.far.waiting += 1;
                }
                continue;
            };
            let key = far_key(source);
            if self
                .far
                .sets
                .get(&terrain.id)
                .is_some_and(|set| set.key == key)
            {
                continue;
            }
            self.unpin_terrain(terrain.id);
            self.far.fetches.push(key);
            self.far.sets.insert(
                terrain.id,
                FarSet {
                    key,
                    level: source.level,
                    pages: far_pages(terrain.id, extent, first),
                    state: State::Asking,
                    keep: false,
                },
            );
        }
        // The host's answers: records upload, missing ones bake.
        for (key, answer) in std::mem::take(&mut self.far.answers) {
            let verifying = self.far.mode.verifying;
            let Some(set) = self
                .far
                .sets
                .values_mut()
                .find(|set| set.key == key && matches!(set.state, State::Asking))
            else {
                continue;
            };
            let decoded = answer.and_then(|bytes| decode(&bytes, &set.pages).ok());
            match decoded {
                Some(texels) => {
                    set.keep = verifying;
                    set.state = State::Uploading(texels, 0);
                }
                None => {
                    set.keep = true;
                    set.state = State::Baking(0);
                }
            }
        }
        // A read-back no set waits for any longer (its set replaced or
        // gone) gives way.
        let sets = &self.far.sets;
        let wanted = self.far.reading.as_ref().is_none_or(|reading| {
            sets.values()
                .any(|set| set.key == reading.key && matches!(set.state, State::Reading))
        });
        if !wanted {
            self.far.reading = None;
        }
        let ids: Vec<u64> = self.far.sets.keys().copied().collect();
        let mut room = FAR_BAKES;
        for id in ids {
            let Some(mut set) = self.far.sets.remove(&id) else {
                continue;
            };
            set.state = match std::mem::replace(&mut set.state, State::Here) {
                State::Uploading(texels, next) => self.upload(queue, maps, &set, (texels, next)),
                State::Baking(next) => self.bake_far(device, &set, next, &mut room),
                State::Reading => self.read_far(&set),
                state @ (State::Asking | State::Here) => state,
            };
            self.entries_far(&set);
            self.far.sets.insert(id, set);
        }
    }

    /// Uploads `set`'s pages from `texels` from page `next` on into slots
    /// they pin in `maps`' sheets: where the sheets run out of slots, the
    /// rest wait for them to grow. A verifying host's pages bake again.
    fn upload(
        &mut self,
        queue: &wgpu::Queue,
        maps: &crate::terrain_maps::TerrainMaps,
        set: &FarSet,
        (texels, mut next): (Vec<u8>, usize),
    ) -> State {
        while let Some(&page) = set.pages.get(next) {
            let Some(slot) = self.pin(page, set.pages.len() - next) else {
                return State::Uploading(texels, next);
            };
            // A near bake of the slot this frame would draw over it.
            self.bakes.list.retain(|(_, _, record)| record.slot != slot);
            let place = [slot / self.per_sheet, slot % self.per_sheet];
            let corner = [
                place[1] % self.per_row * SLOT_TEXELS,
                place[1] / self.per_row * SLOT_TEXELS,
            ];
            for layer in 0..LAYERS {
                let at = (next * LAYERS + layer) * LAYER_BYTES;
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: maps.sheet_texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: corner[0],
                            y: corner[1],
                            z: place[0] * LAYERS as u32 + layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &texels[at..at + LAYER_BYTES],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(SLOT_TEXELS * 4),
                        rows_per_image: Some(SLOT_TEXELS),
                    },
                    wgpu::Extent3d {
                        width: SLOT_TEXELS,
                        height: SLOT_TEXELS,
                        depth_or_array_layers: 1,
                    },
                );
            }
            next += 1;
        }
        if set.keep {
            State::Baking(0)
        } else {
            State::Here
        }
    }

    /// Bakes `set`'s pages from page `next` on into slots they pin, while
    /// `room` of the frame's far bakes lasts; once all are baked, a set
    /// whose bakes are kept is read back (once no other is).
    fn bake_far(
        &mut self,
        device: &wgpu::Device,
        set: &FarSet,
        mut next: usize,
        room: &mut usize,
    ) -> State {
        while let Some(&page) = set.pages.get(next) {
            if *room == 0 {
                return State::Baking(next);
            }
            let Some(slot) = self.pin(page, set.pages.len() - next) else {
                return State::Baking(next);
            };
            // A near bake of the slot this frame gives way.
            self.bakes.list.retain(|(_, _, record)| record.slot != slot);
            self.bakes.list.push((
                slot / self.per_sheet,
                page.terrain,
                BakeRecord {
                    slot,
                    mip: u32::from(page.mip) | ((set.level + 1) << 8),
                    page: page.page,
                },
            ));
            *room -= 1;
            next += 1;
        }
        if !set.keep {
            return State::Here;
        }
        if self.far.reading.is_some() {
            return State::Baking(next);
        }
        let slots: Vec<u32> = set
            .pages
            .iter()
            .filter_map(|page| self.resident.get(page).copied())
            .collect();
        if slots.len() != set.pages.len() {
            // A page lost its slot meanwhile: they bake again.
            return State::Baking(0);
        }
        let bytes = u64::from(READ_ROW * SLOT_TEXELS) * (LAYERS * slots.len()) as u64;
        self.far.reading = Some(Reading {
            key: set.key,
            pages: set.pages.clone(),
            slots,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/terrain-vt-far-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            copied: false,
            mapping: false,
            mapped: Rc::new(RefCell::new(None)),
        });
        State::Reading
    }

    /// `set`'s read-back once mapped: its pages' texels, unpadded, kept as
    /// its record (a verifying host compares it with the one it handed
    /// back); a read-back that failed bakes them again.
    fn read_far(&mut self, set: &FarSet) -> State {
        let Some(reading) = self
            .far
            .reading
            .as_ref()
            .filter(|reading| reading.key == set.key)
        else {
            // Another set's read-back took its place: it bakes again.
            return State::Baking(0);
        };
        let Some(mapped) = reading.mapped.borrow_mut().take() else {
            return State::Reading;
        };
        let pages = reading.pages.clone();
        self.far.reading = None;
        let Ok(bytes) = mapped else {
            return State::Baking(0);
        };
        let row = (SLOT_TEXELS * 4) as usize;
        let mut texels = Vec::with_capacity(pages.len() * LAYERS * LAYER_BYTES);
        for image in bytes.chunks_exact((READ_ROW * SLOT_TEXELS) as usize) {
            for line in image.chunks_exact(READ_ROW as usize) {
                texels.extend_from_slice(&line[..row]);
            }
        }
        if texels.len() != pages.len() * LAYERS * LAYER_BYTES {
            return State::Baking(0);
        }
        self.far.keeps.push((set.key, encode(&pages, &texels)));
        State::Here
    }

    /// Sets the table entry of each of `set`'s pages resident where its
    /// window holds it.
    fn entries_far(&mut self, set: &FarSet) {
        let terrain = set.pages.first().map_or(0, |page| page.terrain);
        let Some(textured) = self.terrains.get_mut(&terrain) else {
            return;
        };
        for page in &set.pages {
            let Some(&slot) = self.resident.get(page) else {
                continue;
            };
            let mip = usize::from(page.mip);
            if !inside(textured.firsts[mip], page.page) {
                continue;
            }
            let at = entry(mip, page.page);
            if textured.entries[at] != slot + 1 {
                textured.entries[at] = slot + 1;
                textured.dirty[mip] = true;
            }
        }
    }

    /// A slot pinned for far page `key`, of `left` still to pin: the one
    /// holding it already, or a free or evicted one; none where the sheets
    /// hold no room, which they then grow to.
    fn pin(&mut self, key: PageKey, left: usize) -> Option<u32> {
        let slot = match self.resident.get(&key) {
            Some(&slot) => slot,
            None => {
                let Some(slot) = self.slot() else {
                    let demand = self.resident.len() as u32 + left as u32;
                    self.want = self.want.max(demand + demand / 4).min(self.limit);
                    return None;
                };
                self.owners[slot as usize] = Some(key);
                self.resident.insert(key, slot);
                slot
            }
        };
        let index = slot as usize;
        if self.far.pinned.len() <= index {
            self.far.pinned.resize(index + 1, false);
        }
        self.far.pinned[index] = true;
        self.asked[index] = self.frame;
        self.baked_from[index] = None;
        self.changed[index] = 0;
        Some(slot)
    }

    /// Releases every slot terrain `id`'s far pages pin.
    fn unpin_terrain(&mut self, id: u64) {
        let pinned: Vec<u32> = self
            .resident
            .iter()
            .filter(|(key, slot)| key.terrain == id && self.far.pinned(**slot))
            .map(|(_, &slot)| slot)
            .collect();
        for slot in pinned {
            self.release(slot);
        }
    }
}
