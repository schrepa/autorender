use engine_core::terrain_data::{ERROR_CELLS, ERROR_TEXELS, TILE_STRIDE};
use engine_core::{RenderTerrains, TileKey};
use std::collections::BTreeMap;
use std::ops::Range;

use crate::buffers::Growable;
use crate::cached::Bound;
use crate::gpu_data::PoseData;
use crate::gpu_layouts::constants::{DEAD, IMPOSTOR_KINDS, NO_PLACED};
use crate::gpu_layouts::world_trace::{PlacedGroup, TraceTerrain};
use crate::impostors::ImpostorRecord;
use crate::scatter::{Record, TerrainInfo};

pub(crate) fn peaks(heights: &[f32]) -> Box<[f32]> {
    let stride = TILE_STRIDE as usize;
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

pub(crate) fn peaks_top(peaks: &[f32]) -> f32 {
    peaks.last().copied().unwrap_or(-f32::MAX)
}

pub(crate) fn peaks_mip(peaks: &[f32], mip: u32) -> &[f32] {
    let side = |level: u32| (ERROR_CELLS >> level) as usize;
    let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
    &peaks[start..start + side(mip) * side(mip)]
}

pub(crate) struct Placements<'a> {
    pub tiles: Vec<((u64, TileKey), Range<u32>, usize)>,
    pub records: &'a [Record],
    pub poses: &'a [PoseData],
    pub impostors: &'a [ImpostorRecord],
    pub colors: &'a [[u32; 2]],
    pub revision: u64,
}

pub(crate) struct TraceFrame<'a> {
    pub terrains: &'a RenderTerrains,
    pub infos: &'a [TerrainInfo],
    pub terrain: &'a crate::terrain::TerrainPass,
    pub placed: Placements<'a>,
    pub cover: f32,
}

struct Grid {
    words: Range<u32>,
    slots: Range<u32>,
    identity: usize,
    level: u8,
    apron: f32,
    top: f32,
}

struct Reach {
    slot: u32,
    middle: [f32; 2],
    radius: f32,
    top: f32,
}

struct Placeholders {
    _textures: [wgpu::Texture; 3],
    views: [wgpu::TextureView; 3],
    record: wgpu::Buffer,
}

pub(crate) struct WorldTrace {
    terrains: Growable,
    written: Vec<TraceTerrain>,
    groups: Growable,
    written_groups: Vec<PlacedGroup>,
    placed: Growable,
    words: Vec<u32>,
    dirty: Vec<Range<usize>>,
    grids: BTreeMap<(u64, TileKey), Grid>,
    free: Vec<Range<u32>>,
    layers: u32,
    boxes: Vec<ImpostorRecord>,
    synced: Option<(u64, u64)>,
    placeholders: Placeholders,
}

impl WorldTrace {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let placeholder = |label, format, layers| {
            crate::targets::texture_2d(
                device,
                label,
                (1, 1),
                layers,
                1,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let textures = [
            placeholder(
                "engine/post-world-heights",
                crate::terrain::HEIGHT_FORMAT,
                1,
            ),
            placeholder("engine/post-world-control", wgpu::TextureFormat::R32Uint, 1),
            placeholder(
                "engine/post-world-frames",
                wgpu::TextureFormat::Rgba8Unorm,
                IMPOSTOR_KINDS,
            ),
        ];
        let views = textures.each_ref().map(|texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        });
        Self {
            terrains: Growable::new(
                device,
                "engine/post-world-terrains",
                size_of::<TraceTerrain>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            written: Vec::new(),
            groups: Growable::new(
                device,
                "engine/post-world-groups",
                size_of::<PlacedGroup>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            written_groups: Vec::new(),
            placed: Growable::new(
                device,
                "engine/post-world-placed",
                16,
                wgpu::BufferUsages::empty(),
            ),
            words: Vec::new(),
            dirty: Vec::new(),
            grids: BTreeMap::new(),
            free: Vec::new(),
            layers: 0,
            boxes: Vec::new(),
            synced: None,
            placeholders: Placeholders {
                _textures: textures,
                views,
                record: crate::buffers::storage(
                    device,
                    "engine/post-world-record",
                    size_of::<Record>() as u64,
                    wgpu::BufferUsages::empty(),
                ),
            },
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.terrains.size() + self.groups.size() + self.placed.size()
    }

    pub(crate) fn walked(terrains: &RenderTerrains) -> u32 {
        terrains
            .terrains
            .iter()
            .filter(|terrain| terrain.drawn)
            .count() as u32
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: Option<&TraceFrame<'_>>,
    ) {
        let Some(frame) = frame else {
            self.release();
            return;
        };
        let (layers, directory) = frame.terrain.layers();
        let rebuilt = !same_boxes(frame.placed.impostors, &self.boxes);
        if self.synced.is_none() || layers != self.layers || rebuilt {
            self.lay_out(layers, rebuilt);
            if rebuilt {
                self.boxes = frame.placed.impostors.to_vec();
            }
        }
        let revision = frame.placed.revision;
        if self.synced.is_none_or(|(pool, _)| pool != revision) {
            self.sync_grids(frame);
        }
        if self.synced.is_none_or(|(_, tables)| tables != directory) {
            self.write_layer_words(frame.terrain);
        }
        self.synced = Some((revision, directory));
        self.upload(device, queue);
        self.write_groups(device, queue, &frame.placed);
        self.write_terrains(device, queue, frame);
    }

    pub(crate) fn release(&mut self) {
        if self.synced.is_some() {
            self.grids.clear();
            self.free.clear();
            self.words.clear();
            self.dirty.clear();
            self.boxes.clear();
            self.synced = None;
        }
    }

    fn lay_out(&mut self, layers: u32, rebuilt: bool) {
        let mut words = vec![NO_PLACED; layers as usize];
        if rebuilt {
            self.grids.clear();
        }
        for grid in self.grids.values_mut() {
            let moved = words.len() as u32;
            let held = grid.words.start as usize..grid.words.end as usize;
            words.extend_from_slice(&self.words[held]);
            grid.words = moved..words.len() as u32;
        }
        self.words = words;
        self.free.clear();
        self.layers = layers;
        self.dirty = std::iter::once(0..self.words.len()).collect();
        self.synced = self.synced.map(|(pool, _)| (pool, u64::MAX));
        if rebuilt {
            self.synced = None;
        }
    }

    fn sync_grids(&mut self, frame: &TraceFrame<'_>) {
        let placed = &frame.placed;
        let current: BTreeMap<(u64, TileKey), (Range<u32>, usize)> = placed
            .tiles
            .iter()
            .map(|(key, slots, identity)| (*key, (slots.clone(), *identity)))
            .collect();
        let gone: Vec<(u64, TileKey)> = self
            .grids
            .iter()
            .filter(|&(key, grid)| {
                current.get(key).is_none_or(|(slots, identity)| {
                    *slots != grid.slots || *identity != grid.identity
                })
            })
            .map(|(key, _)| *key)
            .collect();
        for key in gone {
            if let Some(grid) = self.grids.remove(&key) {
                self.free_run(grid.words);
            }
        }
        for (key, (slots, identity)) in current {
            if self.grids.contains_key(&key) {
                continue;
            }
            let terrains = &frame.terrains.terrains;
            let Some(terrain) = terrains.iter().find(|terrain| terrain.id == key.0) else {
                continue;
            };
            let reaches = reaches(terrain, key.1, slots.clone(), placed);
            let Some((built, apron)) = grid(&reaches, terrain, key.1) else {
                continue;
            };
            let top = reaches.iter().fold(-f32::MAX, |top, reach| {
                engine_core::math::max(top, reach.top)
            });
            let words = self.place(&built);
            self.grids.insert(
                key,
                Grid {
                    words,
                    slots,
                    identity,
                    level: key.1.level,
                    apron,
                    top,
                },
            );
        }
        let freed: u32 = self.free.iter().map(|run| run.end - run.start).sum();
        let used: u32 = self
            .grids
            .values()
            .map(|grid| grid.words.end - grid.words.start)
            .sum();
        if freed > used {
            self.lay_out(self.layers, false);
        }
        self.synced = self.synced.map(|(pool, _)| (pool, u64::MAX));
    }

    fn place(&mut self, built: &[u32]) -> Range<u32> {
        let count = built.len() as u32;
        let start = match self
            .free
            .iter()
            .position(|run| run.end - run.start >= count)
        {
            Some(index) => {
                let start = self.free[index].start;
                self.free[index].start += count;
                if self.free[index].is_empty() {
                    self.free.remove(index);
                }
                start
            }
            None => {
                let start = self.words.len() as u32;
                self.words.resize(self.words.len() + built.len(), 0);
                start
            }
        };
        let range = start as usize..(start + count) as usize;
        self.words[range.clone()].copy_from_slice(built);
        self.mark(range);
        start..start + count
    }

    fn free_run(&mut self, words: Range<u32>) {
        let at = self.free.partition_point(|run| run.start < words.start);
        self.free.insert(at, words);
        if at + 1 < self.free.len() && self.free[at].end == self.free[at + 1].start {
            self.free[at].end = self.free[at + 1].end;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].end == self.free[at].start {
            self.free[at - 1].end = self.free[at].end;
            self.free.remove(at);
        }
    }

    fn write_layer_words(&mut self, terrain: &crate::terrain::TerrainPass) {
        let count = self.layers;
        let layers = 0..count as usize;
        self.words[layers.clone()].fill(NO_PLACED);
        for (&(id, key), grid) in &self.grids {
            if let Some(layer) = terrain.layer_of(id, key).filter(|&layer| layer < count) {
                self.words[layer as usize] = grid.words.start;
            }
        }
        self.mark(layers);
    }

    fn mark(&mut self, words: Range<usize>) {
        self.dirty.push(words);
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let bytes = (self.words.len() * 4) as u64;
        let mut dirty = std::mem::take(&mut self.dirty);
        if self.placed.grow(device, bytes) {
            dirty = std::iter::once(0..self.words.len()).collect();
        }
        let held = ((self.placed.size() / 4) as usize).min(self.words.len());
        for run in dirty {
            let run = run.start.min(held)..run.end.min(held);
            if !run.is_empty() {
                queue.write_buffer(
                    &self.placed,
                    (run.start * 4) as u64,
                    bytemuck::cast_slice(&self.words[run]),
                );
            }
        }
    }

    fn write_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        placed: &Placements<'_>,
    ) {
        let held = placed.impostors.iter().zip(placed.colors);
        let same =
            self.written_groups.len() == placed.impostors.len()
                && self.written_groups.iter().zip(held.clone()).all(
                    |(group, (impostor, color))| {
                        group.impostor == *impostor && group.color == *color
                    },
                );
        if same {
            return;
        }
        self.written_groups = held
            .map(|(&impostor, &color)| PlacedGroup {
                impostor,
                color,
                pad: [0; 2],
            })
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&self.written_groups);
        self.groups.grow(device, bytes.len() as u64);
        if !bytes.is_empty() {
            queue.write_buffer(&self.groups, 0, bytes);
        }
    }

    fn write_terrains(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &TraceFrame<'_>,
    ) {
        let mut records = Vec::with_capacity(self.written.len());
        let (first, last) = (
            TileKey::new(0, 0, 0),
            TileKey::new(u8::MAX, u16::MAX, u16::MAX),
        );
        for (index, terrain) in frame.terrains.terrains.iter().enumerate() {
            if !terrain.drawn {
                continue;
            }
            let Some(&info) = frame.infos.get(index) else {
                continue;
            };
            let ground = frame
                .terrain
                .highest(terrain.id)
                .map_or(-f32::MAX, |top| terrain.origin.y + top);
            let (levels, apron, highest) = self
                .grids
                .range((terrain.id, first)..=(terrain.id, last))
                .fold(
                    (0_u32, 0.0_f32, ground),
                    |(levels, apron, highest), (_, grid)| {
                        (
                            levels | 1 << grid.level,
                            engine_core::math::max(apron, grid.apron),
                            engine_core::math::max(highest, grid.top),
                        )
                    },
                );
            records.push(TraceTerrain {
                info,
                cutout: terrain
                    .cutout
                    .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
                maps: [frame.terrain.sky.layer(index), levels, 0, 0],
                reach: [apron, frame.cover, highest, 0.0],
            });
        }
        if records == self.written {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&records);
        self.terrains.grow(device, bytes.len() as u64);
        if !bytes.is_empty() {
            queue.write_buffer(&self.terrains, 0, bytes);
        }
        self.written = records;
    }

    pub(crate) fn bindings<'a>(
        &'a self,
        sources: crate::terrain::TraceSources<'a>,
        (records, poses, placed): (&'a wgpu::Buffer, &'a wgpu::Buffer, u64),
        frames: Option<(&'a wgpu::TextureView, u64)>,
    ) -> [(&'static str, Bound<'a>); 12] {
        let [heights_placeholder, control_placeholder, frames_placeholder] =
            &self.placeholders.views;
        let [heights, control] = sources.arrays.unwrap_or_else(|| {
            [heights_placeholder, control_placeholder]
                .map(|view| Bound::fixed(crate::pipelines::view(view)))
        });
        let records = if records.size() >= size_of::<Record>() as u64 {
            Bound::counted(placed, records.as_entire_binding())
        } else {
            Bound::fixed(self.placeholders.record.as_entire_binding())
        };
        let frames = frames.map_or_else(
            || Bound::fixed(crate::pipelines::view(frames_placeholder)),
            |(view, generation)| Bound::counted(generation, crate::pipelines::view(view)),
        );
        [
            ("heights", heights),
            ("directory", sources.directory),
            ("control", control),
            ("palette", sources.palette),
            ("macro_maps", sources.macro_maps),
            ("terrain_sky", sources.sky),
            ("world_terrains", self.terrains.bound()),
            ("world_placed", self.placed.bound()),
            ("placed_records", records),
            (
                "placed_poses",
                Bound::counted(placed, poses.as_entire_binding()),
            ),
            ("placed_groups", self.groups.bound()),
            ("placed_frames", frames),
        ]
    }
}

fn same_boxes(a: &[ImpostorRecord], b: &[ImpostorRecord]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.sphere == b.sphere && a.extent == b.extent)
}

fn reaches(
    terrain: &engine_core::RenderTerrain,
    key: TileKey,
    slots: Range<u32>,
    placed: &Placements<'_>,
) -> Vec<Reach> {
    let raised = terrain
        .errors
        .iter()
        .take(usize::from(key.level))
        .fold(0.0, |sum, error| sum + error);
    let mut reaches = Vec::with_capacity((slots.end - slots.start) as usize);
    for slot in slots {
        let (Some(record), Some(pose)) = (
            placed.records.get(slot as usize),
            placed.poses.get(slot as usize),
        ) else {
            continue;
        };
        if record.group == DEAD {
            continue;
        }
        let Some(views) = placed
            .impostors
            .get(record.group as usize)
            .filter(|views| views.framed())
        else {
            continue;
        };
        let [x, y, z, _] = views.sphere;
        let [half_x, half_y, half_z] = views.extent;
        let middle = |row: [f32; 4]| row[0] * x + row[1] * y + row[2] * z + row[3];
        let spread =
            |row: [f32; 4]| row[0].abs() * half_x + row[1].abs() * half_y + row[2].abs() * half_z;
        let (across, along) = (spread(pose[0]), spread(pose[2]));
        let radius = (across * across + along * along).sqrt();
        if radius <= 0.0 || radius.is_nan() {
            continue;
        }
        reaches.push(Reach {
            slot,
            middle: [
                middle(pose[0]) - terrain.origin.x,
                middle(pose[2]) - terrain.origin.z,
            ],
            radius,
            top: middle(pose[1]) + spread(pose[1]) + raised,
        });
    }
    reaches
}

fn grid(
    reaches: &[Reach],
    terrain: &engine_core::RenderTerrain,
    key: TileKey,
) -> Option<(Vec<u32>, f32)> {
    if reaches.is_empty() {
        return None;
    }
    let size = terrain.pyramid.tile_size(u32::from(key.level));
    let apron = reaches.iter().fold(0.0, |widest, reach| {
        engine_core::math::max(widest, reach.radius)
    });
    let wide = size + 2.0 * apron;
    let spacing = wide / (reaches.len() as f32).sqrt();
    let narrowest = engine_core::math::max(2.0 * apron, spacing);
    let across = engine_core::math::max((wide / narrowest).floor(), 1.0) as u32;
    let cells = 1_u32 << across.ilog2();
    let width = wide / cells as f32;
    let corner = [
        f32::from(key.x) * size - apron,
        f32::from(key.z) * size - apron,
    ];
    let last = cells - 1;
    let cells_of = |reach: &Reach| {
        let cell = |value: f32, from: f32| {
            engine_core::math::clamp(((value - from) / width).floor(), 0.0, last as f32) as u32
        };
        [
            cell(reach.middle[0] - reach.radius, corner[0]),
            cell(reach.middle[0] + reach.radius, corner[0]),
            cell(reach.middle[1] - reach.radius, corner[1]),
            cell(reach.middle[1] + reach.radius, corner[1]),
        ]
    };
    let count = (cells * cells) as usize;
    let mut counts = vec![0_u32; count];
    for reach in reaches {
        let [x0, x1, z0, z1] = cells_of(reach);
        for z in z0..=z1 {
            for x in x0..=x1 {
                counts[(z * cells + x) as usize] += 1;
            }
        }
    }
    let mut ends = Vec::with_capacity(count);
    let mut total = 0_u32;
    for &listed in &counts {
        total += listed;
        ends.push(total);
    }
    let mut cursors: Vec<u32> = ends
        .iter()
        .zip(&counts)
        .map(|(end, listed)| end - listed)
        .collect();
    let mut entries = vec![0_u32; total as usize];
    let mut tops = vec![-f32::MAX; count];
    for reach in reaches {
        let [x0, x1, z0, z1] = cells_of(reach);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let at = (z * cells + x) as usize;
                entries[cursors[at] as usize] = reach.slot;
                cursors[at] += 1;
                tops[at] = engine_core::math::max(tops[at], reach.top);
            }
        }
    }
    let pyramid = top_pyramid(tops, cells);
    let head = 4 + pyramid.len() + ends.len();
    let mut words = Vec::with_capacity(head + entries.len());
    words.extend([cells, apron.to_bits(), head as u32, 0]);
    words.extend(pyramid.iter().map(|top| top.to_bits()));
    words.extend(ends);
    words.extend(entries);
    Some((words, apron))
}

fn top_pyramid(tops: Vec<f32>, cells: u32) -> Vec<f32> {
    let mut side = cells as usize;
    let mut pyramid = tops.clone();
    let mut level = tops;
    while side > 1 {
        let half = side / 2;
        let top = |x: usize, z: usize| level[z * side + x];
        let coarser: Vec<f32> = (0..half * half)
            .map(|at| {
                let (x, z) = (2 * (at % half), 2 * (at / half));
                engine_core::math::max(
                    engine_core::math::max(top(x, z), top(x + 1, z)),
                    engine_core::math::max(top(x, z + 1), top(x + 1, z + 1)),
                )
            })
            .collect();
        pyramid.extend_from_slice(&coarser);
        level = coarser;
        side = half;
    }
    pyramid
}

pub(crate) fn terrain_info(
    pass: &crate::terrain::TerrainPass,
    index: usize,
    terrain: &engine_core::RenderTerrain,
) -> TerrainInfo {
    let (directory, palette, _, colors) = pass.cover_terrain(index);
    TerrainInfo {
        origin: terrain.origin.extend(terrain.pyramid.spacing(0)).to_array(),
        levels: [directory, terrain.pyramid.levels(), palette, colors],
    }
}
