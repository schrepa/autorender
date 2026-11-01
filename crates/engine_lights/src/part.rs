use bevy_ecs::prelude::*;
use engine_core::{Cut, DeviceLimits, Error, Readback, View};
use engine_places::lighting::Lighting;
use engine_places::lights::{LIGHT_EXTERIOR, LIGHT_INTERIOR, LightRegion};
use engine_places::probes::ProbeCaptures;
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass,
    Prepare,
};
use glam::Mat4;

use crate::gpu::{
    Clusters, GROUP_LIGHTS, KEPT_REGIONS, LIT_VIEWS, LightInput, NO_SLOT, SHAPED_LIGHTS,
};
use crate::lights::{LightList, LocalLightSettings, RenderLight};
use crate::shaders;

const TILES_X: u32 = 16;
const TILES_Y: u32 = 9;
const SLICES: u32 = 24;
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const RANGES_BYTES: u64 = CLUSTERS * 8;
const SECTION_ALIGN: u64 = 256;
const INDEX_START: u64 = RANGES_BYTES.next_multiple_of(SECTION_ALIGN);
const TOTALS_BYTES: u64 = 16;
const NO_BUDGET: u32 = u32::MAX;
const LIGHT_BYTES: u64 = 64;
const CLUSTER_BYTES: u64 = size_of::<Clusters>() as u64;

impl Clusters {
    fn new(
        view_projection: Mat4,
        width: u32,
        height: u32,
        distance: f32,
        lights: u32,
        budget: Option<u32>,
        capacity: u32,
    ) -> Self {
        let depth = view_projection.row(3);
        let scale_x = view_projection.row(0).truncate().length();
        let scale_y = view_projection.row(1).truncate().length();
        let perspective = (depth.truncate().length_squared() - 1.0).abs() < 1.0e-3;
        let p22 = -view_projection.row(2).truncate().dot(depth.truncate());
        let p23 = view_projection.row(2).w + p22 * depth.w;
        let (mut near, mut far) = (p23 / (p22 + 1.0), p23 / p22);
        if !(perspective
            && p22 > 0.0
            && near.is_finite()
            && far.is_finite()
            && near > 0.0
            && far > near)
        {
            (near, far) = (0.1, 1000.0);
        }
        let far = engine_math::max(engine_math::min(far, distance), near * 2.0);
        let start = engine_math::max(near, engine_math::min(1.0, far * 0.01));
        let slice_scale = SLICES as f32 / engine_math::max((far / start).log2(), 1.0e-3);
        let tile = |size: u32, tiles: u32| size.max(1).div_ceil(tiles) as f32;
        Self {
            view_projection: view_projection.to_cols_array_2d(),
            depth: depth.to_array(),
            projection: [scale_x, scale_y, start, far],
            grid: [
                tile(width, TILES_X),
                tile(height, TILES_Y),
                slice_scale,
                -start.log2() * slice_scale,
            ],
            counts: [TILES_X, TILES_Y, SLICES, budget.unwrap_or(NO_BUDGET)],
            lights: [lights, 0, capacity, 0],
            offsets: [0; 4],
        }
    }
}

fn shading_flags(shaped: bool, regions: bool) -> u32 {
    let mut flags = 0;
    if shaped {
        flags |= SHAPED_LIGHTS;
    }
    if regions {
        flags |= KEPT_REGIONS;
    }
    flags
}

impl From<&RenderLight> for LightInput {
    fn from(light: &RenderLight) -> Self {
        Self {
            position: light.position.to_array(),
            range: light.range,
            color: light.color.to_array(),
            slot: light.slot.unwrap_or(NO_SLOT),
            direction: light.direction.to_array(),
            cos_inner: light.cone[0],
            cos_outer: light.cone[1],
            radius: light.shape[0],
            length: light.shape[1],
            region: match light.region {
                LightRegion::Both => 0,
                LightRegion::Interior => LIGHT_INTERIOR,
                LightRegion::Exterior => LIGHT_EXTERIOR,
            },
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct LightTotals {
    read: Option<[u32; 4]>,
    epoch: u32,
}

impl LightTotals {
    pub(crate) fn forget(&mut self) {
        self.read = None;
        self.epoch = self.epoch.wrapping_add(1);
    }

    pub(crate) fn found(&self) -> Option<(u32, Cut)> {
        self.read.map(|[reached, lost, needed, capacity]| {
            let stored = needed.min(capacity);
            (reached, Cut::of(stored.saturating_add(lost), stored))
        })
    }

    fn needed(&self) -> u64 {
        self.read.map_or(0, |[_, _, needed, _]| u64::from(needed))
    }
}

impl Readback for LightTotals {
    fn arrived(&mut self, _: u64, epoch: u32, bytes: &[u8]) {
        if epoch != self.epoch {
            return;
        }
        let mut words = [0; 4];
        for (word, read) in words.iter_mut().zip(bytes.chunks_exact(4)) {
            *word = u32::from_le_bytes([read[0], read[1], read[2], read[3]]);
        }
        self.read = Some(words);
    }
}

fn index_room(room: u64, capacity: usize, views: u32) -> u64 {
    (room.saturating_sub(capacity as u64 * LIGHT_BYTES) / u64::from(views.max(1)))
        .saturating_sub(INDEX_START)
        / SECTION_ALIGN
        * SECTION_ALIGN
}

fn first_index_bytes(room: u64) -> u64 {
    (CLUSTERS * 4)
        .next_multiple_of(SECTION_ALIGN)
        .min(index_room(room, 1, 1))
}

pub(crate) struct LightsGpu {
    binning: Pass<Self>,
    counting: ComputePipeline<Self>,
    scanning: ComputePipeline<Self>,
    filling: ComputePipeline<Self>,
    clusters: Buffer<Self>,
    inputs: Buffer<Self>,
    data: Buffer<Self>,
    counts: Buffer<Self>,
    totals: Buffer<Self>,
    staged: Vec<LightInput>,
    headers: Vec<Clusters>,
    written: Vec<Clusters>,
    room: u64,
    capacity: usize,
    index_bytes: u64,
    arena: u32,
    regions: bool,
    stale: bool,
    count: u32,
    views: u32,
    epoch: u32,
}

impl LightsGpu {
    fn reset(&mut self, epoch: u32, room: u64) {
        self.epoch = epoch;
        self.room = room;
        self.capacity = 1;
        self.arena = 1;
        self.index_bytes = first_index_bytes(room);
        self.regions = false;
        self.stale = true;
        self.written.clear();
    }

    fn view_bytes(&self) -> u64 {
        INDEX_START + self.index_bytes
    }

    fn lights_offset(&self) -> u64 {
        u64::from(self.arena) * self.view_bytes()
    }

    fn data_bytes(&self) -> u64 {
        self.lights_offset() + self.capacity as u64 * LIGHT_BYTES
    }

    fn offsets(&self, view: u32) -> [u32; 4] {
        let base = u64::from(view) * self.view_bytes();
        [
            (self.lights_offset() / 16) as u32,
            (base / 16) as u32,
            ((base + INDEX_START) / 16) as u32,
            0,
        ]
    }

    fn index_capacity(&self) -> u32 {
        (self.index_bytes / 4).min(u64::from(u32::MAX)) as u32
    }

    fn grow_indices(&mut self, bytes: u64) {
        let size = bytes
            .next_power_of_two()
            .min(index_room(self.room, self.capacity, self.arena));
        if bytes <= self.index_bytes || size <= self.index_bytes {
            return;
        }
        self.index_bytes = size;
    }

    fn place_headers(
        &mut self,
        settings: &LocalLightSettings,
        view: &View,
        captures: &ProbeCaptures,
        shaped: bool,
    ) {
        let size = view.render_size();
        let flags = shading_flags(shaped, self.regions);
        let capacity = self.index_capacity();
        let mut main = Clusters::new(
            view.view_projection(),
            size.x,
            size.y,
            settings.distance,
            self.count,
            settings.lights_per_cluster,
            capacity,
        );
        main.lights[1] = flags;
        main.offsets = self.offsets(0);
        self.headers[0] = main;
        for (index, face) in captures.faces.iter().take(self.views as usize - 1).enumerate() {
            let lit = 1 + index as u32;
            let mut header = Clusters::new(
                face.view,
                captures.size,
                captures.size,
                settings.distance,
                self.count,
                settings.lights_per_cluster,
                capacity,
            );
            header.lights[1] = flags;
            header.lights[3] = 1;
            header.offsets = self.offsets(lit);
            self.headers[lit as usize] = header;
        }
    }
}

impl GpuPart for LightsGpu {
    type Reads = (
        Res<'static, LocalLightSettings>,
        Res<'static, LightList>,
        Res<'static, LightTotals>,
        Res<'static, Lighting>,
        Res<'static, View>,
        Res<'static, ProbeCaptures>,
        Res<'static, DeviceLimits>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::lights::{binning, clusters};
        let cluster_views = cx.buffer("light_clusters", Lifetime::Kept);
        let data = cx.buffer("light_data", Lifetime::Kept);
        let inputs = cx.buffer("light_inputs", Lifetime::Kept);
        let counts = cx.buffer("light_cluster_counts", Lifetime::Frame);
        let totals = cx.buffer("light_totals", Lifetime::Frame);
        cx.frame(clusters::CLUSTERS, Bind::Buffer(cluster_views))?;
        cx.frame(clusters::LIGHT_DATA, Bind::Buffer(data))?;
        let binning_pass = cx
            .pass("lights")
            .label("engine/light-binning")
            .span("light_bins")
            .id();
        let visits = || {
            [
                (binning::INPUTS, Bind::Buffer(inputs)),
                (binning::ARENA, Bind::Buffer(data)),
                (binning::COUNTS, Bind::Buffer(counts)),
                (binning::TOTALS, Bind::Buffer(totals)),
            ]
        };
        let counting = cx.compute_pipeline(binning_pass, binning::COUNT_LIGHTS, &visits())?;
        let filling = cx.compute_pipeline(binning_pass, binning::BIN_LIGHTS, &visits())?;
        let scanning = cx.compute_pipeline(
            binning_pass,
            binning::SCAN_CLUSTERS,
            &[
                (binning::ARENA, Bind::Buffer(data)),
                (binning::COUNTS, Bind::Buffer(counts)),
                (binning::TOTALS, Bind::Buffer(totals)),
            ],
        )?;
        Ok(Self {
            binning: binning_pass,
            counting,
            scanning,
            filling,
            clusters: cluster_views,
            inputs,
            data,
            counts,
            totals,
            staged: Vec::new(),
            headers: vec![Clusters::default(); LIT_VIEWS as usize],
            written: Vec::new(),
            room: 0,
            capacity: 1,
            index_bytes: first_index_bytes(0),
            arena: 1,
            regions: false,
            stale: false,
            count: 0,
            views: 1,
            epoch: 0,
        })
    }

    fn prepare(
        &mut self,
        (settings, list, totals, lighting, view, captures, limits): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if view.debug() {
            self.stale |= list.is_changed();
            return;
        }
        let room = limits.storage_room / 16 * 16;
        if totals.epoch != self.epoch || list.lights.is_empty() || room != self.room {
            self.reset(totals.epoch, room);
        }
        let count = list.lights.len();
        let mut grown = false;
        if count > self.capacity {
            self.capacity = count.next_power_of_two();
            grown = true;
        }
        self.views = 1 + captures.faces.len().min(LIT_VIEWS as usize - 1) as u32;
        if self.views > self.arena {
            self.arena = self.views;
            self.index_bytes = self
                .index_bytes
                .min(index_room(self.room, self.capacity, self.arena));
        }
        let upload = grown || self.stale || list.is_changed();
        if upload {
            self.staged.clear();
            self.staged
                .extend(list.lights.iter().map(LightInput::from));
            self.regions = self.staged.iter().any(|light| light.region != 0);
            self.stale = false;
        }
        let needed = totals.needed().max(count as u64 * 32);
        self.grow_indices(needed * 4);
        cx.allocate_buffer(self.inputs, self.capacity as u64 * LIGHT_BYTES);
        cx.allocate_buffer(self.data, self.data_bytes());
        cx.allocate_buffer(self.counts, u64::from(self.arena) * CLUSTERS * 4);
        cx.allocate_buffer(self.totals, u64::from(self.arena) * TOTALS_BYTES);
        cx.allocate_buffer(self.clusters, u64::from(LIT_VIEWS) * CLUSTER_BYTES);
        if upload && !self.staged.is_empty() {
            cx.write(self.inputs, 0, &self.staged);
        }
        self.count = count as u32;
        self.place_headers(&settings, &view, &captures, lighting.highlights);
        if self.written != self.headers {
            cx.write(self.clusters, 0, &self.headers);
            self.written.clone_from(&self.headers);
        }
        if count == 0 {
            return;
        }
        cx.run(self.binning);
        cx.read_back::<LightTotals>(self.epoch, self.totals, 0..TOTALS_BYTES, AfterRead::Keep);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.binning {
            return 0;
        }
        cx.clear(self.counts);
        cx.clear(self.totals);
        let workgroups = self.count.div_ceil(GROUP_LIGHTS);
        let (columns, rows) = (workgroups.min(65_535), workgroups.div_ceil(65_535));
        cx.dispatch(&self.counting, Count::Groups([columns, rows, self.views]));
        cx.dispatch(&self.scanning, Count::Groups([1, 1, self.views]));
        cx.dispatch(&self.filling, Count::Groups([columns, rows, self.views]));
        0
    }
}
