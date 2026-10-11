//! Independent inputs around the unchanged serial bake kernels. The host
//! owns the graph; no worker waits for another worker.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use super::{BakeKind, Baked, RiverJob, TerrainBake};
use crate::codec::{self, Codec, CodecError, Reader, Writer, record};
use crate::path::RIVER_DEPTH;
use crate::rivers::{BANK_OFFSET, RiverLine, across};
use crate::{Generator, Layers, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

pub(crate) const BAKE_SHARE: u8 = 2;
pub const BAKE_CHILD_JOB: &str = "terrain_bake_child";
const VERSION: u32 = 1;
const POINTS: usize = 256;
const GRID: u8 = 0;
const PATH: u8 = 1;
const RIVER: u8 = 2;
const KERNEL: u8 = 3;
const FINAL: u8 = 4;

/// Ordered messages for the existing pool. Runs have graph-local IDs;
/// hosts allocate separate transport IDs for every ordinary/child job.
pub enum BakeAction {
    Share { key: String, bytes: Vec<u8> },
    Unshare { key: String },
    Run { id: u64, input: Vec<u8> },
}

/// Current retained buffers and cumulative traffic of one graph. Shared
/// bytes describe one copy; hosts count copies sent to their workers.
#[derive(Clone, Copy, Default)]
pub struct BakeDiagnostics {
    pub in_flight: usize,
    pub completed: u64,
    pub sent_bytes: u64,
    pub received_bytes: u64,
    pub share_bytes: usize,
    pub buffer_bytes: usize,
    pub peak_bytes: usize,
}

#[derive(Clone)]
pub(crate) struct Sampling {
    below: Arc<Layers>,
    pyramid: Pyramid,
    kind: BakeKind,
    level: u32,
    width: usize,
    height: usize,
    profile: Vec<glam::Vec2>,
    step: f32,
    rivers: Option<RiverJob>,
}

record!(RiverLine {
    points,
    half,
    depth,
    head
});
record!(RiverJob {
    hydrology,
    lines,
    ground
});
record!(Sampling {
    below,
    pyramid,
    kind,
    level,
    width,
    height,
    profile,
    step,
    rivers
});

impl Codec for BakeKind {
    fn write(&self, out: &mut Writer) {
        match self {
            Self::Hydrology(value) => {
                out.put(&0_u8).put(value);
            }
            Self::Erosion(value) => {
                out.put(&1_u8).put(value);
            }
            Self::Path(value) => {
                out.put(&2_u8).put(value);
            }
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get::<u8>()? {
            0 => Ok(Self::Hydrology(input.get()?)),
            1 => Ok(Self::Erosion(input.get()?)),
            2 => Ok(Self::Path(input.get()?)),
            _ => Err(CodecError("unknown bake kind")),
        }
    }
}

impl Codec for Baked {
    fn write(&self, out: &mut Writer) {
        match self {
            Self::Hydrology(value) => {
                out.put(&0_u8).put(value);
            }
            Self::Erosion(value) => {
                out.put(&1_u8).put(value);
            }
            Self::Path(value) => {
                out.put(&2_u8).put(value);
            }
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match input.get::<u8>()? {
            0 => Ok(Self::Hydrology(input.get()?)),
            1 => Ok(Self::Erosion(input.get()?)),
            2 => Ok(Self::Path(input.get()?)),
            _ => Err(CodecError("unknown baked output")),
        }
    }
}

fn invalid(error: CodecError) -> String {
    error.to_string()
}

fn header(input: &mut Reader<'_>) -> Result<u8, String> {
    if input.get::<u32>().map_err(invalid)? != VERSION {
        return Err("incompatible private bake codec".into());
    }
    input.get().map_err(invalid)
}

fn writer(tag: u8) -> Writer {
    let mut out = Writer::new();
    out.put(&VERSION).put(&tag);
    out
}

fn finite(values: &[f32]) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("non-finite bake samples".into())
    }
}

fn image(value: &crate::layers::HeightImage, width: usize, height: usize) -> bool {
    if value.width as usize != width
        || value.height as usize != height
        || width.checked_mul(height) != Some(value.samples.len())
    {
        return false;
    }
    let (mut width, mut height) = (width, height);
    for level in &value.levels {
        if width <= 2 && height <= 2 {
            return false;
        }
        width = (width - 1).div_ceil(2) + 1;
        height = (height - 1).div_ceil(2) + 1;
        if level.width as usize != width
            || level.height as usize != height
            || width.checked_mul(height) != Some(level.samples.len())
            || level
                .samples
                .iter()
                .any(|sample| !sample.is_finite() || !(0.0..=65_535.0).contains(sample))
        {
            return false;
        }
    }
    width <= 2 && height <= 2
}

impl Sampling {
    fn hydrology(&self, value: &crate::Hydrology) -> bool {
        let count = self.width.checked_mul(self.height);
        let spacing = self.pyramid.spacing(self.level);
        let BakeKind::Hydrology(layer) = &self.kind else {
            return false;
        };
        value.spacing == spacing
            && value.max_depth == layer.max_depth
            && [&value.carve, &value.flow, &value.wetness, &value.lakes]
                .iter()
                .all(|value| image(value, self.width, self.height))
            && value.water.width == self.width
            && value.water.height == self.height
            && value.water.spacing == spacing
            && count == Some(value.water.surface.len())
            && count == Some(value.water.depth.len())
            && count == Some(value.water.flow.len())
            && value.water.surface.iter().all(|value| value.is_finite())
            && value
                .water
                .depth
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && value
                .water
                .flow
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            && count == Some(value.stats.samples)
            && value.stats.channel_samples <= value.stats.samples
            && value.stats.lake_samples <= value.stats.samples
            && value.stats.max_area.is_finite()
            && value.stats.max_area >= 0.0
            && value.stats.falls == value.falls.len()
            && value.falls.iter().all(|fall| {
                fall.top
                    .iter()
                    .chain(&fall.bottom)
                    .chain(fall.path.iter().flatten())
                    .all(|value| value.is_finite())
                    && fall.width.is_finite()
                    && fall.width >= 0.0
                    && fall.area.is_finite()
                    && fall.area >= 0.0
            })
    }

    fn baked(&self, value: &Baked) -> bool {
        match (&self.kind, value) {
            (BakeKind::Hydrology(_), Baked::Hydrology(value)) => self.hydrology(value),
            (BakeKind::Erosion(layer), Baked::Erosion(value)) => {
                value.spacing == self.pyramid.spacing(self.level)
                    && image(&value.change, self.width, self.height)
                    && value.range.iter().all(|value| value.is_finite())
                    && value.range[0] <= value.range[1]
                    && self.width.checked_mul(self.height) == Some(value.stats.samples)
                    && value.stats.rounds == layer.rounds
                    && value.stats.deepest.is_finite()
                    && value.stats.highest.is_finite()
            }
            (BakeKind::Path(layer), Baked::Path(value)) => {
                value.valid_bake(layer, self.step, self.profile.len())
            }
            _ => false,
        }
    }
}

fn zeros(count: usize) -> Result<Vec<f32>, String> {
    count
        .checked_mul(size_of::<f32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("bake grid size overflow")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| error.to_string())?;
    values.resize(count, 0.0);
    Ok(values)
}

impl Sampling {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut input = Reader::new(bytes);
        if input.get::<u32>().map_err(invalid)? != VERSION {
            return Err("incompatible private bake share".into());
        }
        let value: Self = input.get().map_err(invalid)?;
        input.finish().map_err(invalid)?;
        match &value.kind {
            BakeKind::Hydrology(layer) => layer.validate(),
            BakeKind::Erosion(layer) => layer.validate(),
            BakeKind::Path(layer) => layer.validate(),
        }
        .map_err(|error| error.to_string())?;
        if value.level
            != value
                .kind
                .level()
                .unwrap_or(0)
                .min(value.pyramid.levels() - 1)
            || !value.step.is_finite()
            || value.profile.iter().any(|point| !point.is_finite())
        {
            return Err("invalid bake sampling metadata".into());
        }
        if let Some(job) = &value.rivers {
            if !matches!(value.kind, BakeKind::Hydrology(_))
                || job.hydrology.is_none()
                || !job
                    .hydrology
                    .as_ref()
                    .is_some_and(|hydrology| value.hydrology(hydrology))
                || job.ground.len() != job.lines.len()
                || job.ground.iter().any(|line| !line.is_empty())
            {
                return Err("invalid river sampling metadata".into());
            }
            for line in &job.lines {
                if line.points.is_empty()
                    || line.points.len() != line.half.len()
                    || line.points.len() != line.depth.len()
                    || line.points.iter().any(|point| !point.is_finite())
                {
                    return Err("invalid river sampling line".into());
                }
                finite(&line.half)?;
                finite(&line.depth)?;
                if line
                    .half
                    .iter()
                    .chain(&line.depth)
                    .any(|&value| value < 0.0)
                {
                    return Err("negative river sampling width or depth".into());
                }
            }
        }
        if let BakeKind::Path(layer) = &value.kind {
            let (profile, step) = layer.profile_points();
            if value.width != 0
                || value.height != 0
                || value.profile != profile
                || value.step != step
            {
                return Err("invalid path sampling metadata".into());
            }
        } else {
            let [x, z] = value.pyramid.tiles(value.level);
            if value.width != x as usize * TILE_QUADS as usize + 1
                || value.height != z as usize * TILE_QUADS as usize + 1
                || !value.profile.is_empty()
                || value.step != 0.0
            {
                return Err("invalid bake grid dimensions".into());
            }
        }
        Ok(value)
    }

    fn blank_bake(&self) -> TerrainBake {
        TerrainBake {
            index: 0,
            kind: self.kind.clone(),
            level: self.level,
            below: self.below.clone(),
            pending: Vec::new(),
            job: None,
            width: self.width,
            heights: Vec::new(),
            original: None,
            rounds: 0,
            profile: self.profile.clone(),
            step: self.step,
            ground: Vec::new(),
            water: Vec::new(),
            rivers: self.rivers.clone(),
        }
    }

    fn river_points(&self) -> Result<usize, String> {
        let job = self.rivers.as_ref().ok_or("no river sampling stage")?;
        job.lines.iter().try_fold(0_usize, |sum, line| {
            sum.checked_add(line.points.len())
                .ok_or_else(|| "river size overflow".into())
        })
    }
}

#[derive(Clone, Copy)]
enum Expected {
    Grid(TileKey),
    Path(usize, usize),
    River(usize, usize),
    Kernel,
    Final,
}

/// One logical terrain_bake split into deterministic, uncached children.
/// Its input/output remain exactly the original monolithic job's bytes.
pub struct BakeCoordinator {
    layers: Option<Layers>,
    pyramid: Pyramid,
    epoch: u64,
    index: usize,
    sampling: Option<Sampling>,
    key: Option<String>,
    stage: u8,
    next: usize,
    count: usize,
    tile_keys: Vec<TileKey>,
    heights: Vec<f32>,
    ground: Vec<f32>,
    water: Vec<Option<f32>>,
    rivers: Vec<Vec<[f32; 3]>>,
    waiting: HashMap<u64, Expected>,
    actions: VecDeque<BakeAction>,
    serial: u64,
    output: Option<Vec<u8>>,
    cancelled: bool,
    stats: BakeDiagnostics,
}

impl BakeCoordinator {
    pub fn start(input: &[u8], epoch: u64) -> Result<Self, String> {
        let mut reader = Reader::new(input);
        let layers: Layers = reader.get().map_err(invalid)?;
        let pyramid: Pyramid = reader.get().map_err(invalid)?;
        reader.finish().map_err(invalid)?;
        let mut graph = Self {
            layers: Some(layers),
            pyramid,
            epoch,
            index: 0,
            sampling: None,
            key: None,
            stage: GRID,
            next: 0,
            count: 0,
            tile_keys: Vec::new(),
            heights: Vec::new(),
            ground: Vec::new(),
            water: Vec::new(),
            rivers: Vec::new(),
            waiting: HashMap::new(),
            actions: VecDeque::new(),
            serial: 0,
            output: None,
            cancelled: false,
            stats: BakeDiagnostics::default(),
        };
        graph.layer()?;
        Ok(graph)
    }

    fn layer(&mut self) -> Result<(), String> {
        self.release_share();
        let layers = self.layers.as_ref().ok_or("cancelled bake")?;
        let Some((index, kind)) = layers.unbaked() else {
            self.output = Some(codec::encode(layers));
            self.layers = None;
            self.sampling = None;
            self.tile_keys = Vec::new();
            self.next = 0;
            self.count = 0;
            return Ok(());
        };
        self.index = index;
        let level = kind.level().unwrap_or(0).min(self.pyramid.levels() - 1);
        let (profile, step) = match &kind {
            BakeKind::Path(layer) => layer.profile_points(),
            _ => (Vec::new(), 0.0),
        };
        let [x, z] = self.pyramid.tiles(level);
        let (width, height) = if matches!(kind, BakeKind::Path(_)) {
            (0, 0)
        } else {
            (
                x as usize * TILE_QUADS as usize + 1,
                z as usize * TILE_QUADS as usize + 1,
            )
        };
        self.heights = zeros(width.checked_mul(height).ok_or("bake grid size overflow")?)?;
        self.ground = zeros(profile.len())?;
        self.water = vec![None; profile.len()];
        self.rivers = Vec::new();
        self.tile_keys = if width == 0 {
            Vec::new()
        } else {
            self.pyramid.level_tiles(level).collect()
        };
        self.stage = if width == 0 { PATH } else { GRID };
        self.count = if width == 0 {
            profile.len()
        } else {
            self.tile_keys.len()
        };
        self.next = 0;
        self.sampling = Some(Sampling {
            below: Arc::new(layers.below(index)),
            pyramid: self.pyramid,
            kind,
            level,
            width,
            height,
            profile,
            step,
            rivers: None,
        });
        self.share();
        Ok(())
    }

    fn share(&mut self) {
        self.release_share();
        let key = format!("bake-private/{}/{}/{}", self.epoch, self.index, self.stage);
        let mut bytes = Writer::new();
        bytes
            .put(&BAKE_SHARE)
            .put(&VERSION)
            .put(self.sampling.as_ref().expect("active stage"));
        let bytes = bytes.into_bytes();
        self.stats.share_bytes = bytes.len();
        self.key = Some(key.clone());
        self.actions.push_back(BakeAction::Share { key, bytes });
    }

    fn release_share(&mut self) {
        if let Some(key) = self.key.take() {
            self.actions.push_back(BakeAction::Unshare { key });
        }
        self.stats.share_bytes = 0;
    }

    /// Up to `limit` new runs plus ordered share/unshare messages. The
    /// caller bounds runs across all roots and ordinary jobs together.
    pub fn ready(&mut self, limit: usize) -> Result<Vec<BakeAction>, String> {
        let mut actions: Vec<_> = self.actions.drain(..).collect();
        if self.cancelled || self.layers.is_none() {
            return Ok(actions);
        }
        if self.waiting.is_empty()
            && self.next == self.count
            && matches!(self.stage, GRID | PATH | RIVER)
        {
            self.stage = if self.stage == GRID { KERNEL } else { FINAL };
            self.next = 0;
            self.count = 1;
        }
        for _ in 0..limit {
            if self.next >= self.count {
                break;
            }
            let key = self.key.as_ref().ok_or("missing private share")?;
            let mut input = writer(self.stage);
            input.put(key);
            let expected = match self.stage {
                GRID => {
                    let tile = self.tile_keys[self.next];
                    input.put(&tile);
                    self.next += 1;
                    Expected::Grid(tile)
                }
                PATH | RIVER => {
                    let first = self.next;
                    let end = self.count.min(first.saturating_add(POINTS));
                    input.put(&first).put(&end);
                    self.next = end;
                    if self.stage == PATH {
                        Expected::Path(first, end)
                    } else {
                        Expected::River(first, end)
                    }
                }
                KERNEL => {
                    input.put(&self.heights);
                    self.heights = Vec::new();
                    self.next = 1;
                    Expected::Kernel
                }
                FINAL => {
                    input.put(&self.ground).put(&self.water).put(&self.rivers);
                    self.ground = Vec::new();
                    self.water = Vec::new();
                    self.rivers = Vec::new();
                    self.next = 1;
                    Expected::Final
                }
                _ => return Err("unknown bake stage".into()),
            };
            self.serial = self.serial.checked_add(1).ok_or("bake child ID overflow")?;
            let input = input.into_bytes();
            self.stats.sent_bytes = self.stats.sent_bytes.saturating_add(input.len() as u64);
            self.waiting.insert(self.serial, expected);
            actions.push(BakeAction::Run {
                id: self.serial,
                input,
            });
        }
        self.update_stats();
        Ok(actions)
    }

    /// An invalid result poisons the root: no partial output can escape.
    pub fn complete(&mut self, id: u64, output: &[u8]) -> Result<(), String> {
        let result = self.join(id, output);
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn join(&mut self, id: u64, output: &[u8]) -> Result<(), String> {
        if self.cancelled {
            return Err("cancelled bake".into());
        }
        let expected = *self
            .waiting
            .get(&id)
            .ok_or("unexpected or duplicate bake child")?;
        let mut reader = Reader::new(output);
        let tag = header(&mut reader)?;
        let sampling = self.sampling.as_ref().ok_or("missing bake stage")?;
        match expected {
            Expected::Grid(key) => {
                let found: TileKey = reader.get().map_err(invalid)?;
                let samples: Vec<f32> = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let side = TILE_SAMPLES as usize;
                if tag != GRID || found != key || samples.len() != side * side {
                    return Err("wrong bake tile key or sample count".into());
                }
                finite(&samples)?;
                // Serial pending.pop() writes reverse row-major, so the
                // lowest row-major tile owns a shared edge/corner.
                let x = usize::from(key.x) * TILE_QUADS as usize;
                let z = usize::from(key.z) * TILE_QUADS as usize;
                let first_column = usize::from(key.x > 0);
                for row in usize::from(key.z > 0)..side {
                    let from = row * side + first_column;
                    let to = (z + row) * sampling.width + x + first_column;
                    self.heights[to..to + side - first_column]
                        .copy_from_slice(&samples[from..(row + 1) * side]);
                }
            }
            Expected::Path(first, end) => {
                let start: usize = reader.get().map_err(invalid)?;
                let ground: Vec<f32> = reader.get().map_err(invalid)?;
                let water: Vec<Option<f32>> = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if tag != PATH
                    || start != first
                    || ground.len() != end - first
                    || water.len() != ground.len()
                {
                    return Err("wrong path sample range".into());
                }
                finite(&ground)?;
                if water.iter().flatten().any(|value| !value.is_finite()) {
                    return Err("invalid water sample".into());
                }
                self.ground[first..end].copy_from_slice(&ground);
                self.water[first..end].copy_from_slice(&water);
            }
            Expected::River(first, end) => {
                let start: usize = reader.get().map_err(invalid)?;
                let ground: Vec<[f32; 3]> = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if tag != RIVER || start != first || ground.len() != end - first {
                    return Err("wrong river sample range".into());
                }
                for sample in &ground {
                    finite(sample)?;
                }
                let mut offset = 0;
                for line in &mut self.rivers {
                    let begin = first.saturating_sub(offset).min(line.len());
                    let stop = end.saturating_sub(offset).min(line.len());
                    if begin < stop {
                        line[begin..stop].copy_from_slice(
                            &ground[offset + begin - first..offset + stop - first],
                        );
                    }
                    offset += line.len();
                }
            }
            Expected::Kernel if tag == RIVER => {
                let river: RiverJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if !matches!(sampling.kind, BakeKind::Hydrology(_))
                    || river.hydrology.is_none()
                    || river.ground.len() != river.lines.len()
                    || river.ground.iter().any(|line| !line.is_empty())
                {
                    return Err("wrong hydrology kernel result".into());
                }
                let mut next = sampling.clone();
                next.rivers = Some(river);
                // The same validation workers apply also rejects malformed
                // kernel-produced line metadata before graph allocation.
                let mut encoded = Writer::new();
                encoded.put(&VERSION).put(&next);
                let next = Sampling::decode(&encoded.into_bytes())?;
                self.count = next.river_points()?;
                self.rivers = next
                    .rivers
                    .as_ref()
                    .expect("validated rivers")
                    .lines
                    .iter()
                    .map(|line| vec![[0.0; 3]; line.points.len()])
                    .collect();
                self.sampling = Some(next);
                self.stage = RIVER;
                self.next = 0;
                self.share();
            }
            Expected::Kernel | Expected::Final => {
                let baked: Baked = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if tag != FINAL || !sampling.baked(&baked) {
                    return Err("wrong bake kernel result".into());
                }
                let layers = self
                    .layers
                    .as_ref()
                    .ok_or("cancelled bake")?
                    .with_baked(self.index, baked);
                // Re-read the complete layer stack with its ordinary codec
                // validation before exposing it or preparing a later layer.
                self.layers = Some(codec::decode(&codec::encode(&layers)).map_err(invalid)?);
                self.layer()?;
            }
        }
        self.waiting.remove(&id);
        self.stats.completed = self.stats.completed.saturating_add(1);
        self.stats.received_bytes = self
            .stats
            .received_bytes
            .saturating_add(output.len() as u64);
        self.update_stats();
        Ok(())
    }

    pub fn take_result(&mut self) -> Option<Vec<u8>> {
        self.output.take()
    }

    pub fn cancel(&mut self) {
        if self.cancelled {
            return;
        }
        self.cancelled = true;
        // A completed child may already have queued the old stage's
        // release and a new share. Cancellation discards unsent work but
        // must still release every stage that a host could be holding.
        self.actions
            .retain(|action| matches!(action, BakeAction::Unshare { .. }));
        self.release_share();
        self.layers = None;
        self.sampling = None;
        self.output = None;
        self.tile_keys = Vec::new();
        self.heights = Vec::new();
        self.ground = Vec::new();
        self.water = Vec::new();
        self.rivers = Vec::new();
        self.waiting = HashMap::new();
        self.update_stats();
    }

    fn update_stats(&mut self) {
        self.stats.in_flight = self.waiting.len();
        self.stats.buffer_bytes = self.heights.capacity() * size_of::<f32>()
            + self.ground.capacity() * size_of::<f32>()
            + self.water.capacity() * size_of::<Option<f32>>()
            + self
                .rivers
                .iter()
                .map(|line| line.capacity() * size_of::<[f32; 3]>())
                .sum::<usize>()
            + self.tile_keys.capacity() * size_of::<TileKey>();
        self.stats.peak_bytes = self.stats.peak_bytes.max(
            self.stats
                .buffer_bytes
                .saturating_add(self.stats.share_bytes),
        );
    }

    pub fn diagnostics(&self) -> BakeDiagnostics {
        self.stats
    }
}

/// Child work held by an ordinary TerrainWorker.
pub(crate) struct ParallelJob {
    sampling: Arc<Sampling>,
    kind: u8,
    first: usize,
    end: usize,
    tile: Option<(TileKey, Generator)>,
    bake: Option<TerrainBake>,
}

impl ParallelJob {
    pub(crate) fn start(
        input: &[u8],
        shares: &HashMap<String, Arc<Sampling>>,
    ) -> Result<Self, String> {
        let mut reader = Reader::new(input);
        let kind = header(&mut reader)?;
        let key: String = reader.get().map_err(invalid)?;
        let sampling = shares
            .get(&key)
            .ok_or("missing private bake share")?
            .clone();
        let mut job = Self {
            sampling,
            kind,
            first: 0,
            end: 0,
            tile: None,
            bake: None,
        };
        match kind {
            GRID => {
                let key: TileKey = reader.get().map_err(invalid)?;
                if matches!(job.sampling.kind, BakeKind::Path(_))
                    || job.sampling.rivers.is_some()
                    || !job.sampling.pyramid.contains(key)
                    || u32::from(key.level) != job.sampling.level
                {
                    return Err("bake tile outside its level".into());
                }
                job.tile = Some((
                    key,
                    Generator::new(job.sampling.below.clone(), &job.sampling.pyramid, key),
                ));
            }
            PATH | RIVER => {
                if (kind == PATH) != matches!(job.sampling.kind, BakeKind::Path(_)) {
                    return Err("wrong bake sampling stage".into());
                }
                job.first = reader.get().map_err(invalid)?;
                job.end = reader.get().map_err(invalid)?;
                let count = if kind == PATH {
                    job.sampling.profile.len()
                } else {
                    job.sampling.river_points()?
                };
                if job.first >= job.end || job.end > count || job.end - job.first > POINTS {
                    return Err("invalid bake sample range".into());
                }
            }
            KERNEL => {
                let heights: Vec<f32> = reader.get().map_err(invalid)?;
                if matches!(job.sampling.kind, BakeKind::Path(_))
                    || job.sampling.rivers.is_some()
                    || Some(heights.len()) != job.sampling.width.checked_mul(job.sampling.height)
                {
                    return Err("wrong bake kernel grid".into());
                }
                finite(&heights)?;
                let mut bake = job.sampling.blank_bake();
                bake.heights = heights;
                job.bake = Some(bake);
            }
            FINAL => {
                let ground: Vec<f32> = reader.get().map_err(invalid)?;
                let water: Vec<Option<f32>> = reader.get().map_err(invalid)?;
                let rivers: Vec<Vec<[f32; 3]>> = reader.get().map_err(invalid)?;
                if ground.len() != job.sampling.profile.len() || water.len() != ground.len() {
                    return Err("incomplete path profile".into());
                }
                finite(&ground)?;
                if water.iter().flatten().any(|height| !height.is_finite()) {
                    return Err("invalid water profile".into());
                }
                let mut bake = job.sampling.blank_bake();
                bake.ground = ground;
                bake.water = water;
                if let Some(river) = &mut bake.rivers {
                    if rivers.len() != river.lines.len() {
                        return Err("incomplete river profiles".into());
                    }
                    for (line, read) in river.lines.iter().zip(&rivers) {
                        if read.len() != line.points.len() {
                            return Err("short river profile".into());
                        }
                        for value in read {
                            finite(value)?;
                        }
                    }
                    river.ground = rivers;
                } else if !rivers.is_empty() || !matches!(bake.kind, BakeKind::Path(_)) {
                    return Err("unexpected river profile".into());
                }
                job.bake = Some(bake);
            }
            _ => return Err("unknown bake child kind".into()),
        }
        reader.finish().map_err(invalid)?;
        Ok(job)
    }

    pub(crate) fn step(&mut self, budget: &mut u64) -> Result<Option<Vec<u8>>, String> {
        let mut out = writer(self.kind);
        match self.kind {
            GRID => {
                let (key, generator) = self.tile.as_mut().ok_or("missing bake generator")?;
                let Some(tile) = generator.advance(budget) else {
                    if generator.failed() {
                        return Err("invalid bake tile heights".into());
                    }
                    return Ok(None);
                };
                let side = TILE_SAMPLES as usize;
                let stride = TILE_STRIDE as usize;
                let mut values = Vec::with_capacity(side * side);
                for row in 0..side {
                    values.extend_from_slice(
                        &tile.heights()[(row + 1) * stride + 1..(row + 1) * stride + 1 + side],
                    );
                }
                out.put(key).put(&values);
            }
            PATH => {
                let spacing = self.sampling.pyramid.spacing(0);
                let mut ground = Vec::with_capacity(self.end - self.first);
                let mut water = Vec::with_capacity(self.end - self.first);
                for &point in &self.sampling.profile[self.first..self.end] {
                    let sample = [
                        (point.x / spacing).round() as i64,
                        (point.y / spacing).round() as i64,
                    ];
                    ground.push(self.sampling.below.sample_height(
                        &self.sampling.pyramid,
                        0,
                        sample,
                    ));
                    water.push(self.sampling.below.hydrology().and_then(|hydrology| {
                        hydrology
                            .water_at(point.x, point.y)
                            .filter(|&(_, depth)| depth > RIVER_DEPTH)
                            .map(|(surface, _)| surface)
                    }));
                }
                out.put(&self.first).put(&ground).put(&water);
            }
            RIVER => {
                let rivers = self.sampling.rivers.as_ref().ok_or("missing rivers")?;
                let hydrology = rivers.hydrology.as_ref().ok_or("missing hydrology")?;
                let spacing = self.sampling.pyramid.spacing(0);
                let level = hydrology.level_for(spacing);
                let mut values = Vec::with_capacity(self.end - self.first);
                let mut offset = 0;
                for line in &rivers.lines {
                    let first = self.first.saturating_sub(offset).min(line.points.len());
                    let end = self.end.saturating_sub(offset).min(line.points.len());
                    for index in first..end {
                        let point = line.points[index];
                        let side = across(&line.points, index) * (line.half[index] + BANK_OFFSET);
                        let read = |at: glam::Vec2| {
                            let sample = [
                                (at.x / spacing).round() as i64,
                                (at.y / spacing).round() as i64,
                            ];
                            self.sampling
                                .below
                                .sample_height(&self.sampling.pyramid, 0, sample)
                                - hydrology.carve_at(at.x, at.y, level)
                        };
                        values.push([read(point), read(point + side), read(point - side)]);
                    }
                    offset += line.points.len();
                }
                out.put(&self.first).put(&values);
            }
            KERNEL => {
                let bake = self.bake.as_mut().ok_or("missing bake kernel")?;
                let mut whole = u64::MAX;
                if let Some(baked) = bake.compute(
                    self.sampling.pyramid.spacing(self.sampling.level),
                    &mut whole,
                ) {
                    out = writer(FINAL);
                    out.put(&baked);
                } else {
                    out = writer(RIVER);
                    out.put(&bake.rivers.take().ok_or("kernel made no rivers")?);
                }
            }
            FINAL => {
                let bake = self.bake.as_mut().ok_or("missing bake profile")?;
                let mut whole = u64::MAX;
                let baked = bake
                    .advance(&self.sampling.pyramid, &mut whole)
                    .ok_or("profile did not finish")?;
                out.put(&baked);
            }
            _ => return Err("unknown bake child kind".into()),
        }
        Ok(Some(out.into_bytes()))
    }
}
