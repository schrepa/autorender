//! Runtime deformation: ground that remembers where things
//! went. Each fixed tick every [`Deformer`] within its contact distance of
//! a terrain presses a stamp into [`Tracks`], the log of every stamp: a
//! capsule from where it pressed the tick before to where it stands. Frames
//! carry the stamps pressed since the last into a map around the camera that
//! renderers keep, how deep tracks sink the ground and how high their rims
//! rise texel by texel, and terrain surface materials say how deep tracks
//! sink them at most ([`crate::SurfaceMaterial::deform`]): snow deeply, mud
//! less, rock never. The map follows the camera in steps of an eighth of its
//! side; the ground it newly covers gets the log's stamps over it again, so
//! tracks persist where the camera returns. Tracks refill over
//! [`Deformation::refill`] seconds, a decay renderers apply to the whole
//! map at once. Queries read the log, never the GPU.

use std::collections::{HashMap, VecDeque};

use bevy_ecs::prelude::*;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_budget, check_range};
use crate::math::{clamp, max, min};
use crate::{Camera, FixedTime, GlobalTransforms, Terrains};

/// The most meters a stamp sinks the ground.
pub const MAX_TRACK_DEPTH: f32 = 10.0;
/// Where a stamp's rim starts and ends, in radii from its middle line: it
/// rises from 0.9, highest at 1.25, and is gone by 1.6.
pub const RIM_START: f32 = 0.9;
pub const RIM_REACH: f32 = 1.6;
/// The map's texels along a side are a multiple of this; it moves in steps
/// of an eighth of them.
pub const DEFORMATION_GRAIN: u32 = 128;
const STEPS: i32 = 8;
/// Meters on a side of the log's cells, which queries and replays look
/// stamps up by.
const CELL: f32 = 16.0;
/// One deformer's stamps merge into one capsule while it keeps a heading,
/// stamps at most this many seconds apart...
const MERGE_SECONDS: f64 = 1.0;
/// ...up to this many meters long.
const MERGE_LENGTH: f32 = 32.0;
/// Refilled shallower than this (meters), a stamp leaves the log.
const FORGOTTEN: f32 = 5.0e-4;

/// Runtime deformation: whether deformers press tracks; the
/// meters of ground the map around the camera spans (8 to 1024) and its
/// texels along a side (128 to 8192, a multiple of 128; renderers keep
/// fewer where the device's buffers cannot hold them); the seconds tracks
/// take to refill to about a third of their depth (0, never, to 86400);
/// how high the ground rises beside a track for its depth (0 to 1); an
/// optional budget of stamps the log keeps, the oldest dropping beyond it;
/// and whether the log keeps the stamps the map moves away from, so tracks
/// are there when the camera returns. Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneDeformation")
)]
#[serde(default, deny_unknown_fields)]
pub struct Deformation {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 8.0, max = 1024.0)))]
    pub extent: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 128, max = 8192)))]
    pub resolution: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 86400.0)))]
    pub refill: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub rim: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_stamps: Option<u32>,
    pub persist: bool,
}

impl Default for Deformation {
    fn default() -> Self {
        Self {
            enabled: true,
            extent: 64.0,
            resolution: 1024,
            refill: 0.0,
            rim: 0.3,
            max_stamps: None,
            persist: true,
        }
    }
}

impl Deformation {
    /// What a world without these settings reads as: no tracks.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("extent", self.extent, 8.0..=1024.0)?;
        if !(DEFORMATION_GRAIN..=8192).contains(&self.resolution)
            || !self.resolution.is_multiple_of(DEFORMATION_GRAIN)
        {
            return Err(InvalidSetting::new(
                "resolution",
                "must be in [128, 8192] and a multiple of 128",
            ));
        }
        check_range("refill", self.refill, 0.0..=86_400.0)?;
        check_range("rim", self.rim, 0.0..=1.0)?;
        check_budget("max_stamps", self.max_stamps)
    }

    /// Every value in range: out-of-range values clamp, unusable ones fall
    /// back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                clamp(value, low, high)
            } else {
                fallback
            }
        };
        let resolution = self.resolution.clamp(DEFORMATION_GRAIN, 8192);
        Self {
            enabled: self.enabled,
            extent: within(self.extent, 8.0, 1024.0, defaults.extent),
            resolution: resolution / DEFORMATION_GRAIN * DEFORMATION_GRAIN,
            refill: within(self.refill, 0.0, 86_400.0, defaults.refill),
            rim: within(self.rim, 0.0, 1.0, defaults.rim),
            max_stamps: self.max_stamps.map(|budget| budget.max(1)),
            persist: self.persist,
        }
    }

    /// How much of a track's depth is left `age` seconds after it was
    /// pressed.
    pub fn left(&self, age: f64) -> f32 {
        if self.refill > 0.0 {
            (-max(age as f32, 0.0) / self.refill).exp()
        } else {
            1.0
        }
    }
}

/// Presses tracks into deformable ground as a scene version
/// 9 `deformer` field: each fixed tick, while its entity's world position
/// lies at most `contact` meters above the terrain under it (or below it),
/// it presses a footprint `radius` meters across, from where it pressed the
/// tick before, sinking ground as deformable as that `depth` meters along
/// its middle (deeper by however far it lies below the ground), round
/// (`flatness` 0, deepest along its middle like a wheel's rut) to flat (1,
/// a foot's or a plate's level floor). Put it where an object meets the
/// ground: a wheel's or a foot's lowest point. Radius 0.01 to 50, depth
/// and contact 0 to 10, flatness 0 to 1.
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct Deformer {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 50.0)))]
    pub radius: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10.0)))]
    pub depth: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10.0)))]
    pub contact: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub flatness: f32,
}

impl Default for Deformer {
    fn default() -> Self {
        Self {
            radius: 0.3,
            depth: 0.15,
            contact: 0.1,
            flatness: 0.5,
        }
    }
}

impl Deformer {
    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("radius", self.radius, 0.01..=50.0)?;
        check_range("depth", self.depth, 0.0..=MAX_TRACK_DEPTH)?;
        check_range("contact", self.contact, 0.0..=10.0)?;
        check_range("flatness", self.flatness, 0.0..=1.0)
    }
}

/// One stamp: a capsule over the ground from world `from` to `to` (x and
/// z) of `radius` meters (0.01 to 50), sinking ground as deformable as that
/// `depth` meters along its middle line (0 to 10), round (`flatness` 0) to
/// flat (1), pressed at simulation time `seconds`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Track {
    pub from: [f32; 2],
    pub to: [f32; 2],
    pub radius: f32,
    pub depth: f32,
    pub flatness: f32,
    pub seconds: f64,
}

impl Track {
    /// The track with every value in range, or `None` where one is not
    /// finite.
    pub fn usable(self) -> Option<Self> {
        let finite = self
            .from
            .iter()
            .chain(&self.to)
            .all(|value| value.is_finite())
            && self.radius.is_finite()
            && self.depth.is_finite()
            && self.flatness.is_finite()
            && self.seconds.is_finite();
        finite.then(|| Self {
            radius: clamp(self.radius, 0.01, 50.0),
            depth: clamp(self.depth, 0.0, MAX_TRACK_DEPTH),
            flatness: clamp(self.flatness, 0.0, 1.0),
            ..self
        })
    }

    /// How deep the stamp sinks the ground at world `point` (x and z), and
    /// how high its rim stands there for a rim share of 1, as pressed.
    pub fn at(&self, point: Vec2) -> (f32, f32) {
        let s = self.distance(point) / self.radius;
        (
            self.depth * carve_profile(s, self.flatness),
            self.depth * rim_profile(s),
        )
    }

    /// Meters from world `point` to the capsule's middle line.
    fn distance(&self, point: Vec2) -> f32 {
        let from = Vec2::from(self.from);
        let along = Vec2::from(self.to) - from;
        let length = along.length_squared();
        let t = if length > 0.0 {
            clamp((point - from).dot(along) / length, 0.0, 1.0)
        } else {
            0.0
        };
        point.distance(from + along * t)
    }

    /// The corners of the ground the stamp and its rim reach (world x and
    /// z).
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let reach = Vec2::splat(self.radius * RIM_REACH);
        let (from, to) = (Vec2::from(self.from), Vec2::from(self.to));
        (from.min(to) - reach, from.max(to) + reach)
    }
}

/// A stamp's depth `s` radii from its middle line, as a share of its depth:
/// a sphere's cap when round, a level floor with a rounded edge when flat.
pub fn carve_profile(s: f32, flatness: f32) -> f32 {
    if s >= 1.0 {
        return 0.0;
    }
    let round = max(1.0 - s * s, 0.0).sqrt();
    let edge = clamp((s - 0.7) / 0.3, 0.0, 1.0);
    let flat = 1.0 - edge * edge * (3.0 - 2.0 * edge);
    round + (flat - round) * flatness
}

/// A stamp's rim `s` radii from its middle line, as a share of its depth
/// for a rim share of 1: a hump from [`RIM_START`] to [`RIM_REACH`].
pub fn rim_profile(s: f32) -> f32 {
    let u = (s - RIM_START) / (RIM_REACH - RIM_START);
    if !(0.0..=1.0).contains(&u) {
        return 0.0;
    }
    let hump = u * (1.0 - u);
    16.0 * hump * hump
}

/// The tracks at a point, as the log holds them: the meters the deepest
/// sinks deformable ground (`track`) and the meters the highest rim
/// raises it (`rim`, the rim share applied), refilled to `seconds`. Ground
/// shows `min(track, deform)` of its surface material's
/// [`crate::SurfaceMaterial::deform`], and a rim where no track sinks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct TrackDepth {
    pub track: f32,
    pub rim: f32,
}

/// Where a deformer pressed last: the point, and its stamp in the log.
#[derive(Clone, Copy, Debug)]
struct Pressing {
    at: Vec2,
    sequence: u64,
}

/// Every stamp deformers and agents pressed, oldest first,
/// and those pressed since the last frame. Inserted by default; frames and
/// queries read it, [`Deformation`] bounds it.
#[derive(Resource, Default)]
pub struct Tracks {
    stamps: VecDeque<Track>,
    // The sequence number of the oldest stamp; each stamp's is its place
    // after it.
    first: u64,
    // Each cell's stamps, by sequence number.
    cells: HashMap<(i32, i32), Vec<u64>>,
    pending: Vec<Track>,
    pressing: HashMap<Entity, Pressing>,
    dropped: u64,
    pressed: u64,
    // Set by `clear`: the next frame starts the map over.
    cleared: bool,
}

impl Tracks {
    /// Stamps the log holds.
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    /// Stamps the budget dropped, and stamps pressed, since the start.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn pressed(&self) -> u64 {
        self.pressed
    }

    /// The log's stamps, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &Track> {
        self.stamps.iter()
    }

    /// The newest stamp, or the one the newest stamp extended.
    pub fn latest(&self) -> Option<&Track> {
        self.stamps.back()
    }

    /// Presses `track`, as deformers do; `false`, changing nothing, where a
    /// value is not finite. Values out of range clamp.
    pub fn press(&mut self, track: Track, settings: &Deformation) -> bool {
        let Some(track) = track.usable() else {
            return false;
        };
        self.append(track, None, settings);
        true
    }

    /// Forgets every stamp; the next frame clears the map.
    pub fn clear(&mut self) {
        let dropped = self.dropped;
        let pressed = self.pressed;
        *self = Self {
            dropped,
            pressed,
            cleared: true,
            ..Self::default()
        };
    }

    /// The tracks at world `point` (x and z), refilled to simulation time
    /// `seconds` as `settings` refill them.
    pub fn at(&self, point: Vec2, seconds: f64, settings: &Deformation) -> TrackDepth {
        let mut depth = TrackDepth::default();
        let Some(sequences) = self.cells.get(&cell(point)) else {
            return depth;
        };
        for &sequence in sequences {
            let Some(track) = self.stamp(sequence) else {
                continue;
            };
            let left = settings.left(seconds - track.seconds);
            let (carve, rim) = track.at(point);
            depth.track = max(depth.track, carve * left);
            depth.rim = max(depth.rim, rim * left * settings.rim);
        }
        depth
    }

    fn stamp(&self, sequence: u64) -> Option<&Track> {
        self.stamps
            .get(usize::try_from(sequence.checked_sub(self.first)?).ok()?)
    }

    /// Logs `track`, pressed by `deformer` after its stamp `last` if
    /// either, and lists it for the next frame.
    fn append(
        &mut self,
        track: Track,
        deformer: Option<(Entity, Option<u64>)>,
        settings: &Deformation,
    ) {
        self.pressed += 1;
        self.pending.push(track);
        let merged = deformer
            .and_then(|(_, last)| last)
            .and_then(|last| self.merge(last, &track));
        let sequence = merged.unwrap_or_else(|| self.push(track));
        if let Some((entity, _)) = deformer {
            self.pressing.insert(
                entity,
                Pressing {
                    at: Vec2::from(track.to),
                    sequence,
                },
            );
        }
        self.trim(track.seconds, settings);
    }

    /// Extends the stamp `sequence` with `track` where they make one
    /// capsule: the same footprint pressed on from its end, standing still
    /// or keeping its heading, soon after.
    fn merge(&mut self, sequence: u64, track: &Track) -> Option<u64> {
        let index = usize::try_from(sequence.checked_sub(self.first)?).ok()?;
        let last = *self.stamps.get(index)?;
        // Deformers below the ground press deeper as they sink: depths
        // within a tenth or so of each other merge at the deeper.
        let same = last.radius == track.radius
            && (last.depth - track.depth).abs() <= 0.1 * max(last.depth, track.depth) + 0.005
            && last.flatness == track.flatness
            && track.seconds - last.seconds <= MERGE_SECONDS
            && Vec2::from(last.to).distance(Vec2::from(track.from)) <= 1.0e-3;
        if !same {
            return None;
        }
        // Within a degree of its heading, so curves stay within a few
        // millimeters of their capsules; steps too short to have one, and
        // capsules too short to, always merge.
        let along = Vec2::from(last.to) - Vec2::from(last.from);
        let step = Vec2::from(track.to) - Vec2::from(track.from);
        let turned = step.length() >= 1.0e-3
            && along.length() >= 0.01
            && along.dot(step) < 0.99985 * along.length() * step.length();
        let length = Vec2::from(last.from).distance(Vec2::from(track.to));
        if turned || length > MERGE_LENGTH {
            return None;
        }
        let merged = Track {
            to: track.to,
            depth: max(last.depth, track.depth),
            seconds: track.seconds,
            ..last
        };
        let (old_min, old_max) = cell_range(last.bounds());
        let (new_min, new_max) = cell_range(merged.bounds());
        for z in new_min.1..=new_max.1 {
            for x in new_min.0..=new_max.0 {
                let inside =
                    (old_min.0..=old_max.0).contains(&x) && (old_min.1..=old_max.1).contains(&z);
                if !inside {
                    self.cells.entry((x, z)).or_default().push(sequence);
                }
            }
        }
        self.stamps[index] = merged;
        Some(sequence)
    }

    fn push(&mut self, track: Track) -> u64 {
        let sequence = self.first + self.stamps.len() as u64;
        let (low, high) = cell_range(track.bounds());
        for z in low.1..=high.1 {
            for x in low.0..=high.0 {
                self.cells.entry((x, z)).or_default().push(sequence);
            }
        }
        self.stamps.push_back(track);
        sequence
    }

    fn pop_front(&mut self) {
        let Some(track) = self.stamps.pop_front() else {
            return;
        };
        let sequence = self.first;
        self.first += 1;
        let (low, high) = cell_range(track.bounds());
        for z in low.1..=high.1 {
            for x in low.0..=high.0 {
                let Some(list) = self.cells.get_mut(&(x, z)) else {
                    continue;
                };
                if list.first() == Some(&sequence) {
                    list.remove(0);
                } else {
                    list.retain(|&other| other != sequence);
                }
                if list.is_empty() {
                    self.cells.remove(&(x, z));
                }
            }
        }
    }

    /// Drops the oldest stamps beyond the budget, and those refilled away.
    fn trim(&mut self, seconds: f64, settings: &Deformation) {
        if let Some(budget) = settings.max_stamps {
            while self.stamps.len() > budget as usize {
                self.pop_front();
                self.dropped += 1;
            }
        }
        if settings.refill > 0.0 {
            while let Some(front) = self.stamps.front()
                && front.depth * settings.left(seconds - front.seconds) < FORGOTTEN
            {
                self.pop_front();
            }
        }
    }

    /// Appends the stamps reaching any of `rects` (world corners, x and z),
    /// each once and oldest first, to `out`.
    fn within(&self, rects: &[(Vec2, Vec2)], out: &mut Vec<Track>) {
        let mut sequences = Vec::new();
        for &(low, high) in rects {
            let (first, last) = cell_range((low, high));
            for z in first.1..=last.1 {
                for x in first.0..=last.0 {
                    if let Some(list) = self.cells.get(&(x, z)) {
                        sequences.extend(list.iter().copied());
                    }
                }
            }
        }
        sequences.sort_unstable();
        sequences.dedup();
        out.extend(sequences.into_iter().filter_map(|sequence| {
            let track = self.stamp(sequence)?;
            let (min, max) = track.bounds();
            rects
                .iter()
                .any(|&(low, high)| min.cmple(high).all() && max.cmpge(low).all())
                .then_some(*track)
        }));
    }

    /// Keeps only the stamps reaching the ground from `low` to `high`
    /// (world x and z).
    fn retain_within(&mut self, low: Vec2, high: Vec2) {
        let mut kept = Vec::new();
        self.within(&[(low, high)], &mut kept);
        if kept.len() == self.stamps.len() {
            return;
        }
        let (dropped, pressed, pending) = (
            self.dropped,
            self.pressed,
            std::mem::take(&mut self.pending),
        );
        *self = Self {
            dropped,
            pressed,
            pending,
            ..Self::default()
        };
        for track in kept {
            self.push(track);
        }
    }
}

/// The log cell holding world `point`.
fn cell(point: Vec2) -> (i32, i32) {
    (
        (point.x / CELL).floor() as i32,
        (point.y / CELL).floor() as i32,
    )
}

/// The first and last cells of the ground from `low` to `high`.
fn cell_range((low, high): (Vec2, Vec2)) -> ((i32, i32), (i32, i32)) {
    (cell(low), cell(high))
}

/// Each deformer within its contact distance of the ground presses a stamp
/// from where it pressed the tick before (or where it stands, if it did
/// not, or jumped) to where it stands, at the tick's time. Runs once the
/// tick's transforms are final.
pub(crate) fn press(world: &mut World, deformers: &mut QueryState<(Entity, &Deformer)>) {
    let settings = world
        .get_resource::<Deformation>()
        .copied()
        .map(Deformation::sanitized)
        .filter(|settings| settings.enabled);
    let Some(settings) = settings else {
        if let Some(mut tracks) = world.get_resource_mut::<Tracks>()
            && !tracks.pressing.is_empty()
        {
            tracks.pressing.clear();
        }
        return;
    };
    let mut presses = Vec::new();
    {
        let world = &*world;
        let (Some(terrains), Some(globals)) = (
            world.get_resource::<Terrains>(),
            world.get_resource::<GlobalTransforms>(),
        ) else {
            return;
        };
        for (entity, deformer) in deformers.iter(world) {
            let Some(at) = globals.get(entity).and_then(|global| global.translation()) else {
                continue;
            };
            let Some(ground) = terrains.height_at(at.x, at.z) else {
                continue;
            };
            let gap = at.y - ground;
            if gap > deformer.contact || !gap.is_finite() {
                continue;
            }
            presses.push((
                entity,
                Vec2::new(at.x, at.z),
                *deformer,
                min(deformer.depth + max(-gap, 0.0), MAX_TRACK_DEPTH),
            ));
        }
    }
    let seconds = world.resource::<FixedTime>().elapsed_seconds;
    let Some(mut tracks) = world.get_resource_mut::<Tracks>() else {
        return;
    };
    if presses.is_empty() {
        if !tracks.pressing.is_empty() {
            tracks.pressing.clear();
        }
        return;
    }
    // Deformers that lifted off press anew where they next land: only
    // this tick's are kept.
    let last = std::mem::take(&mut tracks.pressing);
    for (entity, at, deformer, depth) in presses {
        let previous = last.get(&entity).copied();
        // A deformer jumping farther than it could roll in a tick starts
        // over where it lands.
        let reach = max(5.0, 4.0 * deformer.radius);
        let from = previous
            .filter(|previous| previous.at.distance(at) <= reach)
            .map_or(at, |previous| previous.at);
        let track = Track {
            from: from.to_array(),
            to: at.to_array(),
            radius: deformer.radius,
            depth,
            flatness: deformer.flatness,
            seconds,
        };
        if let Some(track) = track.usable() {
            let sequence = previous.map(|previous| previous.sequence);
            tracks.append(track, Some((entity, sequence)), &settings);
        }
    }
}

/// Where a frame's deformation map lies: the world texel
/// index of its first texel along x and z (texel `i` covers world meters
/// `i * texel` to `(i + 1) * texel`), its texels' size (meters) and count
/// along a side, and the settings renderers shade it with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeformationMap {
    pub first: [i32; 2],
    pub texel: f32,
    pub resolution: u32,
    pub refill: f32,
    pub rim: f32,
}

impl DeformationMap {
    /// The map `settings` lay around `eye`: an eighth of its side at a time,
    /// the eye always within its middle quarter.
    pub fn around(eye: glam::Vec3, settings: &Deformation) -> Self {
        let resolution = settings.resolution;
        let texel = settings.extent / resolution as f32;
        let step = (resolution / STEPS as u32) as i32;
        let meters = texel * step as f32;
        let at = |value: f32| ((value / meters).floor() as i32 - STEPS / 2) * step;
        Self {
            first: [at(eye.x), at(eye.z)],
            texel,
            resolution,
            refill: settings.refill,
            rim: settings.rim,
        }
    }

    /// The world corners (x and z) of the texels from `low` to `high`,
    /// world texel indices, `high` excluded.
    fn meters(&self, low: [i32; 2], high: [i32; 2]) -> (Vec2, Vec2) {
        (
            Vec2::new(low[0] as f32, low[1] as f32) * self.texel,
            Vec2::new(high[0] as f32, high[1] as f32) * self.texel,
        )
    }

    /// The world corners of the whole map.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let size = self.resolution as i32;
        self.meters(self.first, [self.first[0] + size, self.first[1] + size])
    }

    /// The ground this map covers that `last` did not, as world corners.
    fn uncovered(&self, last: &Self) -> Vec<(Vec2, Vec2)> {
        let size = self.resolution as i32;
        let [x, z] = self.first;
        let [lx, lz] = last.first;
        let overlap = [
            (x.max(lx), (x + size).min(lx + size)),
            (z.max(lz), (z + size).min(lz + size)),
        ];
        if overlap[0].0 >= overlap[0].1 || overlap[1].0 >= overlap[1].1 {
            return vec![self.bounds()];
        }
        let mut rects = Vec::new();
        // Columns along x the last map did not cover, every row.
        if x < lx {
            rects.push(self.meters([x, z], [lx, z + size]));
        } else if x > lx {
            rects.push(self.meters([lx + size, z], [x + size, z + size]));
        }
        // Rows along z it did not cover, within the shared columns.
        if z < lz {
            rects.push(self.meters([overlap[0].0, z], [overlap[0].1, lz]));
        } else if z > lz {
            rects.push(self.meters([overlap[0].0, lz + size], [overlap[0].1, z + size]));
        }
        rects
    }
}

/// A frame's deformation: where the map lies (`None` while
/// deformation is off), whether renderers clear it first (it started, its
/// size changed or the log was cleared), and the stamps to press into it,
/// oldest first: the log's over ground the map newly covers, then those
/// pressed since the last frame. Also the stamps the log holds and those
/// its budget dropped.
#[derive(Clone, Debug, Default)]
pub struct RenderDeformation {
    pub map: Option<DeformationMap>,
    pub reset: bool,
    pub tracks: Vec<Track>,
    pub logged: usize,
    pub dropped: u64,
}

/// Lays each frame's deformation map around the camera and lists the stamps
/// it presses.
#[derive(Default)]
pub(crate) struct DeformationExtraction {
    map: Option<DeformationMap>,
}

impl DeformationExtraction {
    pub(crate) fn extract(&mut self, world: &mut World, out: &mut RenderDeformation) {
        out.tracks.clear();
        out.reset = false;
        let settings = world
            .get_resource::<Deformation>()
            .copied()
            .map(Deformation::sanitized)
            .filter(|settings| settings.enabled);
        let eye = world.get_resource::<Camera>().map(Camera::eye);
        let Some(mut tracks) = world.get_resource_mut::<Tracks>() else {
            out.map = None;
            self.map = None;
            return;
        };
        out.logged = tracks.len();
        out.dropped = tracks.dropped;
        let (Some(settings), Some(eye)) = (settings, eye) else {
            tracks.pending.clear();
            out.map = None;
            self.map = None;
            return;
        };
        let map = DeformationMap::around(eye, &settings);
        let cleared = std::mem::take(&mut tracks.cleared);
        let reset = cleared
            || self
                .map
                .is_none_or(|last| last.texel != map.texel || last.resolution != map.resolution);
        let moved = self.map.is_some_and(|last| last.first != map.first);
        if (reset || moved) && !settings.persist {
            let (low, high) = map.bounds();
            tracks.retain_within(low, high);
        }
        if reset {
            tracks.pending.clear();
            tracks.within(&[map.bounds()], &mut out.tracks);
        } else {
            if let Some(last) = self.map.filter(|_| moved) {
                tracks.within(&map.uncovered(&last), &mut out.tracks);
            }
            out.tracks.append(&mut tracks.pending);
        }
        out.reset = reset;
        out.map = Some(map);
        out.logged = tracks.len();
        out.dropped = tracks.dropped;
        self.map = Some(map);
    }
}
