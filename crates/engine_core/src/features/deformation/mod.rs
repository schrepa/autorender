use crate::collections::HashMap;
use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::math::{clamp, max, min};
use crate::{Camera, FixedTime, GlobalTransforms};
use engine_settings::settings;

pub const MAX_TRACK_DEPTH: f32 = 10.0;
pub const RIM_START: f32 = 0.9;
pub const RIM_REACH: f32 = 1.6;
pub const DEFORMATION_GRAIN: u32 = 128;
const STEPS: i32 = 8;
const CELL: f32 = 16.0;
const MERGE_SECONDS: f64 = 1.0;
const MERGE_LENGTH: f32 = 32.0;
const FORGOTTEN: f32 = 5.0e-4;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDeformation")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Deformation {
        pub enabled: bool = true => any,
        pub extent: f32 = 64.0 => range(8.0, 1024.0),
        pub resolution: u32 = 1024 => multiple(DEFORMATION_GRAIN, DEFORMATION_GRAIN, 8192),
        pub refill: f32 = 0.0 => range(0.0, 86_400.0),
        pub rim: f32 = 0.3 => range(0.0, 1.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_stamps: Option<u32> = None => budget,
        pub persist: bool = true => any,
    }
}

impl Deformation {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn left(&self, age: f64) -> f32 {
        if self.refill > 0.0 {
            (-max(age as f32, 0.0) / self.refill).exp()
        } else {
            1.0
        }
    }
}

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Deformer {
        pub radius: f32 = 0.3 => range(0.01, 50.0),
        pub depth: f32 = 0.15 => range(0.0, MAX_TRACK_DEPTH),
        pub contact: f32 = 0.1 => range(0.0, 10.0),
        pub flatness: f32 = 0.5 => range(0.0, 1.0),
    }
}

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

    pub fn at(&self, point: Vec2) -> (f32, f32) {
        let s = self.distance(point) / self.radius;
        (
            self.depth * carve_profile(s, self.flatness),
            self.depth * rim_profile(s),
        )
    }

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

    pub fn bounds(&self) -> (Vec2, Vec2) {
        let reach = Vec2::splat(self.radius * RIM_REACH);
        let (from, to) = (Vec2::from(self.from), Vec2::from(self.to));
        (from.min(to) - reach, from.max(to) + reach)
    }
}

pub fn carve_profile(s: f32, flatness: f32) -> f32 {
    if s >= 1.0 {
        return 0.0;
    }
    let round = max(1.0 - s * s, 0.0).sqrt();
    let edge = clamp((s - 0.7) / 0.3, 0.0, 1.0);
    let flat = 1.0 - edge * edge * (3.0 - 2.0 * edge);
    round + (flat - round) * flatness
}

pub fn rim_profile(s: f32) -> f32 {
    let u = (s - RIM_START) / (RIM_REACH - RIM_START);
    if !(0.0..=1.0).contains(&u) {
        return 0.0;
    }
    let hump = u * (1.0 - u);
    16.0 * hump * hump
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct TrackDepth {
    pub track: f32,
    pub rim: f32,
}

#[derive(Clone, Copy, Debug)]
struct Pressing {
    at: Vec2,
    sequence: u64,
}

#[derive(Resource, Default)]
pub struct Tracks {
    stamps: VecDeque<Track>,
    first: u64,
    cells: HashMap<(i32, i32), Vec<u64>>,
    pending: Vec<Track>,
    pressing: HashMap<Entity, Pressing>,
    dropped: u64,
    pressed: u64,
    cleared: bool,
}

impl Tracks {
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn pressed(&self) -> u64 {
        self.pressed
    }

    pub fn iter(&self) -> impl Iterator<Item = &Track> {
        self.stamps.iter()
    }

    pub fn latest(&self) -> Option<&Track> {
        self.stamps.back()
    }

    pub fn press(&mut self, track: Track, settings: &Deformation) -> bool {
        let Some(track) = track.usable() else {
            return false;
        };
        self.append(track, None, settings);
        true
    }

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

    fn merge(&mut self, sequence: u64, track: &Track) -> Option<u64> {
        let index = usize::try_from(sequence.checked_sub(self.first)?).ok()?;
        let last = *self.stamps.get(index)?;
        let same = last.radius == track.radius
            && (last.depth - track.depth).abs() <= 0.1 * max(last.depth, track.depth) + 0.005
            && last.flatness == track.flatness
            && track.seconds - last.seconds <= MERGE_SECONDS
            && Vec2::from(last.to).distance(Vec2::from(track.from)) <= 1.0e-3;
        if !same {
            return None;
        }
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

fn cell(point: Vec2) -> (i32, i32) {
    (
        (point.x / CELL).floor() as i32,
        (point.y / CELL).floor() as i32,
    )
}

fn cell_range((low, high): (Vec2, Vec2)) -> ((i32, i32), (i32, i32)) {
    (cell(low), cell(high))
}

pub(crate) fn press(world: &mut World, deformers: &mut QueryState<(Entity, &'static Deformer)>) {
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
        let (Some(ground), Some(globals)) = (
            world.get_resource::<crate::Ground>(),
            world.get_resource::<GlobalTransforms>(),
        ) else {
            return;
        };
        for (entity, deformer) in deformers.iter(world) {
            let Some(at) = globals.get(entity).and_then(|global| global.translation()) else {
                continue;
            };
            let Some(ground) = (ground.sample)(world, at.x, at.z).map(|sample| sample.height)
            else {
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
    let last = std::mem::take(&mut tracks.pressing);
    for (entity, at, deformer, depth) in presses {
        let previous = last.get(&entity).copied();
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeformationMap {
    pub first: [i32; 2],
    pub texel: f32,
    pub resolution: u32,
    pub refill: f32,
    pub rim: f32,
}

impl DeformationMap {
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

    fn meters(&self, low: [i32; 2], high: [i32; 2]) -> (Vec2, Vec2) {
        (
            Vec2::new(low[0] as f32, low[1] as f32) * self.texel,
            Vec2::new(high[0] as f32, high[1] as f32) * self.texel,
        )
    }

    pub fn bounds(&self) -> (Vec2, Vec2) {
        let size = self.resolution as i32;
        self.meters(self.first, [self.first[0] + size, self.first[1] + size])
    }

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
        if x < lx {
            rects.push(self.meters([x, z], [lx, z + size]));
        } else if x > lx {
            rects.push(self.meters([lx + size, z], [x + size, z + size]));
        }
        if z < lz {
            rects.push(self.meters([overlap[0].0, z], [overlap[0].1, lz]));
        } else if z > lz {
            rects.push(self.meters([overlap[0].0, lz + size], [overlap[0].1, z + size]));
        }
        rects
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderDeformation {
    pub map: Option<DeformationMap>,
    pub reset: bool,
    pub tracks: Vec<Track>,
    pub logged: usize,
    pub dropped: u64,
}

impl RenderDeformation {
    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.map = other.map;
        self.reset = other.reset;
        self.tracks.clone_from(&other.tracks);
        self.logged = other.logged;
        self.dropped = other.dropped;
    }
}

#[derive(Resource, Default)]
pub(crate) struct DeformationFrame {
    pub(crate) out: RenderDeformation,
    map: Option<DeformationMap>,
}

pub(crate) fn follow_camera(
    settings: Option<Res<Deformation>>,
    camera: Option<Res<Camera>>,
    tracks: Option<ResMut<Tracks>>,
    mut frame: ResMut<DeformationFrame>,
) {
    let frame = &mut *frame;
    follow(
        settings.map(|settings| *settings),
        camera.map(|camera| camera.eye()),
        tracks.map(ResMut::into_inner),
        &mut frame.map,
        &mut frame.out,
    );
}

pub(crate) fn follow(
    settings: Option<Deformation>,
    eye: Option<Vec3>,
    tracks: Option<&mut Tracks>,
    last: &mut Option<DeformationMap>,
    out: &mut RenderDeformation,
) {
    out.tracks.clear();
    out.reset = false;
    let settings = settings
        .map(Deformation::sanitized)
        .filter(|settings| settings.enabled);
    let Some(tracks) = tracks else {
        out.map = None;
        *last = None;
        return;
    };
    out.logged = tracks.len();
    out.dropped = tracks.dropped;
    let (Some(settings), Some(eye)) = (settings, eye) else {
        tracks.pending.clear();
        out.map = None;
        *last = None;
        return;
    };
    let map = DeformationMap::around(eye, &settings);
    let cleared = std::mem::take(&mut tracks.cleared);
    let reset = cleared
        || last.is_none_or(|last| last.texel != map.texel || last.resolution != map.resolution);
    let moved = last.is_some_and(|last| last.first != map.first);
    if (reset || moved) && !settings.persist {
        let (low, high) = map.bounds();
        tracks.retain_within(low, high);
    }
    if reset {
        tracks.pending.clear();
        tracks.within(&[map.bounds()], &mut out.tracks);
    } else {
        if let Some(last) = last.filter(|_| moved) {
            tracks.within(&map.uncovered(&last), &mut out.tracks);
        }
        out.tracks.append(&mut tracks.pending);
    }
    out.reset = reset;
    out.map = Some(map);
    out.logged = tracks.len();
    out.dropped = tracks.dropped;
    *last = Some(map);
}

#[derive(Default)]
pub struct DeformationPlugin;

impl bevy_app::Plugin for DeformationPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Deformation);
        app.init_resource::<Tracks>()
            .init_resource::<DeformationFrame>()
            .add_systems(
                crate::ViewSync,
                follow_camera.in_set(crate::ViewSyncSystems::Follow),
            )
            .add_systems(
                crate::FixedTickEnd,
                press.in_set(crate::FixedTickEndSystems::PressTracks),
            );
    }
}
