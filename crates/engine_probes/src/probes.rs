use bevy_ecs::prelude::*;
use engine_core::{Cut, DeviceLimits, FrameTime, View};
use engine_math as math;
use engine_places::meshes::{Instances, STILL_FRAMES};
use engine_places::probes::{ProbeCaptures, ProbeFace, ProbeView};
use engine_places::sky::SkyLight;
use engine_render::CameraUniform;
use glam::{Mat4, Vec3, Vec4};

use crate::gather::{ProbeList, RenderReflectionProbe};
use crate::{PROBE_ENTRY_WORDS, ReflectionProbeSettings, ReflectionProbeStats};

pub(crate) const FACES: usize = 6;
pub(crate) const SETS: u32 = 2;
const CAPTURE_NEAR: f32 = 0.05;
const CAPTURE_FAR: f32 = 4_000.0;
const GRID_CELLS: f32 = 64.0;

struct Slot {
    probe: RenderReflectionProbe,
    ready: bool,
    stale: bool,
    age: u32,
}

#[derive(Default)]
struct ProbeSlots {
    slots: Vec<Option<Slot>>,
    capturing: Option<(usize, usize)>,
    changed: bool,
    changed_during: bool,
}

impl ProbeSlots {
    fn plan(
        &mut self,
        probes: &[RenderReflectionProbe],
        capacity: usize,
        lost: bool,
        sky_changed: bool,
    ) {
        if lost {
            self.slots.clear();
            self.capturing = None;
            self.changed = true;
        }
        if self.slots.len() != capacity {
            self.slots.resize_with(capacity, || None);
            if self
                .capturing
                .is_some_and(|(capturing, _)| capturing >= capacity)
            {
                self.capturing = None;
            }
            self.changed = true;
        }
        let wanted = &probes[..probes.len().min(capacity)];
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let keep = slot.as_ref().is_some_and(|slot| {
                wanted.iter().any(|probe| {
                    probe.key == slot.probe.key
                        && probe.center == slot.probe.center
                        && probe.extent == slot.probe.extent
                })
            });
            if !keep && slot.is_some() {
                *slot = None;
                self.changed = true;
                if self
                    .capturing
                    .is_some_and(|(capturing, _)| capturing == index)
                {
                    self.capturing = None;
                }
            }
            if let Some(slot) = slot {
                if sky_changed {
                    slot.stale = true;
                }
                if let Some(probe) = wanted.iter().find(|probe| probe.key == slot.probe.key)
                    && (probe.blend != slot.probe.blend || probe.interior != slot.probe.interior)
                {
                    slot.probe.blend = probe.blend;
                    slot.probe.interior = probe.interior;
                    self.changed = true;
                }
            }
        }
        for probe in wanted {
            if self
                .slots
                .iter()
                .flatten()
                .any(|slot| slot.probe.key == probe.key)
            {
                continue;
            }
            if let Some(free) = self.slots.iter_mut().find(|slot| slot.is_none()) {
                *free = Some(Slot {
                    probe: *probe,
                    ready: false,
                    stale: false,
                    age: 0,
                });
                self.changed |= probe.interior.is_some();
            }
        }
        if sky_changed && self.capturing.is_some() {
            self.changed_during = true;
        }
    }

    fn tick(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            slot.age = (slot.age + 1).min(STILL_FRAMES);
        }
    }

    fn next_face(
        &mut self,
        probes: &[RenderReflectionProbe],
    ) -> Option<(u32, usize, RenderReflectionProbe)> {
        if self.capturing.is_none() {
            let waiting = probes.iter().find_map(|probe| {
                self.slots.iter().position(|slot| {
                    slot.as_ref().is_some_and(|slot| {
                        slot.probe.key == probe.key
                            && (!slot.ready || slot.stale)
                            && slot.age >= STILL_FRAMES
                    })
                })
            })?;
            self.capturing = Some((waiting, 0));
        }
        let (slot, face) = self.capturing?;
        let probe = self.slots[slot].as_ref()?.probe;
        Some((slot as u32 + 1, face, probe))
    }

    fn face_drawn(&mut self) -> bool {
        let Some((slot, face)) = self.capturing else {
            return false;
        };
        if face + 1 < FACES {
            self.capturing = Some((slot, face + 1));
            return false;
        }
        self.capturing = None;
        let stale = std::mem::take(&mut self.changed_during);
        if let Some(slot) = self.slots[slot].as_mut() {
            slot.ready = true;
            slot.stale = stale;
            self.changed = true;
        }
        true
    }

    fn waiting(&self) -> bool {
        self.capturing.is_some()
            || self
                .slots
                .iter()
                .flatten()
                .any(|slot| !slot.ready || slot.stale)
    }

    fn ready(&self) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.ready)
            .count() as u32
    }

    fn changed_ready(&mut self) -> Option<Vec<(RenderReflectionProbe, u32)>> {
        if !std::mem::take(&mut self.changed) {
            return None;
        }
        Some(
            self.slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| {
                    slot.as_ref()
                        .filter(|slot| slot.ready || slot.probe.interior.is_some())
                        .map(|slot| (slot.probe, if slot.ready { index as u32 + 1 } else { 0 }))
                })
                .collect(),
        )
    }
}

#[derive(Default)]
struct Stillness {
    slots: u64,
    counting: bool,
    counted: u32,
}

impl Stillness {
    fn ensure(&mut self, stillness: Option<bool>, capacity: u64) {
        let slots = if stillness.is_some() { capacity } else { 0 };
        if slots > self.slots || (slots == 0 && self.slots != 0) {
            self.slots = slots;
            self.counted = 0;
        }
        let counting = stillness == Some(true);
        if counting && !self.counting {
            self.counted = 0;
        }
        self.counting = counting;
        if counting {
            self.counted = (self.counted + 1).min(STILL_FRAMES);
        }
    }

    fn settled(&self) -> bool {
        self.counting && self.counted >= STILL_FRAMES
    }
}

#[derive(Resource)]
pub(crate) struct ProbeState {
    slots: ProbeSlots,
    stillness: Stillness,
    pub(crate) size: u32,
    pub(crate) cubes: u32,
    sky: u64,
    set: u32,
    pub(crate) listed: bool,
    pub(crate) captures: bool,
    views: bool,
    still: bool,
    pub(crate) timed: bool,
    pub(crate) faces: Vec<ProbeFace>,
    pub(crate) drawn: Vec<(ProbeView, u32)>,
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) finished: Option<(u32, u32)>,
    words: Vec<[u32; 4]>,
    words_changed: bool,
    stats: ReflectionProbeStats,
}

impl Default for ProbeState {
    fn default() -> Self {
        Self {
            slots: ProbeSlots {
                changed: true,
                ..ProbeSlots::default()
            },
            stillness: Stillness::default(),
            size: ReflectionProbeSettings::default().resolution,
            cubes: 1,
            sky: 0,
            set: 0,
            listed: false,
            captures: false,
            views: false,
            still: false,
            timed: false,
            faces: Vec::new(),
            drawn: Vec::new(),
            cameras: Vec::new(),
            finished: None,
            words: Vec::new(),
            words_changed: false,
            stats: ReflectionProbeStats::default(),
        }
    }
}

impl ProbeState {
    pub(crate) fn probe_cubes(&self) -> u32 {
        (self.cubes - 1).max(1)
    }

    fn place(&mut self, probes: &[RenderReflectionProbe], size: u32, most_cubes: u32) -> bool {
        let wanted_cubes = 1 + (probes.len() as u32).min(most_cubes - 1);
        let cubes = if wanted_cubes == 1 {
            1
        } else if wanted_cubes > self.cubes {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.cubes
        };
        if self.size == size && self.cubes == cubes {
            return false;
        }
        let kept = self.size == size && cubes > self.cubes;
        (self.size, self.cubes) = (size, cubes);
        !kept
    }

    fn capture(
        &mut self,
        probes: &[RenderReflectionProbe],
        (faces, culls): (u32, bool),
        seconds: f32,
    ) -> u32 {
        let mut used = [false; FACES];
        let mut drawn = 0;
        while drawn < faces {
            let Some((cube, face, probe)) = self.slots.next_face(probes) else {
                break;
            };
            if used[face] {
                break;
            }
            used[face] = true;
            self.timed = true;
            if !culls {
                break;
            }
            let view = face_view_projection(probe.center, face);
            let index = self.faces.len() as u32;
            self.faces.push(ProbeFace {
                face: face as u32,
                center: probe.center,
                view,
                cube,
            });
            self.drawn.push((
                ProbeView {
                    face: face as u32,
                    center: probe.center,
                    still: self.still,
                },
                self.set * FACES as u32 + face as u32,
            ));
            self.cameras.push(probe_camera(
                view,
                probe.center,
                seconds,
                (self.size, 1 + index),
            ));
            drawn += 1;
            if self.slots.face_drawn() {
                self.finished = Some((cube, self.set));
                self.set = (self.set + 1) % SETS;
            }
        }
        drawn
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the plan joins the probes, the sky, the meshes' culling and the view"
)]
pub(crate) fn plan_captures(
    settings: Res<ReflectionProbeSettings>,
    limits: Res<DeviceLimits>,
    view: Res<View>,
    sky: Res<SkyLight>,
    instances: Res<Instances>,
    time: Res<FrameTime>,
    gathered: Res<ProbeList>,
    mut state: ResMut<ProbeState>,
) {
    let state = &mut *state;
    let probes = gathered.list.as_slice();
    let present = sky.environment != 0;
    let gpu = !view.exact();
    state.listed = !probes.is_empty();
    state.views = state.listed && present;
    state.still = state.views && state.slots.waiting();
    if gpu {
        state.stillness.ensure(
            state.views.then_some(state.still),
            u64::from(instances.capacity),
        );
    }
    let (cubes, most_cubes, lost, sky_changed) = if present {
        let most_cubes = (limits.texture_layers / FACES as u32).max(1);
        let lost = state.place(probes, settings.resolution, most_cubes);
        let sky_changed = sky.environment != state.sky;
        state.sky = sky.environment;
        (state.cubes, most_cubes, lost, sky_changed)
    } else {
        (1, 1, false, false)
    };
    state
        .slots
        .plan(probes, (cubes - 1) as usize, lost, sky_changed);
    if let Some(ready) = state.slots.changed_ready() {
        grid_words(&ready, &mut state.words);
        state.words_changed = true;
    }
    state.faces.clear();
    state.drawn.clear();
    state.cameras.clear();
    state.finished = None;
    state.timed = false;
    let count = probes.len() as u64;
    let kept = count.min(u64::from(most_cubes - 1));
    let asked = gathered.cut as u64 + count;
    let counted = |count: u64| count.min(u64::from(u32::MAX)) as u32;
    let probe_cut = Cut::of(counted(asked), counted(kept));
    let mut drawn = 0;
    state.captures = gpu && !view.debug() && state.listed;
    if state.captures {
        state.slots.tick();
        if state.stillness.settled() && present {
            drawn = state.capture(
                probes,
                (settings.faces_per_frame, instances.candidates != 0),
                time.elapsed_seconds as f32,
            );
        }
    }
    state.stats = ReflectionProbeStats {
        ready: state.slots.ready(),
        faces: drawn,
        probe_cut,
    };
}

pub(crate) fn publish_captures(
    mut state: ResMut<ProbeState>,
    mut words: ResMut<ProbeWords>,
    mut captures: ResMut<ProbeCaptures>,
    mut stats: ResMut<ReflectionProbeStats>,
) {
    let state = &mut *state;
    if std::mem::take(&mut state.words_changed) {
        std::mem::swap(&mut words.words, &mut state.words);
    }
    if captures.faces != state.faces
        || captures.size != state.size
        || captures.views != state.views
        || captures.still != state.still
    {
        let captures = &mut *captures;
        captures.faces.clone_from(&state.faces);
        captures.size = state.size;
        captures.views = state.views;
        captures.still = state.still;
    }
    stats.set_if_neq(state.stats.clone());
}

pub(crate) fn forget_probes(
    mut gathered: ResMut<ProbeList>,
    mut state: ResMut<ProbeState>,
    mut words: ResMut<ProbeWords>,
    mut stats: ResMut<ReflectionProbeStats>,
) {
    *gathered = ProbeList::default();
    *state = ProbeState::default();
    *words = ProbeWords::default();
    *stats = ReflectionProbeStats::default();
}

#[derive(Resource, Default)]
pub(crate) struct ProbeWords {
    pub(crate) words: Vec<[u32; 4]>,
}

fn face_view_projection(center: Vec3, face: usize) -> Mat4 {
    let (right, up, back) = match face {
        0 => (Vec3::NEG_Z, Vec3::Y, Vec3::NEG_X),
        1 => (Vec3::Z, Vec3::Y, Vec3::X),
        2 => (Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
        3 => (Vec3::X, Vec3::Z, Vec3::Y),
        4 => (Vec3::X, Vec3::Y, Vec3::NEG_Z),
        _ => (Vec3::NEG_X, Vec3::Y, Vec3::Z),
    };
    let row = |axis: Vec3| axis.extend(-axis.dot(center));
    let view = Mat4::from_cols(row(right), row(up), row(back), Vec4::W).transpose();
    Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, CAPTURE_FAR, CAPTURE_NEAR) * view
}

fn probe_camera(
    view_projection: Mat4,
    position: Vec3,
    seconds: f32,
    (size, view): (u32, u32),
) -> CameraUniform {
    let columns = view_projection.to_cols_array_2d();
    CameraUniform {
        view_projection: columns,
        previous_view_projection: columns,
        inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
        position: position.extend(seconds).to_array(),
        jitter: [0.0; 4],
        options: [1.0, pixel_angle(view_projection, size), 0.0, 0.0],
        wind: [0.0, 0.0, 0.0, seconds],
        region: [size as f32, size as f32, 1.0, 1.0],
        detail: position.extend(0.0).to_array(),
        view: [view, 0, 0, 0],
    }
}

fn pixel_angle(view_projection: Mat4, height: u32) -> f32 {
    let row = view_projection.row(1).truncate().length() * height.max(1) as f32;
    2.0 / math::max(row, 1.0e-12)
}

fn grid_words(probes: &[(RenderReflectionProbe, u32)], words: &mut Vec<[u32; 4]>) {
    words.clear();
    if probes.is_empty() {
        words.extend([[0; 4], [0; 4]]);
        return;
    }
    let (mut low, mut high) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let mut smallest = f32::INFINITY;
    for (probe, _) in probes {
        low = low.min(probe.center - probe.extent);
        high = high.max(probe.center + probe.extent);
        smallest = math::min(smallest, math::min(probe.extent.x, probe.extent.z) * 2.0);
    }
    let span = high - low;
    let cell = math::max(smallest * 0.5, math::max(span.x, span.z) / GRID_CELLS);
    let across = |length: f32| ((length / cell).ceil() as u32).max(1);
    let (columns, rows) = (across(span.x), across(span.z));
    let volume = |index: usize| {
        let extent = probes[index].0.extent;
        extent.x * extent.y * extent.z
    };
    let mut order: Vec<usize> = (0..probes.len()).collect();
    order.sort_by(|&a, &b| volume(a).total_cmp(&volume(b)).then(a.cmp(&b)));
    let mut lists = vec![Vec::new(); (columns * rows) as usize];
    for &index in &order {
        let probe = probes[index].0;
        let first = ((probe.center - probe.extent - low) / cell).floor();
        let last = ((probe.center + probe.extent - low) / cell).floor();
        let rows_reached = (math::max(first.z, 0.0) as u32)..=(last.z as u32).min(rows - 1);
        for row in rows_reached {
            for column in (math::max(first.x, 0.0) as u32)..=(last.x as u32).min(columns - 1) {
                lists[(row * columns + column) as usize].push(index as u32);
            }
        }
    }
    let count = probes.len() as u32;
    let cells_at = 2;
    let entries_at = cells_at + columns * rows;
    words.push([low.x.to_bits(), low.z.to_bits(), cell.to_bits(), count]);
    words.push([columns, rows, cells_at, entries_at]);
    let mut start = 0;
    for list in &lists {
        let interiors = list
            .iter()
            .filter(|&&index| probes[index as usize].0.interior.is_some())
            .count() as u32;
        words.push([start, list.len() as u32, interiors, 0]);
        start += list.len() as u32;
    }
    for &index in lists.iter().flatten() {
        let (probe, cube) = probes[index as usize];
        let interior = probe.interior;
        let entry: [[u32; 4]; PROBE_ENTRY_WORDS as usize] = [
            [
                probe.center.x.to_bits(),
                probe.center.y.to_bits(),
                probe.center.z.to_bits(),
                probe.blend.to_bits(),
            ],
            [
                probe.extent.x.to_bits(),
                probe.extent.y.to_bits(),
                probe.extent.z.to_bits(),
                cube,
            ],
            [
                interior.map_or(1.0, |interior| interior.sky).to_bits(),
                interior.map_or(1.0, |interior| interior.sun).to_bits(),
                u32::from(interior.is_some()),
                u32::from(cube != 0),
            ],
        ];
        words.extend(entry);
    }
}
