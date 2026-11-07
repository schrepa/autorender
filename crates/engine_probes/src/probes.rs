use bevy_ecs::prelude::*;
use engine_core::{
    Cursor, Cut, DeviceLimits, Events, FrameBudget, FrameTime, GlobalTransforms, Mobility,
    PassName, PrimaryView, Readable, Readback, OwnedView, OwnedViews, Spawner, StoreId,
};
use engine_math as math;
use engine_places_rendering::probes::{ProbeCaptures, ProbeFace};
use engine_places_rendering::sky::SkyLight;
use engine_render::CameraUniform;
use glam::{Mat4, Vec3, Vec4};

use crate::gather::{ProbeList, RenderReflectionProbe};
use crate::{
    PROBE_ENTRY_WORDS, ProbePass, ReflectionProbeSettings, ReflectionProbeStats,
    ReflectionProbesBrick,
};

pub(crate) const FACES: usize = 6;
pub(crate) const SETS: u32 = 2;
pub(crate) const CAPTURED_ACK: u32 = 0;
pub(crate) const LOST_ACK: u32 = 1;
const FACE_TEXEL_BYTES: u64 = 12;
const SLOT_ACK_BYTES: usize = 12;
const CAPTURE_NEAR: f32 = 0.05;
const CAPTURE_FAR: f32 = 4_000.0;
const GRID_CELLS: f32 = 64.0;
const BUDGET: &str = "reflection_probes";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Capture {
    Waiting,
    Ready,
    Stale,
}

struct Slot {
    probe: RenderReflectionProbe,
    capture: Capture,
}

impl Slot {
    fn shown(&self) -> bool {
        self.capture != Capture::Waiting
    }
}

enum ProbeAck {
    Captured { ran: bool, slots: Vec<(usize, u64)> },
    Lost,
}

#[derive(Resource, Default)]
pub(crate) struct ProbeAcks {
    acks: Vec<ProbeAck>,
}

pub(crate) fn captured_bytes(ran: bool, slots: &[(usize, u64)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + slots.len() * SLOT_ACK_BYTES);
    bytes.push(u8::from(ran));
    for &(slot, key) in slots {
        bytes.extend_from_slice(&(slot as u32).to_le_bytes());
        bytes.extend_from_slice(&key.to_le_bytes());
    }
    bytes
}

impl Readable for ProbeAcks {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        if tag == LOST_ACK {
            self.acks.push(ProbeAck::Lost);
            return;
        }
        let Some((&ran, listed)) = read.value.split_first() else {
            return;
        };
        let slots = listed
            .as_chunks::<SLOT_ACK_BYTES>()
            .0
            .iter()
            .map(|&[s0, s1, s2, s3, k0, k1, k2, k3, k4, k5, k6, k7]| {
                (
                    u32::from_le_bytes([s0, s1, s2, s3]) as usize,
                    u64::from_le_bytes([k0, k1, k2, k3, k4, k5, k6, k7]),
                )
            })
            .collect();
        self.acks.push(ProbeAck::Captured {
            ran: ran != 0,
            slots,
        });
    }
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
        stale: bool,
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
                if stale && slot.capture == Capture::Ready {
                    slot.capture = Capture::Stale;
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
                    capture: Capture::Waiting,
                });
                self.changed |= probe.interior.is_some();
            }
        }
        if stale && self.capturing.is_some() {
            self.changed_during = true;
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
                        slot.probe.key == probe.key && slot.capture != Capture::Ready
                    })
                })
            })?;
            self.capturing = Some((waiting, 0));
        }
        let (slot, face) = self.capturing?;
        let held = self.slots[slot].as_ref()?;
        Some((slot as u32 + 1, face, held.probe))
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
            slot.capture = if stale {
                Capture::Stale
            } else {
                Capture::Ready
            };
            self.changed = true;
        }
        true
    }

    fn missed(&mut self, slot: usize, key: u64) {
        let Some(held) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return;
        };
        if held.probe.key != key {
            return;
        }
        if self.capturing.is_some_and(|(capturing, _)| capturing == slot) {
            self.capturing = None;
        }
        held.capture = match held.capture {
            Capture::Waiting => Capture::Waiting,
            Capture::Ready | Capture::Stale => Capture::Stale,
        };
    }

    fn capturing(&self) -> Option<(usize, u64)> {
        let (slot, _) = self.capturing?;
        Some((slot, self.slots[slot].as_ref()?.probe.key))
    }

    fn ready(&self) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.shown())
            .count() as u32
    }

    fn unready(&self) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.capture != Capture::Ready)
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
                        .filter(|slot| slot.shown() || slot.probe.interior.is_some())
                        .map(|slot| (slot.probe, if slot.shown() { index as u32 + 1 } else { 0 }))
                })
                .collect(),
        )
    }
}

#[derive(Resource)]
pub(crate) struct ProbeState {
    slots: ProbeSlots,
    moves_seen: Cursor,
    pub(crate) size: u32,
    pub(crate) cubes: u32,
    pub(crate) layout: StoreId,
    sky: Option<StoreId>,
    set: u32,
    maps_lost: bool,
    pub(crate) listed: bool,
    pub(crate) captures: bool,
    views: bool,
    pub(crate) timed: bool,
    pub(crate) faces: Vec<ProbeFace>,
    pub(crate) drawn: Vec<u32>,
    pub(crate) cameras: Vec<CameraUniform>,
    pub(crate) finished: Option<(u32, u32)>,
    pub(crate) taken: Vec<(usize, u64)>,
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
            moves_seen: Cursor::default(),
            size: ReflectionProbeSettings::default().resolution,
            cubes: 1,
            layout: StoreId::next(),
            sky: None,
            set: 0,
            maps_lost: false,
            listed: false,
            captures: false,
            views: false,
            timed: false,
            faces: Vec::new(),
            drawn: Vec::new(),
            cameras: Vec::new(),
            finished: None,
            taken: Vec::new(),
            words: Vec::new(),
            words_changed: false,
            stats: ReflectionProbeStats::default(),
        }
    }
}

impl ProbeState {
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
        if !kept {
            self.layout = StoreId::next();
        }
        !kept
    }

    fn static_moved(&mut self, globals: &GlobalTransforms) -> bool {
        match globals.static_moves_since(&mut self.moves_seen) {
            Events::Listed(listed) => listed.iter().any(|&entity| globals.drawn(entity)),
            Events::Missed(_) => true,
        }
    }

    fn acknowledge(&mut self, acks: &mut ProbeAcks) {
        for ack in acks.acks.drain(..) {
            match ack {
                ProbeAck::Captured { ran: true, .. } => {}
                ProbeAck::Captured { ran: false, slots } => {
                    for (slot, key) in slots {
                        self.slots.missed(slot, key);
                    }
                }
                ProbeAck::Lost => self.maps_lost = true,
            }
        }
    }

    fn capture(
        &mut self,
        probes: &[RenderReflectionProbe],
        budget: &mut FrameBudget,
        seconds: f32,
    ) -> u32 {
        let face_bytes = u64::from(self.size) * u64::from(self.size) * FACE_TEXEL_BYTES;
        let mut used = [false; FACES];
        let mut drawn = 0;
        loop {
            let Some((cube, face, probe)) = self.slots.next_face(probes) else {
                break;
            };
            if used[face] {
                break;
            }
            self.timed = true;
            if !budget.spend(BUDGET, face_bytes) {
                break;
            }
            used[face] = true;
            let view = face_view_projection(probe.center, face);
            let index = self.faces.len() as u32;
            self.faces.push(ProbeFace {
                face: face as u32,
                center: probe.center,
                view,
            });
            self.drawn.push(self.set * FACES as u32 + face as u32);
            self.cameras.push(probe_camera(
                view,
                probe.center,
                seconds,
                (self.size, 1 + index),
            ));
            drawn += 1;
            if let Some(held) = self.slots.capturing()
                && !self.taken.contains(&held)
            {
                self.taken.push(held);
            }
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
    reason = "the plan joins the probes, the sky, the static moves, the budget and the view"
)]
pub(crate) fn plan_captures(
    settings: Res<ReflectionProbeSettings>,
    limits: Res<DeviceLimits>,
    view: Res<PrimaryView>,
    sky: Res<SkyLight>,
    (globals, mut budget): (Res<GlobalTransforms>, ResMut<FrameBudget>),
    time: Res<FrameTime>,
    gathered: Res<ProbeList>,
    mut acks: ResMut<ProbeAcks>,
    mut state: ResMut<ProbeState>,
) {
    let state = &mut *state;
    state.acknowledge(&mut acks);
    let probes = gathered.list.as_slice();
    let present = sky.environment.is_some();
    state.listed = !probes.is_empty();
    state.views = state.listed && present;
    let moved = state.static_moved(&globals);
    let (cubes, most_cubes, placed, sky_changed) = if present {
        let most_cubes = (limits.texture_layers / FACES as u32).max(1);
        let placed = state.place(probes, settings.resolution, most_cubes);
        let sky_changed = sky.environment != state.sky;
        state.sky = sky.environment;
        (state.cubes, most_cubes, placed, sky_changed)
    } else {
        (1, 1, false, false)
    };
    let lost = placed || std::mem::take(&mut state.maps_lost);
    state
        .slots
        .plan(probes, (cubes - 1) as usize, lost, sky_changed || moved);
    if let Some(ready) = state.slots.changed_ready() {
        grid_words(&ready, &mut state.words);
        state.words_changed = true;
    }
    state.faces.clear();
    state.drawn.clear();
    state.cameras.clear();
    state.taken.clear();
    state.finished = None;
    state.timed = false;
    let count = probes.len() as u64;
    let kept = count.min(u64::from(most_cubes - 1));
    let asked = gathered.cut as u64 + count;
    let counted = |count: u64| count.min(u64::from(u32::MAX)) as u32;
    let probe_cut = Cut::of(counted(asked), counted(kept));
    let mut drawn = 0;
    state.captures = !view.debug() && state.listed;
    if state.captures && state.views {
        drawn = state.capture(probes, &mut budget, time.elapsed as f32);
    }
    state.stats = ReflectionProbeStats {
        ready: state.slots.ready(),
        waiting: if state.captures && state.views {
            state.slots.unready()
        } else {
            0
        },
        faces: drawn,
        probe_cut,
    };
}

#[derive(Component)]
pub(crate) struct CaptureViews;

fn capture_views(state: &ProbeState) -> OwnedViews {
    OwnedViews {
        pass: ProbePass::NAME,
        views: state
            .faces
            .iter()
            .map(|face| OwnedView {
                view_projection: face.view.to_cols_array_2d(),
                planes: None,
                eye: face.center.to_array(),
                size: [state.size, state.size],
                detail: 0.0,
                takes: Some(Mobility::Static),
            })
            .collect(),
    }
}

pub(crate) fn publish_captures(
    mut state: ResMut<ProbeState>,
    mut words: ResMut<ProbeWords>,
    mut captures: ResMut<ProbeCaptures>,
    mut stats: ResMut<ReflectionProbeStats>,
    mut published: Query<&mut OwnedViews, With<CaptureViews>>,
    mut spawner: Spawner<ReflectionProbesBrick>,
) {
    let state = &mut *state;
    if std::mem::take(&mut state.words_changed) {
        std::mem::swap(&mut words.words, &mut state.words);
    }
    if captures.faces != state.faces || captures.size != state.size {
        let captures = &mut *captures;
        captures.faces.clone_from(&state.faces);
        captures.size = state.size;
    }
    let views = capture_views(state);
    match published.single_mut() {
        Ok(mut held) => {
            held.set_if_neq(views);
        }
        Err(_) => {
            spawner.spawn((CaptureViews, views));
        }
    }
    stats.set_if_neq(state.stats.clone());
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
