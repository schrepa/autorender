use bevy_ecs::prelude::*;
use engine_core::{
    Camera, Clear, Cursor, Cut, Detail, DeviceLimits, Events, FrameBudget, GlobalTransforms,
    Mobility, Mover, Moves, Node, OwnedView, OwnedViews, Pace, PassName, Readable, Readback,
    Settling, Spawner, StoreId, Transform, ViewLayer, mover_of,
};
use engine_math as math;
use engine_places_rendering::probes::{ProbeCaptures, ProbeFace};
use engine_places_rendering::sky::SkyLight;
use engine_places_rendering::sun::Sun;
use glam::{Mat4, Vec3, Vec4};

use crate::captures::{lit_texel_bytes, stored_texel_bytes};
use crate::gather::{ProbeList, RenderReflectionProbe};
use crate::{
    PROBE_ENTRY_WORDS, ProbeLightPass, ProbePass, ReflectionProbeSettings, ReflectionProbeStats,
    ReflectionProbesBrick,
};

pub(crate) const FACES: usize = 6;
pub(crate) const SETS: u32 = 2;
pub(crate) const CAPTURED_ACK: u32 = 0;
pub(crate) const LOST_ACK: u32 = 1;
const SLOT_ACK_BYTES: usize = 12;
const CAPTURE_NEAR: f32 = 0.05;
const CAPTURE_FAR: f32 = 4_000.0;
const GRID_CELLS: f32 = 64.0;
const BUDGET: &str = "reflection_probes";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum Capture {
    #[default]
    Ready,
    Relight,
    Recapture,
    Waiting,
}

impl Capture {
    fn draws(self) -> bool {
        matches!(self, Self::Recapture | Self::Waiting)
    }
}

struct Slot {
    probe: RenderReflectionProbe,
    capture: Capture,
    sun: Vec3,
}

impl Slot {
    fn shown(&self) -> bool {
        self.capture != Capture::Waiting
    }
}

#[derive(Clone, Copy)]
struct Sequence {
    slot: usize,
    face: usize,
    draws: bool,
}

#[derive(Clone, Copy)]
struct Due {
    cube: u32,
    face: usize,
    draws: bool,
    probe: RenderReflectionProbe,
}

struct Marks {
    lost: bool,
    stale: Capture,
    sun: Vec3,
    least: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FaceJob {
    pub(crate) known: ProbeFace,
    pub(crate) stored: u32,
    pub(crate) lit: u32,
    pub(crate) draws: bool,
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

fn turned(from: Vec3, to: Vec3, least: f32) -> bool {
    let (from, to) = (from.normalize_or_zero(), to.normalize_or_zero());
    from != to && from != Vec3::ZERO && to != Vec3::ZERO && from.dot(to) < least
}

#[derive(Default)]
struct ProbeSlots {
    slots: Vec<Option<Slot>>,
    capturing: Option<Sequence>,
    changed: bool,
    during: Capture,
}

impl ProbeSlots {
    fn plan(&mut self, probes: &[RenderReflectionProbe], capacity: usize, marks: &Marks) {
        if marks.lost {
            self.slots.clear();
            self.capturing = None;
            self.changed = true;
        }
        if self.slots.len() != capacity {
            self.slots.resize_with(capacity, || None);
            if self
                .capturing
                .is_some_and(|sequence| sequence.slot >= capacity)
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
            let capturing = self
                .capturing
                .is_some_and(|sequence| sequence.slot == index);
            if !keep && slot.is_some() {
                *slot = None;
                self.changed = true;
                if capturing {
                    self.capturing = None;
                }
            }
            if let Some(slot) = slot {
                let mark = if slot.shown() && turned(slot.sun, marks.sun, marks.least) {
                    marks.stale.max(Capture::Recapture)
                } else {
                    marks.stale
                };
                slot.capture = slot.capture.max(mark);
                if capturing {
                    self.during = self.during.max(mark);
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
                    sun: Vec3::ZERO,
                });
                self.changed |= probe.interior.is_some();
            }
        }
    }

    fn next_face(&mut self, probes: &[RenderReflectionProbe], sun: Vec3) -> Option<Due> {
        if self.capturing.is_none() {
            let waiting = probes.iter().find_map(|probe| {
                self.slots.iter().position(|slot| {
                    slot.as_ref().is_some_and(|slot| {
                        slot.probe.key == probe.key && slot.capture != Capture::Ready
                    })
                })
            })?;
            let held = self.slots[waiting].as_mut()?;
            let draws = held.capture.draws();
            if draws {
                held.sun = sun;
            }
            self.capturing = Some(Sequence {
                slot: waiting,
                face: 0,
                draws,
            });
            self.during = Capture::Ready;
        }
        let sequence = self.capturing?;
        let held = self.slots[sequence.slot].as_ref()?;
        Some(Due {
            cube: sequence.slot as u32 + 1,
            face: sequence.face,
            draws: sequence.draws,
            probe: held.probe,
        })
    }

    fn face_drawn(&mut self) -> bool {
        let Some(sequence) = self.capturing else {
            return false;
        };
        if sequence.face + 1 < FACES {
            self.capturing = Some(Sequence {
                face: sequence.face + 1,
                ..sequence
            });
            return false;
        }
        self.capturing = None;
        let during = std::mem::take(&mut self.during);
        if let Some(slot) = self.slots[sequence.slot].as_mut() {
            slot.capture = during;
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
        if self.capturing.is_some_and(|sequence| sequence.slot == slot) {
            self.capturing = None;
        }
        held.capture = held.capture.max(Capture::Recapture);
    }

    fn capturing(&self) -> Option<(usize, u64)> {
        let sequence = self.capturing?;
        Some((sequence.slot, self.slots[sequence.slot].as_ref()?.probe.key))
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
    views: bool,
    pub(crate) timed: bool,
    pub(crate) jobs: Vec<FaceJob>,
    faces: Vec<ProbeFace>,
    pub(crate) finished: Option<(u32, u32)>,
    pub(crate) taken: Vec<(usize, u64)>,
    words: Vec<[u32; 4]>,
    words_changed: bool,
    stats: ReflectionProbeStats,
    settling: Settling,
    cameras: Vec<Entity>,
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
            views: false,
            timed: false,
            jobs: Vec::new(),
            faces: Vec::new(),
            finished: None,
            taken: Vec::new(),
            words: Vec::new(),
            words_changed: false,
            stats: ReflectionProbeStats::default(),
            settling: Settling::default(),
            cameras: Vec::new(),
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

    fn capture(&mut self, probes: &[RenderReflectionProbe], sun: Vec3, budget: &mut FrameBudget) {
        let texels = u64::from(self.size) * u64::from(self.size);
        let mut used = [false; FACES];
        while let Some(due) = self.slots.next_face(probes, sun) {
            if used[due.face] {
                break;
            }
            self.timed = true;
            let texel_bytes = if due.draws {
                stored_texel_bytes() + lit_texel_bytes()
            } else {
                lit_texel_bytes()
            };
            if !budget.spend(BUDGET, texels * texel_bytes) {
                break;
            }
            used[due.face] = true;
            let face = due.face as u32;
            let known = ProbeFace {
                face,
                center: due.probe.center,
                view: face_view_projection(due.probe.center, due.face),
            };
            self.jobs.push(FaceJob {
                known,
                stored: (due.cube - 1) * FACES as u32 + face,
                lit: self.set * FACES as u32 + face,
                draws: due.draws,
            });
            self.faces.push(known);
            if let Some(held) = self.slots.capturing()
                && !self.taken.contains(&held)
            {
                self.taken.push(held);
            }
            if self.slots.face_drawn() {
                self.finished = Some((due.cube, self.set));
                self.set = (self.set + 1) % SETS;
            }
        }
    }
}

pub(crate) fn plan_captures(
    settings: Res<ReflectionProbeSettings>,
    limits: Res<DeviceLimits>,
    (sky, sun): (Res<SkyLight>, Res<Sun>),
    (globals, mut budget): (Res<GlobalTransforms>, ResMut<FrameBudget>),
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
    let sun = Vec4::from_array(sun.direction).truncate();
    let stale = if moved {
        Capture::Recapture
    } else if sky_changed {
        Capture::Relight
    } else {
        Capture::Ready
    };
    let marks = Marks {
        lost: placed || std::mem::take(&mut state.maps_lost),
        stale,
        sun,
        least: math::sin_cos(settings.sun_turn.to_radians()).1,
    };
    state.slots.plan(probes, (cubes - 1) as usize, &marks);
    if let Some(ready) = state.slots.changed_ready() {
        grid_words(&ready, &mut state.words);
        state.words_changed = true;
    }
    state.jobs.clear();
    state.faces.clear();
    state.taken.clear();
    state.finished = None;
    state.timed = false;
    let count = probes.len() as u64;
    let kept = count.min(u64::from(most_cubes - 1));
    let asked = gathered.cut as u64 + count;
    let counted = |count: u64| count.min(u64::from(u32::MAX)) as u32;
    let probe_cut = Cut::of(counted(asked), counted(kept));
    if state.views {
        state.capture(probes, sun, &mut budget);
    }
    let waiting = if state.views {
        state.slots.unready()
    } else {
        0
    };
    state.settling.left(waiting);
    state.stats = ReflectionProbeStats {
        ready: state.slots.ready(),
        faces: state.jobs.iter().filter(|job| job.draws).count() as u32,
        relit: state.jobs.len() as u32,
        probe_cut,
    };
}

#[derive(Component)]
pub(crate) struct ProbeViews;

#[derive(Component)]
pub(crate) struct ProbeCamera;

type Lenses<'w, 's> = Query<'w, 's, &'static mut Camera, With<ProbeCamera>>;
type Published<'w, 's> = Query<'w, 's, &'static mut OwnedViews, With<ProbeViews>>;

fn face_views(
    (pass, (size, detail)): (&'static str, (u32, f32)),
    cameras: &[Entity],
    job: impl Fn(u32) -> Option<(Vec3, u32)>,
) -> OwnedViews {
    OwnedViews {
        pass,
        views: (0_u32..)
            .zip(cameras)
            .map(|(face, &camera)| {
                let found = job(face);
                OwnedView {
                    camera,
                    target: ViewLayer {
                        layer: found.map_or(0, |(_, layer)| layer),
                        rect: [0, 0, size, size],
                    },
                    clear: Clear::All,
                    capture: true,
                    passes: None,
                    draws: found.is_some(),
                    planes: None,
                    eye: found.map_or(Vec3::ZERO, |(center, _)| center).to_array(),
                    detail,
                    takes: Some(Mobility::Static),
                }
            })
            .collect(),
    }
}

fn publish_views(
    views: OwnedViews,
    published: &mut Published<'_, '_>,
    spawner: &mut Spawner<'_, '_, ReflectionProbesBrick>,
) {
    match published.iter_mut().find(|held| held.pass == views.pass) {
        Some(mut held) => {
            held.set_if_neq(views);
        }
        None => {
            spawner.spawn((ProbeViews, views));
        }
    }
}

fn place_cameras(
    jobs: &[FaceJob],
    pool: &mut Vec<Entity>,
    (spawner, moves, lenses): (
        &mut Spawner<'_, '_, ReflectionProbesBrick>,
        &mut Moves<'_, '_, ReflectionProbesBrick>,
        &mut Lenses<'_, '_>,
    ),
) {
    for face in 0..FACES {
        let captured = jobs
            .iter()
            .find(|job| job.known.face as usize == face)
            .map(|job| job.known);
        let (center, clip) = captured.map_or_else(
            || (Vec3::ZERO, face_view_projection(Vec3::ZERO, face)),
            |known| (known.center, known.view),
        );
        let Some(&node) = pool.get(face) else {
            pool.push(spawner.spawn((
                ProbeCamera,
                Node::default(),
                Transform::from_translation(center),
                Camera::fitted(clip, center),
                Mover {
                    by: mover_of::<ReflectionProbesBrick>(),
                    pace: Pace::Frame,
                },
            )));
            continue;
        };
        if captured.is_none() {
            continue;
        }
        if let Some(mut transform) = moves.get_mut(node)
            && transform.translation != center
        {
            transform.translation = center;
        }
        let origin = moves
            .local(node)
            .map_or(center, |transform| transform.translation);
        if let Ok(mut lens) = lenses.get_mut(node) {
            lens.set_if_neq(Camera::fitted(clip, origin));
        }
    }
}

pub(crate) fn publish_captures(
    (detail, mut state): (Res<Detail>, ResMut<ProbeState>),
    mut words: ResMut<ProbeWords>,
    mut captures: ResMut<ProbeCaptures>,
    mut stats: ResMut<ReflectionProbeStats>,
    mut published: Published,
    mut spawner: Spawner<ReflectionProbesBrick>,
    (mut moves, mut lenses): (Moves<ReflectionProbesBrick>, Lenses),
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
    place_cameras(
        &state.jobs,
        &mut state.cameras,
        (&mut spawner, &mut moves, &mut lenses),
    );
    let shape = (state.size, detail.chosen_error().unwrap_or(0.0));
    let jobs = &state.jobs;
    let drawn = face_views((ProbePass::NAME, shape), &state.cameras, |face| {
        jobs.iter()
            .find(|job| job.draws && job.known.face == face)
            .map(|job| (job.known.center, job.stored))
    });
    let lit = face_views((ProbeLightPass::NAME, shape), &state.cameras, |face| {
        jobs.iter()
            .find(|job| job.known.face == face)
            .map(|job| (job.known.center, job.lit))
    });
    publish_views(drawn, &mut published, &mut spawner);
    publish_views(lit, &mut published, &mut spawner);
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
