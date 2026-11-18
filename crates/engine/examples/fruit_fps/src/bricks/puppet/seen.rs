use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::clamp;
use engine_core::{Camera, Drawn, GlobalTransforms, Projection, SceneContext};

use super::{DeviceShown, DroppedWeapon, PropShown, PuppetBrick, PuppetStats};
use crate::names::CAMERA;
use crate::play::actors::Actor;
use crate::play::camera::TestCamera;
use crate::play::figures::{Figure, FigureJoints, Form};
use crate::play::puppet::{FormsTable, PuppetFile, Seen, ViewTest};
use crate::play::screen::Covered;
use crate::play::view::Watched;
use crate::play::world::Arena;
use crate::solids::Kind;

#[derive(Clone, Copy, Debug)]
struct Eye {
    at: Vec3,
    forward: Vec3,
    half: Option<f32>,
    vertical: Option<f32>,
}

#[derive(Default)]
pub(super) struct Memo {
    last: BTreeMap<Entity, u64>,
    forms: BTreeMap<Entity, Form>,
    eye: Option<Vec3>,
}

fn next_form(was: Form, distance: f32, table: &FormsTable, scale: f32) -> Form {
    let full_to = table.full_to_m * scale;
    let mid_from = table.mid_from_m * scale;
    let mid_back = table.mid_back_m * scale;
    let far_from = table.far_from_m * scale;
    match was {
        Form::Full if distance > far_from => Form::Far,
        Form::Full if distance > mid_from => Form::Mid,
        Form::Mid if distance > far_from => Form::Far,
        Form::Mid | Form::Far if distance < full_to => Form::Full,
        Form::Far if distance < mid_back => Form::Mid,
        Form::Full | Form::Mid | Form::Far => was,
    }
}

fn form_scale(eye: &Eye, table: &FormsTable) -> f32 {
    let reference = (table.reference_fov_degrees.to_radians() * 0.5).tan();
    eye.vertical
        .map(|vertical| (vertical * 0.5).tan())
        .filter(|half| *half > 0.0)
        .map_or(1.0, |half| reference / half)
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    low: Vec3,
    high: Vec3,
}

impl Bounds {
    fn padded(self, pad: f32) -> Self {
        Self {
            low: self.low - Vec3::splat(pad),
            high: self.high + Vec3::splat(pad),
        }
    }

    fn centre(self) -> Vec3 {
        (self.low + self.high) * 0.5
    }

    fn points(self) -> [Vec3; 9] {
        let (low, high) = (self.low, self.high);
        [
            self.centre(),
            low,
            Vec3::new(high.x, low.y, low.z),
            Vec3::new(low.x, high.y, low.z),
            Vec3::new(high.x, high.y, low.z),
            Vec3::new(low.x, low.y, high.z),
            Vec3::new(high.x, low.y, high.z),
            Vec3::new(low.x, high.y, high.z),
            high,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Shown,
    OutOfView,
    BehindMap,
}

struct Tests<'a> {
    eye: Eye,
    table: &'a ViewTest,
    arena: Option<&'a Arena>,
    clear: Vec<Kind>,
}

impl Tests<'_> {
    fn in_view(&self, bounds: Bounds) -> bool {
        let Some(half) = self.eye.half.filter(|_| self.table.frustum) else {
            return true;
        };
        let toward = bounds.centre() - self.eye.at;
        let distance = toward.length();
        let radius = (bounds.high - bounds.low).length() * 0.5;
        if distance <= radius {
            return true;
        }
        let angle = clamp((toward / distance).dot(self.eye.forward), -1.0, 1.0).acos();
        angle <= half + (radius / distance).asin()
    }

    fn reached(&self, arena: &Arena, point: Vec3) -> bool {
        let toward = point - self.eye.at;
        let distance = toward.length();
        if distance <= 1.0e-3 {
            return true;
        }
        match arena.solids().ray(self.eye.at, toward / distance, distance) {
            None => true,
            Some(hit) => hit.distance >= distance - 0.05 || self.clear.contains(&hit.kind),
        }
    }

    fn open(&self, bounds: Bounds, exempt: bool) -> bool {
        if exempt || !self.table.map {
            return true;
        }
        let Some(arena) = self.arena.filter(|arena| arena.ready()) else {
            return true;
        };
        let padded = bounds.padded(self.table.map_pad_m);
        padded
            .points()
            .into_iter()
            .any(|point| self.reached(arena, point))
    }

    fn judge(&self, bounds: Bounds, exempt: bool) -> Verdict {
        if !self.in_view(bounds) {
            Verdict::OutOfView
        } else if !self.open(bounds, exempt) {
            Verdict::BehindMap
        } else {
            Verdict::Shown
        }
    }
}

fn eye_of(
    (scene, globals): (Option<&SceneContext>, &GlobalTransforms),
    cameras: &Query<&Camera>,
    aspect_max: f32,
) -> Option<Eye> {
    let camera = scene?.entities.get(CAMERA).copied()?;
    let matrix = globals.get(camera)?.matrix()?;
    let vertical = match cameras.get(camera).ok()?.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => Some(vertical_fov_radians),
        Projection::Orthographic { .. } | Projection::Fitted { .. } => None,
    };
    let across = (1.0 + aspect_max * aspect_max).sqrt();
    Some(Eye {
        at: matrix.w_axis.truncate(),
        forward: (-matrix.z_axis.truncate()).normalize_or_zero(),
        half: vertical.map(|vertical| ((vertical * 0.5).tan() * across).atan()),
        vertical,
    })
}

fn joint_bounds(joints: Option<&FigureJoints>, margin: f32) -> Option<Bounds> {
    let joints = joints?;
    let mut points = joints
        .joints
        .iter()
        .filter(|joint| joint.found)
        .map(|joint| joint.position)
        .chain(joints.head_top);
    let first = points.next()?;
    let (low, high) = points.fold((first, first), |(low, high), at| {
        (low.min(at), high.max(at))
    });
    Some(Bounds { low, high }.padded(margin))
}

type Actors<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        (&'static Actor, &'static Figure),
        Option<&'static FigureJoints>,
        &'static Node,
        &'static Drawn,
    ),
>;

type Items<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Node),
    Or<(With<DroppedWeapon>, With<PropShown>, With<DeviceShown>)>,
>;

type Reads<'w, 's> = (
    (Res<'w, FrameTime>, Res<'w, GlobalTransforms>),
    (
        Option<Res<'w, SceneContext>>,
        Query<'w, 's, &'static Camera>,
    ),
    (Option<Res<'w, Arena>>, Query<'w, 's, &'static PuppetFile>),
    (
        Option<Res<'w, TestCamera>>,
        Option<Res<'w, Watched>>,
        Option<Res<'w, Covered>>,
    ),
);

fn shown(spawner: &mut Spawner<PuppetBrick>, (entity, node): (Entity, &Node), visible: bool) {
    if node.visible != visible {
        spawner.insert(entity, Node { visible, ..*node });
    }
}

pub(super) fn cull(
    ((time, globals), (scene, cameras), (arena, files), (test, watched, covered)): Reads,
    (actors, items): (Actors, Items),
    (mut seen, mut memo, mut stats): (ResMut<Seen>, Local<Memo>, ResMut<PuppetStats>),
    mut spawner: Spawner<PuppetBrick>,
) {
    let Memo {
        last,
        forms,
        eye: was_at,
    } = &mut *memo;
    let Some(book) = files.iter().find_map(PuppetFile::book) else {
        return;
    };
    if covered.as_deref().is_some_and(|covered| covered.0) {
        return;
    }
    let table = &book.table.view_test;
    let outside = test.as_deref().is_some_and(|test| test.outside);
    let followed = watched.as_deref().and_then(|watched| watched.entity);
    let own = followed.filter(|_| !outside);
    let Some(eye) = eye_of((scene.as_deref(), &globals), &cameras, table.aspect_max) else {
        return;
    };
    let tests = Tests {
        eye,
        table,
        arena: arena.as_deref(),
        clear: table
            .see_through
            .iter()
            .map(|name| Kind::named(name))
            .collect(),
    };
    let now = time.frame;
    if was_at.is_some_and(|at| at.distance(eye.at) > table.cut_m) {
        last.clear();
    }
    *was_at = Some(eye.at);
    let mut fresh = Seen {
        frame: now,
        ..Seen::default()
    };
    let mut judged = |entity: Entity, bounds: Option<Bounds>, exempt: bool| {
        let verdict = bounds.map_or(Verdict::Shown, |bounds| tests.judge(bounds, exempt));
        if verdict == Verdict::Shown {
            last.insert(entity, now);
        }
        let kept = verdict == Verdict::BehindMap
            && last
                .get(&entity)
                .is_some_and(|at| now - at <= table.hold_frames);
        (verdict, verdict == Verdict::Shown || kept)
    };
    let counted = |fresh: &mut Seen, (verdict, visible): (Verdict, bool)| {
        match (verdict, visible) {
            (Verdict::OutOfView, _) => fresh.frustum += 1,
            (Verdict::BehindMap, false) => fresh.map += 1,
            (Verdict::BehindMap, true) => fresh.held += 1,
            (Verdict::Shown, _) => {}
        }
        visible
    };
    let scale = form_scale(&eye, &book.table.forms);
    let mut formless: Vec<String> = Vec::new();
    for (entity, (actor, figure), joints, node, drawn) in &actors {
        let revealed = drawn.values.emission != [0.0; 3];
        let exempt = revealed || followed == Some(entity);
        let bounds = joint_bounds(joints, table.mesh_margin_m);
        let judgement = judged(entity, bounds, exempt);
        let visible = counted(&mut fresh, judgement) && own != Some(entity);
        if figure.forms.is_none() && !formless.contains(&actor.character) {
            formless.push(actor.character.clone());
        }
        let distance = bounds.map_or(0.0, |bounds| (bounds.centre() - eye.at).length());
        let was = forms.get(&entity).copied().unwrap_or_default();
        let form = if figure.forms.is_none() {
            Form::Full
        } else {
            next_form(was, distance, &book.table.forms, scale)
        };
        forms.insert(entity, form);
        fresh.judged.push(entity);
        if visible {
            fresh.shown.push((entity, form, distance));
        }
        shown(&mut spawner, (entity, node), visible);
    }
    formless.sort_unstable();
    if stats.formless != formless {
        stats.formless = formless;
    }
    for (entity, node) in &items {
        let bounds = globals
            .get(entity)
            .and_then(|global| global.matrix())
            .map(|matrix| {
                let at = matrix.w_axis.truncate();
                Bounds { low: at, high: at }.padded(table.item_radius_m)
            });
        let judgement = judged(entity, bounds, false);
        let visible = counted(&mut fresh, judgement);
        shown(&mut spawner, (entity, node), visible);
    }
    let alive: Vec<Entity> = actors
        .iter()
        .map(|(entity, ..)| entity)
        .chain(items.iter().map(|(entity, _)| entity))
        .collect();
    last.retain(|entity, _| alive.contains(entity));
    forms.retain(|entity, _| alive.contains(entity));
    fresh.shown.sort_unstable_by_key(|(entity, ..)| *entity);
    fresh.judged.sort_unstable();
    if *seen != fresh {
        *seen = fresh;
    }
}
