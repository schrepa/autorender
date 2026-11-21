use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::clamp;
use engine_core::{Camera, Drawn, GlobalTransforms, Projection, SceneContext, Stats, ViewSkip};
use engine_solids::Kind;

use super::{DeviceShown, DroppedWeapon, PropShown, PuppetBrick, PuppetStats};
use crate::names::CAMERA;
use crate::play::actors::Actor;
use crate::play::camera::TestCamera;
use crate::play::figures::{Form, Forms};
use crate::play::lobby::LobbyView;
use crate::play::puppet::{FormsTable, Puppet, Seen, ViewTest};
use crate::play::skinned::{Figure, FigureClock, FigureSight};
use crate::play::view::Watched;
use crate::play::world::Arena;

#[derive(Clone, Copy, Debug)]
struct Eye {
    at: Vec3,
    forward: Vec3,
    half: Option<f32>,
    vertical: Option<f32>,
}

#[derive(Default)]
pub(super) struct Memo {
    last: BTreeMap<Entity, f64>,
    forms: BTreeMap<Entity, Form>,
    eye: Option<Vec3>,
    judged: Vec<Entity>,
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
            Vec3::new(low.x, high.y, low.z),
            Vec3::new(high.x, high.y, low.z),
            Vec3::new(low.x, high.y, high.z),
            high,
            low,
            Vec3::new(high.x, low.y, low.z),
            Vec3::new(low.x, low.y, high.z),
            Vec3::new(high.x, low.y, high.z),
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
        match arena.bullet_ray(self.eye.at, toward / distance, distance) {
            None => true,
            Some(hit) => {
                hit.distance >= distance - 0.05
                    || self.table.see_through.iter().any(|name| Kind::named(name) == hit.kind)
            }
        }
    }

    fn open(&self, bounds: Bounds, exempt: bool) -> bool {
        if exempt || !self.table.map {
            return true;
        }
        let Some(arena) = self.arena.filter(|arena| arena.ready()) else {
            return true;
        };
        bounds
            .padded(self.table.map_pad_m)
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

type Lenses<'w, 's> = Query<'w, 's, (&'static Camera, &'static Transform)>;

fn eye_of(scene: Option<&SceneContext>, cameras: &Lenses, aspect_max: f32) -> Option<Eye> {
    let camera = scene?.entities.get(CAMERA).copied()?;
    let (camera, placed) = cameras.get(camera).ok()?;
    let vertical = match camera.projection {
        Projection::Perspective {
            vertical_fov_radians,
            ..
        } => Some(vertical_fov_radians),
        Projection::Orthographic { .. } | Projection::Fitted { .. } => None,
    };
    let across = (1.0 + aspect_max * aspect_max).sqrt();
    Some(Eye {
        at: placed.translation,
        forward: (placed.rotation * Vec3::NEG_Z).normalize_or_zero(),
        half: vertical.map(|vertical| ((vertical * 0.5).tan() * across).atan()),
        vertical,
    })
}

fn fixed_bounds(clock: Option<&FigureClock>, margin: f32) -> Option<Bounds> {
    let [low, high] = clock?.bounds?;
    Some(Bounds { low, high }.padded(margin))
}

type Actors<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        (&'static Actor, Option<&'static Forms>),
        Option<&'static FigureClock>,
        &'static Node,
        &'static Drawn,
    ),
    With<Figure>,
>;

type Items<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Node),
    Or<(With<DroppedWeapon>, With<PropShown>, With<DeviceShown>)>,
>;

type Reads<'w, 's> = (
    (Res<'w, FrameTime>, Res<'w, GlobalTransforms>),
    (Option<Res<'w, SceneContext>>, Lenses<'w, 's>),
    (Option<Res<'w, Arena>>, Res<'w, Puppet>),
    (
        Option<Res<'w, TestCamera>>,
        Option<Res<'w, Watched>>,
        Query<'w, 's, (), With<ViewSkip>>,
        Option<Res<'w, LobbyView>>,
    ),
);

fn shown(spawner: &mut Spawner<PuppetBrick>, (entity, node): (Entity, &Node), visible: bool) {
    if node.visible != visible {
        spawner.insert(entity, Node { visible, ..*node });
    }
}

pub(super) fn cull(
    ((time, globals), (scene, cameras), (arena, puppet), (test, watched, skipped, lobby)): Reads,
    (actors, items): (Actors, Items),
    (mut seen, mut memo, mut stats): (ResMut<Seen>, Local<Memo>, Stats<PuppetStats>),
    (mut sight, mut spawner): (ResMut<FigureSight>, Spawner<PuppetBrick>),
) {
    let Memo {
        last,
        forms,
        eye: was_at,
        judged: judged_actors,
    } = &mut *memo;
    let Some(table) = puppet.table.as_deref() else {
        return;
    };
    if !skipped.is_empty() {
        return;
    }
    let forms_table = &table.forms;
    let table = &table.view_test;
    let outside = test.as_deref().is_some_and(|test| test.outside);
    let followed = watched.as_deref().and_then(|watched| watched.entity);
    let own = followed.filter(|_| !outside);
    let lobby = !outside && lobby.as_deref().is_some_and(|lobby| lobby.camera.is_some());
    stats.edit(|stats| stats.lobby = lobby);
    let Some(eye) = eye_of(scene.as_deref(), &cameras, table.aspect_max) else {
        return;
    };
    let tests = Tests {
        eye,
        table,
        arena: arena.as_deref(),
    };
    let (frame, now, hold) = (time.frame, time.real_elapsed, f64::from(table.hold_seconds));
    if was_at.is_some_and(|at| at.distance(eye.at) > table.cut_m) {
        last.clear();
    }
    *was_at = Some(eye.at);
    let mut fresh = Seen::default();
    let mut judged = |entity: Entity, bounds: Option<Bounds>, exempt: bool| {
        let verdict = bounds.map_or(Verdict::Shown, |bounds| tests.judge(bounds, exempt));
        if verdict == Verdict::Shown {
            last.insert(entity, now);
        }
        let kept = verdict == Verdict::BehindMap
            && last.get(&entity).is_some_and(|at| now - at <= hold);
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
    let scale = form_scale(&eye, forms_table);
    judged_actors.clear();
    let mut formless: Vec<String> = Vec::new();
    for (entity, (actor, figure_forms), clock, node, drawn) in &actors {
        let revealed = drawn.values.emission != [0.0; 3];
        let exempt = revealed || followed == Some(entity);
        let bounds = fixed_bounds(clock, table.mesh_margin_m);
        let judgement = judged(entity, bounds, exempt);
        let visible = !lobby && counted(&mut fresh, judgement) && own != Some(entity);
        if figure_forms.is_none() && !formless.contains(&actor.character) {
            formless.push(actor.character.clone());
        }
        let distance = bounds.map_or(0.0, |bounds| (bounds.centre() - eye.at).length());
        let was = forms.get(&entity).copied().unwrap_or_default();
        let form = if figure_forms.is_none() {
            Form::Full
        } else {
            next_form(was, distance, forms_table, scale)
        };
        forms.insert(entity, form);
        judged_actors.push(entity);
        if visible {
            fresh.shown.push((entity, form, distance));
        }
        shown(&mut spawner, (entity, node), visible);
    }
    formless.sort_unstable();
    stats.edit(|stats| stats.formless = formless);
    for (entity, node) in &items {
        let bounds = globals
            .get(entity)
            .and_then(|global| global.matrix())
            .map(|matrix| {
                let at = matrix.w_axis.truncate();
                Bounds { low: at, high: at }.padded(table.item_radius_m)
            });
        let judgement = judged(entity, bounds, false);
        let visible = !lobby && counted(&mut fresh, judgement);
        shown(&mut spawner, (entity, node), visible);
    }
    let alive = |entity: &Entity| actors.contains(*entity) || items.contains(*entity);
    last.retain(|entity, _| alive(entity));
    forms.retain(|entity, _| alive(entity));
    fresh.shown.sort_unstable_by_key(|(entity, ..)| *entity);
    judged_actors.sort_unstable();
    let shown_same = sight.shown.iter().eq(fresh.shown.iter().map(|(entity, ..)| entity));
    if sight.judged != *judged_actors || !shown_same {
        sight.judged.clone_from(judged_actors);
        sight.shown.clear();
        sight.shown.extend(fresh.shown.iter().map(|(entity, ..)| *entity));
    }
    sight.bypass_change_detection().frame = frame;
    fresh.frame = seen.frame;
    seen.set_if_neq(fresh);
    seen.bypass_change_detection().frame = frame;
}
