use engine::math;
use engine::prelude::*;
use engine_core::{Assets, DevSetting, Material, MeshFile, Mobility};
use glam::{Quat, Vec3};

use super::gather::{Setup, view_direction};
use super::numbers::{ShowcaseLook, TrapModels};
use super::{EffectsBrick, EffectsShow, Show};
use crate::play::actors::{Body, Human, Wish};
use crate::play::figures::GROUND;

#[derive(Component, Clone, Debug, PartialEq)]
pub(super) struct ShowcaseModel {
    name: String,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(super) struct ShowcaseModels {
    pub shown: Option<String>,
    pub traps: bool,
}

pub(super) struct TrapsLayout {
    pub trap: Vec3,
    pub sensor: Vec3,
    pub wire: (Vec3, Vec3),
    pub projectile: Vec3,
}

pub(super) fn traps_layout(body: &Body, wish: &Wish, case: &ShowcaseLook) -> TrapsLayout {
    let toward = view_direction(wish);
    let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::NEG_Z);
    let side = flat.cross(Vec3::Y);
    let near = body.feet + flat * case.model_m;
    TrapsLayout {
        trap: near - side,
        sensor: near,
        wire: (near + side, near + side * 3.0),
        projectile: near + side * 0.5 + Vec3::Y * 1.2,
    }
}

type Reads<'w, 's> = (
    (
        Res<'w, FixedTime>,
        DevSetting<'w, EffectsShow>,
        Res<'w, Setup>,
    ),
    Res<'w, Assets>,
    Query<'w, 's, (&'static Body, &'static Wish), With<Human>>,
    Query<'w, 's, (Entity, &'static ShowcaseModel)>,
);

type Writes<'w, 's> = (
    ResMut<'w, ShowcaseModels>,
    Spawner<'w, 's, EffectsBrick>,
    Moves<'w, 's, EffectsBrick>,
);

fn drawn(assets: &Assets, name: &str) -> Option<Drawn> {
    let mesh = assets.named::<MeshFile>(name)?;
    let material = assets.named::<Material>(name)?;
    Some(Drawn::new(GROUND, mesh, material))
}

fn cycled(assets: &Assets, prefix: &str, step: u64) -> Option<String> {
    let mut names: Vec<&str> = assets
        .each::<MeshFile>()
        .filter_map(|(mesh, _)| assets.name(mesh))
        .filter(|name| name.starts_with(prefix) && assets.named::<Material>(name).is_some())
        .collect();
    names.sort_unstable();
    let at = usize::try_from(step).ok()? % names.len().max(1);
    names.get(at).map(|name| (*name).to_owned())
}

fn wanted(
    (show, case, now): (Show, &ShowcaseLook, f64),
    (body, wish): (&Body, &Wish),
    assets: &Assets,
) -> Vec<(String, Vec3)> {
    let models: &TrapModels = &case.trap_models;
    if show == Show::Models {
        let every = f64::from(math::max(case.every_seconds, 0.1)) * 2.0;
        let toward = view_direction(wish);
        let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::NEG_Z);
        cycled(assets, &case.model_prefix, (now / every) as u64)
            .map(|name| vec![(name, body.feet + flat * case.model_m)])
            .unwrap_or_default()
    } else if show == Show::Traps {
        let at = traps_layout(body, wish, case);
        vec![
            (models.trap.clone(), at.trap),
            (models.sensor.clone(), at.sensor),
            (models.wire.clone(), at.wire.0),
            (models.wire.clone(), at.wire.1),
            (models.projectile.clone(), at.projectile),
        ]
    } else {
        Vec::new()
    }
}

pub(super) fn showcase_models(
    ((fixed, show, setup), assets, humans, shown): Reads,
    (mut models, mut spawner, mut moves): Writes,
) {
    let show = show.get().map_or(Show::Off, |show| show.show);
    let case = setup.parsed.as_ref().map(|parsed| &parsed.look.showcase);
    let human = humans.iter().next();
    let (Some(case), Some((body, wish))) = (case, human) else {
        for (entity, _) in &shown {
            spawner.despawn(entity);
        }
        models.set_if_neq(ShowcaseModels::default());
        return;
    };
    let toward = view_direction(wish);
    let flat = Vec3::new(toward.x, 0.0, toward.z).normalize_or(Vec3::NEG_Z);
    let facing = Quat::from_rotation_arc(Vec3::Z, -flat);
    let list: Vec<(String, Drawn, Transform)> =
        wanted((show, case, fixed.elapsed_seconds), (body, wish), &assets)
            .into_iter()
            .filter_map(|(name, at)| {
                let drawn = drawn(&assets, &name)?;
                Some((
                    name,
                    drawn,
                    Transform::from_translation(at).with_rotation(facing),
                ))
            })
            .collect();
    let mut placed = vec![false; list.len()];
    for (entity, model) in &shown {
        let free = list
            .iter()
            .enumerate()
            .find(|(at, (name, ..))| !placed[*at] && *name == model.name);
        match free {
            Some((at, (_, _, transform))) => {
                placed[at] = true;
                if let Some(mut moved) = moves.get_mut(entity) {
                    moved.set_if_neq(*transform);
                }
            }
            None => spawner.despawn(entity),
        }
    }
    let node = Node {
        visible: true,
        mobility: Some(Mobility::Moving),
        ..Node::default()
    };
    for ((name, drawn, transform), done) in list.iter().zip(&placed) {
        if !done {
            let model = ShowcaseModel { name: name.clone() };
            spawner.spawn((*transform, node, drawn.clone(), model));
        }
    }
    models.set_if_neq(ShowcaseModels {
        shown: (show == Show::Models)
            .then(|| list.first().map(|(name, ..)| name.clone()))
            .flatten(),
        traps: show == Show::Traps && list.len() == 5,
    });
}
