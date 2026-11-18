use engine::prelude::*;
use engine_core::math::sin_cos;
use engine_core::{Error, ErrorCode, Refusals, SceneContext};
use glam::{Quat, Vec3};

use super::drops::{drawn_of, ground_node, lying_at};
use super::{DeviceShown, PuppetBrick};
use crate::play::puppet::{DEVICE, DeviceLamp, PuppetBook, PuppetFile};
use crate::play::round::{DeviceSpot, Round};

const DEVICE_PART: &str = "puppet_device";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Form {
    Dropped(Vec3),
    Planted(Vec3),
}

fn form_of(spot: DeviceSpot) -> Option<Form> {
    match spot {
        DeviceSpot::Dropped { at } => Some(Form::Dropped(at)),
        DeviceSpot::Planted { at, .. } | DeviceSpot::Defused { at, .. } => Some(Form::Planted(at)),
        DeviceSpot::Nowhere | DeviceSpot::Carried { .. } | DeviceSpot::Exploded { .. } => None,
    }
}

fn yaw_of(spot: DeviceSpot) -> f32 {
    match spot {
        DeviceSpot::Planted { yaw, .. }
        | DeviceSpot::Defused { yaw, .. }
        | DeviceSpot::Exploded { yaw, .. } => yaw,
        DeviceSpot::Nowhere | DeviceSpot::Carried { .. } | DeviceSpot::Dropped { .. } => 0.0,
    }
}

fn turned(yaw: f32) -> Quat {
    let (sin, cos) = sin_cos((yaw + std::f32::consts::PI) * 0.5);
    Quat::from_xyzw(0.0, sin, 0.0, cos)
}

fn shown_as(
    book: &PuppetBook,
    scene: Option<&SceneContext>,
    (form, yaw): (Form, f32),
) -> Result<(Transform, Drawn), Error> {
    match form {
        Form::Dropped(at) => {
            let look = book.weapons.get(DEVICE).ok_or_else(|| {
                Error::new(ErrorCode::UnknownReference, "$.puppet.weapons").got(DEVICE)
            })?;
            let drawn = drawn_of(scene, (&look.files.held_mesh, &look.files.held_material))?;
            Ok((lying_at(book, at, 0, 1.0), drawn))
        }
        Form::Planted(at) => {
            let planted = book.planted.as_ref().ok_or_else(|| {
                Error::new(ErrorCode::UnknownReference, "$.puppet.planted").got("absent")
            })?;
            let drawn = drawn_of(scene, (&planted.files.mesh, &planted.files.material))?;
            let placed = Transform {
                translation: at,
                rotation: turned(yaw),
                scale: Vec3::ONE,
            };
            Ok((placed, drawn))
        }
    }
}

type Reads<'w, 's> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, (Entity, &'static PuppetFile)>,
);

pub(super) fn place(
    (round, scene, files): Reads,
    shown: Query<(Entity, &DeviceShown)>,
    (mut lamp, mut refusals, mut spawner): (
        ResMut<DeviceLamp>,
        ResMut<Refusals>,
        Spawner<PuppetBrick>,
    ),
) {
    let Some((owner, book)) = files
        .iter()
        .find_map(|(entity, file)| file.book().map(|book| (entity, book)))
    else {
        return;
    };
    let form = round.as_deref().and_then(|round| form_of(round.device));
    let yaw = round.as_deref().map_or(0.0, |round| yaw_of(round.device));
    let light = match form {
        Some(Form::Planted(at)) => book
            .planted
            .as_ref()
            .and_then(|planted| planted.markers.light)
            .map(|light| at + turned(yaw) * light),
        Some(Form::Dropped(at)) => book
            .weapons
            .get(DEVICE)
            .and_then(|look| look.markers.light)
            .map(|light| {
                let lying = lying_at(book, at, 0, 1.0);
                lying.translation + lying.rotation * light
            }),
        None => None,
    };
    lamp.set_if_neq(DeviceLamp { position: light });
    let mut kept = false;
    for (entity, device) in &shown {
        if Some(device.form) == form && !kept {
            kept = true;
        } else {
            spawner.despawn(entity);
        }
    }
    let verdict = match form {
        Some(form) if !kept => {
            shown_as(book, scene.as_deref(), (form, yaw)).map(|(placed, drawn)| {
                spawner.spawn((placed, ground_node(), drawn, DeviceShown { form }));
            })
        }
        Some(_) | None => Ok(()),
    };
    refusals.judge(owner, DEVICE_PART, verdict);
}
