//! The metro's keyframed props: each prop's moving parts under its own root
//! at a site, bound by a player to its model's clip.

use std::f32::consts::PI;

use engine::prelude::*;

use super::materials::{AD_SLOTS, Clip, Mat};
use super::models::Model;
use super::{
    Anchor, CANVAS, CRANE, Extra, Metro, Part, Prop, Site, WHITE, at, look, piece, reparent, v,
};
use crate::kit::plan::compose;
use crate::kit::random::Rng;

/// The moving parts of a keyframed prop at `site`, under its own root.
/// Every animated model turns about its build-space origin.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(super) fn animated_prop(city: &mut Metro, rng: &mut Rng, prop: Prop, site: &Site) {
    let frame = Transform {
        translation: site.at,
        rotation: Quat::from_rotation_y(site.yaw),
        scale: Vec3::ONE,
    };
    let (clip, parts): (Clip, Vec<Part>) = match prop {
        Prop::Flag => (
            Clip::Flag,
            vec![piece(
                Vec3::ZERO,
                look(Model::Flag, Mat::Canvas, rng.pick(&CANVAS)),
            )],
        ),
        Prop::Dish => (
            Clip::Dish,
            vec![piece(
                Vec3::ZERO,
                look(Model::DishHead, Mat::Fixture, WHITE),
            )],
        ),
        Prop::Trivision => {
            let slot = rng.below(u32::from(AD_SLOTS)) as u8;
            (
                Clip::Trivision,
                vec![piece(
                    Vec3::ZERO,
                    look(Model::Trivision, Mat::Ad(slot), WHITE),
                )],
            )
        }
        Prop::Door => (
            Clip::Door,
            vec![piece(
                Vec3::ZERO,
                look(Model::DoorWings, Mat::ClearGlass, WHITE),
            )],
        ),
        Prop::Turbine => {
            // The tower is static; its rotor turns at the nacelle.
            let tower = Model::TurbineTower;
            let height = city.size(tower).y * site.size;
            let stand = Transform {
                translation: Vec3::Y * (height * 0.5),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(site.size),
            };
            city.part(
                &site.anchor,
                compose(&frame, &stand),
                look(tower, Mat::Fixture, WHITE),
            );
            let mut rotor = piece(
                v(1.9, 24.5, 0.0) * site.size,
                look(Model::TurbineRotor, Mat::Fixture, WHITE),
            );
            rotor.scale = Vec3::splat(site.size);
            (Clip::Turbine, vec![rotor])
        }
        Prop::Excavator => (
            Clip::Excavator,
            vec![
                piece(Vec3::ZERO, look(Model::ExcavatorBody, Mat::Vehicle, CRANE)),
                piece(
                    v(1.0, 1.6, -0.4),
                    look(Model::ExcavatorBoom, Mat::Vehicle, CRANE),
                ),
                piece(
                    v(4.5, 3.1, -0.4),
                    look(Model::ExcavatorStick, Mat::Vehicle, CRANE),
                ),
            ],
        ),
        Prop::Carousel => {
            let mut top = piece(
                Vec3::ZERO,
                look(Model::CarouselTop, Mat::Canvas, rng.pick(&CANVAS)),
            );
            top.scale = Vec3::splat(site.size);
            (Clip::Carousel, vec![top])
        }
        Prop::Drawbridge => {
            let mut east = piece(v(10.0, 0.0, 0.0), look(Model::BridgeLeaf, Mat::Road, WHITE));
            east.rotation = Quat::from_rotation_y(PI);
            (
                Clip::Drawbridge,
                vec![
                    piece(
                        v(-10.0, 0.0, 0.0),
                        look(Model::BridgeLeaf, Mat::Road, WHITE),
                    ),
                    east,
                ],
            )
        }
        Prop::Ring => (
            Clip::Ring,
            vec![piece(
                Vec3::ZERO,
                look(Model::TowerRing, Mat::Chrome, WHITE),
            )],
        ),
        Prop::Gondola(class) => {
            // The clip moves the pivot up and down the face.
            let mut gondola = piece(Vec3::ZERO, look(Model::Gondola, Mat::Fixture, WHITE));
            gondola.shift = v(0.0, 0.0, 0.05);
            (Clip::Gondola(class), vec![gondola])
        }
    };
    if city.moving_left < parts.len() as u32 {
        return;
    }
    let root = city.model_root(&site.anchor, frame, None);
    let mut targets = Vec::with_capacity(parts.len());
    for part in parts {
        let pivot = city.push(&root, at(part.pivot), None, None, false);
        city.extras.push((pivot, Extra::Animated));
        let center = city.shape(part.look.model).1;
        let rest = Transform {
            translation: part.shift + part.rotation * (center * part.scale),
            rotation: part.rotation,
            scale: part.scale,
        };
        city.moving_left -= 1;
        city.push(
            &Anchor {
                node: Some(pivot),
                frame: Transform::default(),
            },
            rest,
            Some(part.look),
            None,
            true,
        );
        targets.push(pivot);
    }
    // The excavator's stick hangs from its boom, the boom from the body.
    if matches!(prop, Prop::Excavator) {
        let (body, boom, stick) = (targets[0], targets[1], targets[2]);
        reparent(city, boom, body);
        reparent(city, stick, boom);
    }
    let speed = rng.range(0.85, 1.15);
    city.extra(
        root.node,
        Extra::Player {
            clip,
            targets,
            time: rng.range(0.0, 60.0),
            speed,
        },
    );
}
