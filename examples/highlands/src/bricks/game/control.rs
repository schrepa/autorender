use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::{
    AssetLibrary, Camera, Error, ErrorCode, Expected, Jobs, Operation, ProvidedAssets,
    RecordExchange, Transform,
};
use engine_places::ground::Ground;
use engine_places::physics::RigidBody;
use glam::Vec2;
use serde::Deserialize;
use serde_json::Value;

use crate::bricks::game::bodies::Bodies;
use crate::bricks::game::cave::{Cave, CaveMouth};
use crate::bricks::game::content::{self, Content};
use crate::bricks::game::views::{Views, bodies_view};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum HighlandsControl {
    Land,
    Cave {
        mouth: [f32; 2],
        radius: f32,
        rock: String,
        wall: String,
    },
    Bodies {
        count: u32,
    },
    Stand {
        place: [f32; 4],
    },
    WatchBodies {
        on: bool,
    },
    Reset,
}

impl Operation for HighlandsControl {
    const NAME: &'static str = "highlands";
    type Params = (
        (
            ResMut<'static, AssetLibrary>,
            ResMut<'static, RecordExchange>,
            Option<ResMut<'static, ProvidedAssets>>,
            Jobs<'static, 'static>,
        ),
        (
            ResMut<'static, Content>,
            ResMut<'static, Cave>,
            ResMut<'static, Bodies>,
            ResMut<'static, Views>,
        ),
        (
            Option<Res<'static, Ground>>,
            Query<'static, 'static, &'static Transform, With<RigidBody>>,
            ResMut<'static, Camera>,
        ),
    );
    type Request = Self;

    fn run(
        (
            (mut assets, mut records, provided, mut jobs),
            (mut content, mut cave, mut bodies, mut views),
            (ground, rigid, mut camera),
        ): SystemParamItem<'_, '_, Self::Params>,
        control: Self,
    ) -> Result<Value, Error> {
        match control {
            Self::Land => {
                let mut provided = provided.ok_or_else(|| {
                    Error::new(ErrorCode::MissingResource, "$.provided").expected(Expected::Present)
                })?;
                content::provide_land(
                    (&mut *assets, &mut *records, &mut *provided),
                    (&mut *content, &mut jobs),
                )?;
            }
            Self::Cave {
                mouth,
                radius,
                rock,
                wall,
            } => {
                let material = |name: &str, path: &str| {
                    provided
                        .as_deref()
                        .and_then(|provided| provided.material(name))
                        .ok_or_else(|| {
                            Error::new(ErrorCode::GameRule, path)
                                .expected(Expected::Present)
                                .got(name)
                        })
                };
                cave.mouth = Some(CaveMouth {
                    center: Vec2::from(mouth),
                    radius,
                    material: material(&rock, "$.cave.rock")?,
                    wall: material(&wall, "$.cave.wall")?,
                });
            }
            Self::Bodies { count } => {
                if count > 0 {
                    let boulder = content::bank_boulder(&mut content, (&mut jobs, &mut *assets))?;
                    bodies.bank = Some((boulder, count));
                }
            }
            Self::Stand { place } => views.stand(ground.as_deref(), place, &mut camera),
            Self::WatchBodies { on } => {
                if let Some(view) = views.watch_bodies(on, bodies_view(&rigid)) {
                    *camera = view;
                }
            }
            Self::Reset => views.reset(),
        }
        Ok(Value::Null)
    }
}
