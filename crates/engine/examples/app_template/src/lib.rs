extern crate engine_ecs as bevy_ecs;

use engine::prelude::*;
use engine_fly_camera::FlyCameraBrick;

pub mod bricks {
    pub mod glimmer;
    pub mod turning;
}

use crate::bricks::turning::{Turning, TurningBrick};

const EYE: Vec3 = Vec3::new(24.0, 19.0, 28.0);

pub struct Template;

impl Game for Template {
    fn reset_camera(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        if let Some(camera) = main_camera(world) {
            let at = Transform::looking_at(EYE, Vec3::ZERO);
            place_camera(world, camera, at, Camera::default());
        }
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard_rendering::rendering(list)
        .with::<FlyCameraBrick>()
        .with::<TurningBrick>()
}

pub fn build(engine: &mut Engine) -> engine::Result<Template> {
    let world = engine.app.world_mut();
    engine::host::apply_setting(world, "dynamic_resolution.enabled", &true.into())?;
    let (mesh, ground, block) = {
        let mut assets = world.resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::cube())?,
            assets.insert_material(Material::new([0.35, 0.4, 0.45, 1.0])?)?,
            assets.insert_material(Material::new([0.9, 0.5, 0.2, 1.0])?)?,
        )
    };
    let spawned = world.spawn((
        Transform {
            translation: Vec3::new(0.0, -0.5, 0.0),
            scale: Vec3::new(20.0, 1.0, 20.0),
            ..Transform::default()
        },
    )).id();
    world.entity_mut(spawned).insert(Drawn::new(
        DrawKind::MESH,
        mesh,
        ground,
        spawned,
    ));
    let spawned = world.spawn((
        Transform {
            translation: Vec3::new(0.0, 1.0, 0.0),
            scale: Vec3::splat(2.0),
            ..Transform::default()
        },
        Turning,
    )).id();
    world.entity_mut(spawned).insert(Drawn::new(
        DrawKind::MESH,
        mesh,
        block,
        spawned,
    ));
    let camera = world
        .spawn((
            Transform::looking_at(EYE, Vec3::ZERO),
            Camera::default(),
            Node {
                clock: Some(Clock::Real),
                mobility: Some(Mobility::Moving),
                ..Node::default()
            },
        ))
        .id();
    world.spawn(View::new(camera));
    Ok(Template)
}

engine::game!(pub struct TemplateGame(Template), "Template", bricks, build);
