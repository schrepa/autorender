extern crate engine_ecs as bevy_ecs;

use engine::prelude::*;
use engine_fly_camera::FlyCameraBrick;

pub mod bricks {
    pub mod turning;
}

use crate::bricks::turning::{Turning, TurningBrick};

pub struct Template;

impl Game for Template {
    fn reset_camera(&mut self, engine: &mut Engine) {
        engine.app.world_mut().insert_resource(Camera::default());
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::standard(list)
        .with::<FlyCameraBrick>()
        .with::<TurningBrick>()
}

const DAY: [&str; 4] = [
    "sun.preset",
    "sky_light.preset",
    "surface_light.preset",
    "grading.preset",
];

pub fn build(engine: &mut Engine) -> engine::Result<Template> {
    let world = engine.app.world_mut();
    for key in DAY {
        engine::host::apply_setting(world, key, &serde_json::json!("day"))?;
    }
    let (mesh, ground, block) = {
        let mut assets = world.resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::cube())?,
            assets.insert_material(Material::new([0.35, 0.4, 0.45, 1.0])?)?,
            assets.insert_material(Material::new([0.9, 0.5, 0.2, 1.0])?)?,
        )
    };
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, -0.5, 0.0),
            scale: Vec3::new(20.0, 1.0, 20.0),
            ..Transform::default()
        },
        Mesh3d {
            mesh,
            material: ground,
        },
    ));
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, 1.0, 0.0),
            scale: Vec3::splat(2.0),
            ..Transform::default()
        },
        Mesh3d {
            mesh,
            material: block,
        },
        InterpolateTransform::default(),
        Turning,
    ));
    Ok(Template)
}

engine::export_game!(
    pub struct TemplateGame(Template);
);

engine_worker::export_worker!(bricks);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl TemplateGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<TemplateGame, wasm_bindgen::JsValue> {
        Self::launch(canvas, bricks, build).await
    }
}
