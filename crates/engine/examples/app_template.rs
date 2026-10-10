//! A new game: a cube turning on a slab under the fly camera. `cargo xtask
//! new app NAME` copies this file as the app's `src/lib.rs`; lint compiles
//! it for the browser and natively, so every new app starts from code that
//! builds.

use engine::prelude::*;

/// What the game keeps beside its engine.
pub struct Template;

impl Game for Template {
    fn reset_camera(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        let camera = Camera::default();
        if let Some(mut controller) = world.get_resource_mut::<FlyCameraController>() {
            controller.align_to(&camera);
        }
        world.insert_resource(camera);
    }
}

/// The cube that turns.
#[derive(Component)]
struct Turning;

/// Turns the cube a little each fixed tick.
fn turn(time: Res<FixedTime>, mut cubes: Query<&mut Transform, With<Turning>>) {
    for mut transform in &mut cubes {
        transform.rotation = Quat::from_rotation_y(time.delta_seconds) * transform.rotation;
    }
}

/// The game's own systems, as a plugin like the engine's features: it
/// turns the cube each tick after the features moved and posed things.
struct TemplatePlugin;

impl Plugin for TemplatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedSimulation, turn.after(FixedSimulationSystems::Pose));
    }
}

/// The game's engine and state.
pub fn build() -> engine::Result<(Engine, Template)> {
    let mut engine = Engine::default();
    // Every engine feature (or `EnginePlugins::without([...])`), the fly
    // camera and the game's own.
    engine
        .app
        .add_plugins((EnginePlugins, FlyCameraPlugin, TemplatePlugin))
        // The look is data: presets, or a scene's `environment`.
        .insert_resource(Atmosphere::day());
    let world = engine.app.world_mut();
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, -0.5, 0.0),
            scale: Vec3::new(20.0, 1.0, 20.0),
            ..Transform::default()
        },
        Cube {
            color: [0.35, 0.4, 0.45, 1.0],
        },
    ));
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, 1.0, 0.0),
            scale: Vec3::splat(2.0),
            ..Transform::default()
        },
        Cube {
            color: [0.9, 0.5, 0.2, 1.0],
        },
        InterpolateTransform,
        Turning,
    ));
    Ok((engine, Template))
}

engine::export_game!(
    /// The game, as the page runs it.
    pub struct TemplateGame(Template);
);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl TemplateGame {
    /// The game on `canvas`.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<TemplateGame, wasm_bindgen::JsValue> {
        Self::launch(canvas, build).await
    }
}
