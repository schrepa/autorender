//! An engine client: the entire scene uses only the public gameplay façade.
use engine::glam::{Quat, Vec3};
use engine::prelude::*;

/// The demo is deliberately bounded; changes to the workload happen outside frames.
pub const MAX_CUBES: u32 = 50_000;

#[derive(Component)]
struct Orbiting {
    anchor: Vec3,
    phase: f32,
    speed: f32,
}

/// Demonstrates extension using an ordinary plugin and named scheduling points.
pub struct OrbitDemoPlugin;

impl Plugin for OrbitDemoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FlyCameraController::new(0.0, -0.46));
        app.insert_resource(Camera {
            position: Vec3::new(0.0, 31.0, 64.0),
            ..Camera::default()
        });
        app.add_systems(FixedSimulation, animate_cubes);
        app.add_plugins(FlyCameraPlugin);
    }
}

fn animate_cubes(time: Res<FixedTime>, mut cubes: Query<(&Orbiting, &mut Transform)>) {
    let t = (time.elapsed_seconds % 10_000.0) as f32;
    for (orbit, mut transform) in &mut cubes {
        let angle = t * orbit.speed + orbit.phase;
        transform.translation = orbit.anchor + Vec3::Y * (math::sin(angle) * 1.3);
        transform.rotation = Quat::from_rotation_y(angle) * Quat::from_rotation_x(angle * 0.37);
    }
}

fn replace_cubes(engine: &mut Engine, count: u32) {
    let count = count.clamp(1, MAX_CUBES);
    let world = engine.app.world_mut();
    let entities: Vec<_> = world
        .query_filtered::<Entity, With<Orbiting>>()
        .iter(world)
        .collect();
    for entity in entities {
        world.despawn(entity);
    }
    let side = (count as f32).sqrt().ceil() as u32;
    world.spawn_batch((0..count).map(|i| {
        let x = (i % side) as f32 - (side - 1) as f32 * 0.5;
        let z = (i / side) as f32 - (side - 1) as f32 * 0.5;
        let anchor = Vec3::new(x * 2.1, 0.0, z * 2.1);
        let phase = (i as f32 * 2.399_963_1) % std::f32::consts::TAU;
        let color = match i % 6 {
            0 => [0.16, 0.78, 0.68, 1.0],
            1 => [0.36, 0.55, 0.96, 1.0],
            2 => [0.65, 0.43, 0.94, 1.0],
            3 => [0.98, 0.61, 0.31, 1.0],
            4 => [0.93, 0.35, 0.49, 1.0],
            _ => [0.64, 0.85, 0.92, 1.0],
        };
        (
            Transform {
                translation: anchor,
                scale: Vec3::splat(0.68),
                ..Transform::default()
            },
            Cube { color },
            InterpolateTransform,
            Orbiting {
                anchor,
                phase,
                speed: 0.3 + (i % 13) as f32 * 0.025,
            },
        )
    }));
}

/// The same gameplay world can run without a GPU for deterministic contract checks.
pub fn create_demo(count: u32) -> Engine {
    let mut engine = Engine::new(EngineConfig::default());
    engine.app.add_plugins(OrbitDemoPlugin);
    replace_cubes(&mut engine, count);
    engine
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use engine::browser::BrowserApp;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct Game {
        app: BrowserApp,
        #[cfg(feature = "dev")]
        development: engine::dev_browser::DevBrowser,
    }

    #[wasm_bindgen]
    impl Game {
        pub async fn create(
            canvas: web_sys::HtmlCanvasElement,
            count: u32,
        ) -> Result<Game, JsValue> {
            console_error_panic_hook::set_once();
            Ok(Self {
                app: BrowserApp::new(canvas, create_demo(count)).await?,
                #[cfg(feature = "dev")]
                development: engine::dev_browser::DevBrowser::default(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn frame(
            &mut self,
            timestamp_ms: f64,
            width: u32,
            height: u32,
            forward: f32,
            right: f32,
            up: f32,
            yaw: f32,
            pitch: f32,
        ) -> Result<(), JsValue> {
            self.app.frame(
                timestamp_ms,
                width,
                height,
                InputState {
                    move_forward: forward,
                    move_right: right,
                    move_up: up,
                    look_yaw: yaw,
                    look_pitch: pitch,
                },
            )
        }

        /// The last frame's statistics as a JSON object.
        pub fn stats(&self) -> String {
            self.app.stats_json()
        }

        pub fn set_count(&mut self, count: u32) {
            replace_cubes(&mut self.app.engine, count);
        }
        pub fn suspend(&mut self) {
            self.app.suspend();
            #[cfg(feature = "dev")]
            self.development.clear_input();
        }
        pub fn reset_camera(&mut self) {
            *self.app.engine.app.world_mut().resource_mut::<Camera>() = Camera {
                position: Vec3::new(0.0, 31.0, 64.0),
                ..Camera::default()
            };
            *self
                .app
                .engine
                .app
                .world_mut()
                .resource_mut::<FlyCameraController>() = FlyCameraController::new(0.0, -0.46);
        }
    }

    #[cfg(feature = "dev")]
    #[wasm_bindgen]
    impl Game {
        pub fn dev_command(&mut self, request: &str, width: u32, height: u32) -> String {
            self.development
                .command(&mut self.app, None, request, width, height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plugin_scene_advances_and_replaces_workload() {
        let mut engine = create_demo(100);
        engine.advance(1.0 / 60.0);
        let mut snapshot = engine::RenderSnapshot::default();
        engine.extract(16.0 / 9.0, &mut snapshot);
        assert_eq!(snapshot.instances.len(), 100);
        assert!(
            snapshot
                .instances
                .iter()
                .all(|instance| instance.model.is_finite())
        );
        replace_cubes(&mut engine, 12);
        engine.extract(1.0, &mut snapshot);
        assert_eq!(snapshot.instances.len(), 12);
    }

    #[test]
    fn camera_consumes_frame_input_even_without_a_fixed_tick() {
        let mut engine = create_demo(1);
        engine.advance(0.0);
        let before = engine.app.world().resource::<Camera>().position;
        engine
            .app
            .world_mut()
            .resource_mut::<InputState>()
            .move_forward = 1.0;
        engine.app.world_mut().resource_mut::<InputState>().look_yaw = 30.0;
        let report = engine.advance(0.01);
        assert_eq!(report.fixed_steps, 0);
        let after = engine.app.world().resource::<Camera>();
        assert!((after.position.distance(before) - 0.22).abs() < 0.0001);
        assert_ne!(after.target.x, after.position.x);
        assert!(after.view_projection(16.0 / 9.0).is_finite());
    }
}
