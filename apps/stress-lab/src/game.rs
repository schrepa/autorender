//! The Stress Lab as a game the engine hosts: a workload or an authored
//! scene, and the page's choices that outlive each workload's world.

use engine::ErrorCode;
use engine::collections::HashMap;

use engine::prelude::*;
use engine::scene::{SceneError, SceneFiles};

use crate::scene::EFFECTS;
use crate::{
    AuthoredScene, FrameMetrics, PlayerWalk, StressConfig, StressFile, StressMetrics,
    create_workload_with, look, set_view, stress_reflections, stress_shafts,
};

/// The Stress Lab beside its engine.
pub struct StressLab {
    // One more each time a configuration rebuilds the workload.
    generation: u32,
    // Files the page supplied, kept for later workloads.
    files: HashMap<String, StressFile>,
    // The lighting preset and effect selection, kept across workloads.
    look: String,
    effects: String,
    // Settings of the world's resources, kept across workloads, which
    // replace the world: levels of detail and impostors and the visibility
    // buffer.
    level_of_detail: bool,
    impostors: bool,
    visibility_buffer: bool,
    basin_fog: Option<bool>,
    basin_demo: Option<crate::basin_demo::BasinDemo>,
    // An authored scene in place of a workload, when the page loads one.
    authored: Option<AuthoredScene>,
}

impl StressLab {
    /// The workload `config` builds from `files`.
    pub fn workload(
        config: StressConfig,
        files: HashMap<String, StressFile>,
    ) -> engine::Result<(Engine, Self)> {
        let engine = create_workload_with(config, &files)?;
        Ok((engine, Self::new(files, None)))
    }

    /// The authored scene `source` with the `files` it references.
    pub fn authored(
        source: &str,
        files: &SceneFiles,
        without: &[Feature],
    ) -> engine::Result<(Engine, Self)> {
        let (engine, authored) = AuthoredScene::new(source, files, without)?;
        Ok((engine, Self::new(HashMap::default(), Some(authored))))
    }

    fn new(files: HashMap<String, StressFile>, authored: Option<AuthoredScene>) -> Self {
        Self {
            generation: 1,
            files,
            look: "scene".into(),
            effects: "all".into(),
            level_of_detail: true,
            impostors: true,
            visibility_buffer: false,
            basin_fog: None,
            basin_demo: None,
            authored,
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The engine of the workload `config` builds from the files supplied so
    /// far and `files`, for the host to run in place of the current one;
    /// then [`Self::configured`].
    pub fn configure(
        &mut self,
        config: StressConfig,
        files: HashMap<String, StressFile>,
    ) -> engine::Result<Engine> {
        self.files.extend(files);
        create_workload_with(config, &self.files)
    }

    /// Applies the kept look, effects and world settings to the engine
    /// [`Self::configure`] built, which replaced an authored scene too.
    pub fn configured(&mut self, engine: &mut Engine) -> engine::Result<()> {
        self.authored = None;
        self.basin_demo = None;
        self.generation = self.generation.saturating_add(1);
        self.set_look(engine, &self.look.clone())?;
        self.set_effects(engine, &self.effects.clone())?;
        self.apply_world_settings(engine);
        if let Some(enabled) = self.basin_fog {
            self.basin_fog = None;
            self.set_basin_fog(engine, enabled);
        }
        Ok(())
    }

    /// Rendering effects: `all`, `none`, or comma-separated `no-<effect>`
    /// exclusions.
    pub fn set_effects(&mut self, engine: &mut Engine, name: &str) -> engine::Result<()> {
        let mut on = [true; EFFECTS.len()];
        match name {
            "all" => {}
            "none" => on = [false; EFFECTS.len()],
            _ => {
                for token in name.split(',') {
                    let excluded = token
                        .strip_prefix("no-")
                        .and_then(|effect| EFFECTS.iter().position(|(name, _)| *name == effect));
                    match excluded {
                        Some(index) => on[index] = false,
                        None => {
                            return Err(engine::Error::game_rule(format!(
                                "effects must be all, none or comma-separated no-<effect> for {}",
                                EFFECTS.map(|(name, _)| name).join(", ")
                            )));
                        }
                    }
                }
            }
        }
        let [
            shadows,
            occlusion,
            bloom,
            lights,
            shafts,
            reflections,
            particles,
            refraction,
            probes,
        ] = on;
        let scene = engine.app.world().resource::<StressConfig>().scene;
        let (_, lamps) = look(scene, &self.look)?;
        let world = engine.app.world_mut();
        world.resource_mut::<Shadows>().enabled = shadows;
        world.resource_mut::<AmbientOcclusion>().intensity = if occlusion {
            AmbientOcclusion::default().intensity
        } else {
            0.0
        };
        world.resource_mut::<Bloom>().intensity = if bloom {
            Bloom::default().intensity
        } else {
            0.0
        };
        // Opt-in effects are removed entirely when off.
        if shafts {
            world.insert_resource(stress_shafts());
        } else {
            world.remove_resource::<SunShafts>();
        }
        if reflections {
            world.insert_resource(stress_reflections(scene));
        } else {
            world.remove_resource::<Reflections>();
        }
        {
            let mut local = world.resource_mut::<LocalLightSettings>();
            local.enabled = lights;
            local.intensity = lamps;
        }
        world.resource_mut::<ParticleSettings>().enabled = particles;
        if refraction && scene.world().refracts() {
            world.insert_resource(Refraction::default());
        } else {
            world.remove_resource::<Refraction>();
        }
        // Reflection probes capture nothing where none stand.
        if probes {
            world.insert_resource(ReflectionProbeSettings::default());
        } else {
            world.remove_resource::<ReflectionProbeSettings>();
        }
        self.effects = name.into();
        Ok(())
    }

    /// Lighting preset: `scene` (the scene's own), `golden`, `dusk`,
    /// `night` or `day`.
    pub fn set_look(&mut self, engine: &mut Engine, name: &str) -> engine::Result<()> {
        let scene = engine.app.world().resource::<StressConfig>().scene;
        let (atmosphere, lamps) = look(scene, name)?;
        engine.app.insert_resource(atmosphere);
        crate::scene::apply_sky(engine.app.world_mut(), scene, name);
        if let Some(mut local) = engine
            .app
            .world_mut()
            .get_resource_mut::<LocalLightSettings>()
        {
            local.intensity = lamps;
        }
        self.look = name.into();
        Ok(())
    }

    /// Whether instances draw at levels of detail (on by default) or always
    /// as their mesh, in this and later workloads.
    pub fn set_level_of_detail(&mut self, engine: &mut Engine, enabled: bool) {
        self.level_of_detail = enabled;
        self.apply_world_settings(engine);
    }

    /// Whether draw groups draw impostors where they fit (on by default) or
    /// only their meshes' levels, in this and later workloads.
    pub fn set_impostors(&mut self, engine: &mut Engine, enabled: bool) {
        self.impostors = enabled;
        self.apply_world_settings(engine);
    }

    /// Whether virtual geometry shades through the visibility buffer or as
    /// it draws (the default), in this and later workloads.
    pub fn set_visibility_buffer(&mut self, engine: &mut Engine, enabled: bool) {
        self.visibility_buffer = enabled;
        self.apply_world_settings(engine);
    }

    /// Enables optional basin fog; only the actual Highlands workload adds
    /// the fixed basin demonstration. Repeating the same choice is idempotent.
    pub fn set_basin_fog(&mut self, engine: &mut Engine, enabled: bool) {
        if self.basin_fog == Some(enabled) {
            return;
        }
        self.basin_fog = Some(enabled);
        if enabled {
            if self.authored.is_none()
                && engine
                    .app
                    .world()
                    .get_resource::<StressConfig>()
                    .is_some_and(|c| c.scene == StressScene::Highlands)
            {
                self.basin_demo = Some(crate::basin_demo::BasinDemo::new(engine));
            }
            engine.app.world_mut().insert_resource(BasinFog::default());
        } else {
            if let Some(demo) = self.basin_demo.take() {
                demo.restore(engine);
            }
            engine.app.world_mut().remove_resource::<BasinFog>();
        }
    }

    /// The authored scene's ID, revision, frame, player position and asset
    /// counts, as a JSON object (`null` for workloads).
    pub fn scene_progress(&self, engine: &Engine) -> serde_json::Value {
        self.authored
            .as_ref()
            .map_or(serde_json::Value::Null, |authored| {
                authored.progress(engine)
            })
    }

    /// Writes the kept settings into the world's resources: each workload
    /// builds a new world with their defaults.
    fn apply_world_settings(&self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        if let Some(mut settings) = world.get_resource_mut::<LevelOfDetail>() {
            settings.enabled = self.level_of_detail;
            settings.impostors = self.impostors;
        }
        if let Some(mut settings) = world.get_resource_mut::<VirtualGeometry>() {
            settings.visibility_buffer = self.visibility_buffer;
        }
    }
}

impl Game for StressLab {
    fn before_frame(&mut self, engine: &mut Engine) {
        if let Some(demo) = &mut self.basin_demo {
            demo.prepare(engine);
        }
        engine
            .app
            .world_mut()
            .resource_mut::<FrameMetrics>()
            .begin_frame();
    }

    /// The workload's overview, or the authored scene's player back where
    /// it started.
    fn reset_camera(&mut self, engine: &mut Engine) {
        match &self.authored {
            Some(authored) => authored.reset_player(engine),
            None => set_view(engine, "overview").expect("known view"),
        }
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.authored.as_ref().and_then(AuthoredScene::instance)
    }

    fn reload_scene(
        &mut self,
        engine: &mut Engine,
        source: &str,
        files: &SceneFiles,
    ) -> Result<(), SceneError> {
        let Some(authored) = self.authored.as_mut() else {
            return Err(SceneError {
                code: ErrorCode::UnsupportedCommand,
                path: "$.op".into(),
                message: "workloads have no authored scene to reload".into(),
            });
        };
        authored.reload(engine, source, files)?;
        self.apply_world_settings(engine);
        Ok(())
    }

    fn inspectors(inspectors: &mut Inspectors) {
        // The types' schemas, as `cargo xtask schema` generates them.
        let schemas = include_str!("generated/inspectors.json");
        inspectors.resource::<crate::BasinDemoView>(schemas, "BasinDemo", |view| {
            serde_json::to_value(view).expect("finite demo values")
        });
        inspectors
            .resource::<StressConfig>(schemas, "StressConfig", |config| {
                serde_json::to_value(config).expect("configurations serialize")
            })
            .resource::<FrameMetrics>(schemas, "StressMetrics", |metrics| {
                serde_json::to_value(StressMetrics::from(metrics)).expect("metrics serialize")
            })
            .resource::<PlayerWalk>(schemas, "PlayerWalk", |walk| {
                serde_json::to_value(walk).expect("walk counts serialize")
            });
    }
}
