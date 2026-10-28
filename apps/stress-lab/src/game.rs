use engine::ErrorCode;
use engine::collections::HashMap;
use engine::host::TerrainDebug;

use engine::prelude::*;
use engine::scene::{SceneError, SceneFiles};

use crate::scene::EFFECTS;
use crate::{
    AuthoredScene, FrameMetrics, PlayerWalk, StressConfig, StressFile, StressMetrics,
    create_workload_with, look, set_view, stress_reflections,
};

const FREEZE_DETAIL: ActionId = ActionId::new(2).expect("an action's index");

pub struct StressLab {
    generation: u32,
    files: HashMap<String, StressFile>,
    look: String,
    effects: String,
    level_of_detail: bool,
    impostors: bool,
    visibility_buffer: bool,
    authored: Option<AuthoredScene>,
    freeze_held: bool,
}

impl StressLab {
    pub fn workload(
        config: StressConfig,
        files: HashMap<String, StressFile>,
    ) -> engine::Result<(Engine, Self)> {
        let engine = create_workload_with(config, &files)?;
        Ok((engine, Self::new(files, None)))
    }

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
            authored,
            freeze_held: false,
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn configure(
        &mut self,
        config: StressConfig,
        files: HashMap<String, StressFile>,
    ) -> engine::Result<Engine> {
        self.files.extend(files);
        create_workload_with(config, &self.files)
    }

    pub fn configured(&mut self, engine: &mut Engine) -> engine::Result<()> {
        self.authored = None;
        self.generation = self.generation.saturating_add(1);
        self.set_look(engine, &self.look.clone())?;
        self.set_effects(engine, &self.effects.clone())?;
        self.apply_world_settings(engine);
        Ok(())
    }

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
        if probes {
            world.insert_resource(ReflectionProbeSettings::default());
        } else {
            world.remove_resource::<ReflectionProbeSettings>();
        }
        self.effects = name.into();
        Ok(())
    }

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

    pub fn set_level_of_detail(&mut self, engine: &mut Engine, enabled: bool) {
        self.level_of_detail = enabled;
        self.apply_world_settings(engine);
    }

    pub fn set_impostors(&mut self, engine: &mut Engine, enabled: bool) {
        self.impostors = enabled;
        self.apply_world_settings(engine);
    }

    pub fn set_visibility_buffer(&mut self, engine: &mut Engine, enabled: bool) {
        self.visibility_buffer = enabled;
        self.apply_world_settings(engine);
    }

    pub fn scene_progress(&self, engine: &Engine) -> serde_json::Value {
        self.authored
            .as_ref()
            .map_or(serde_json::Value::Null, |authored| {
                authored.progress(engine)
            })
    }

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
    const ACTIONS: &'static [(&'static str, u32)] = &[("Space", 0), ("KeyL", 2)];

    fn before_frame(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        world.resource_mut::<FrameMetrics>().begin_frame();
        let held = world.resource::<ActionInput>().pressed(FREEZE_DETAIL);
        if held && !self.freeze_held {
            let eye = world.resource::<Camera>().eye();
            let mut debug = world
                .get_resource::<TerrainDebug>()
                .copied()
                .unwrap_or_default();
            debug.freeze = match debug.freeze {
                Some(_) => None,
                None => Some(eye),
            };
            world.insert_resource(debug);
        }
        self.freeze_held = held;
    }

    fn reset_camera(&mut self, engine: &mut Engine) {
        match &self.authored {
            Some(authored) => authored.reset_player(engine),
            None => set_view(engine, "overview").expect("known view"),
        }
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.authored.as_ref().and_then(AuthoredScene::instance)
    }

    fn content(&self, engine: &Engine) -> Option<Vec<u8>> {
        let mut content = crate::SOURCE.to_vec();
        if let Some(config) = engine.app.world().get_resource::<StressConfig>() {
            content.extend(serde_json::to_vec(config).expect("configurations serialize"));
        }
        if let Some(scene) = self.scene() {
            content.extend(scene.id().as_bytes());
            content.extend(scene.revision().to_le_bytes());
        }
        for choice in [&self.look, &self.effects] {
            content.extend(choice.as_bytes());
            content.push(0);
        }
        content.extend([
            u8::from(self.level_of_detail),
            u8::from(self.impostors),
            u8::from(self.visibility_buffer),
        ]);
        Some(content)
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
        let schemas = include_str!("generated/inspectors.json");
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
