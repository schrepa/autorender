use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_core::{
    BrickList, Bricks, Camera, DeviceLimits, Engine, Error, Smoke, Transform, main_camera,
    place_camera,
};
use engine_render::{Renderer, wgpu};
use engine_scene::{
    RemoveOperation, SceneDocument, SceneFiles, SceneInstance, SceneLookControl,
    SceneMaterialControl, SpawnOperation,
};

pub fn engine(bricks: impl FnOnce(BrickList) -> BrickList) -> crate::Result<Engine> {
    listed(bricks(BrickList::default()))
}

pub(crate) fn listed(list: BrickList) -> crate::Result<Engine> {
    let mut engine = Engine::listed(list.with_core(Renderer::register_places))?;
    crate::resolution::register(&mut engine);
    crate::pacing::register(&mut engine);
    engine
        .operation::<SceneLookControl>()
        .operation::<SceneMaterialControl>()
        .operation::<SpawnOperation>()
        .operation::<RemoveOperation>();
    let unwired = wgpu::Limits {
        max_texture_array_layers: 0,
        ..wgpu::Limits::default()
    };
    limit(
        &mut engine,
        engine_render::device_limits(&unwired, wgpu::Features::empty(), 0),
    );
    Ok(engine)
}

pub(crate) fn limit(engine: &mut Engine, limits: DeviceLimits) {
    let world = engine.app.world_mut();
    let cores = world.resource::<DeviceLimits>().cores;
    world.insert_resource(DeviceLimits { cores, ..limits });
}

pub trait Game: Resource + Sized {
    const FILES: &'static [&'static str] = &[];

    const SMOKES: &'static [Smoke] = &[];

    fn register(_bricks: &mut Bricks<'_>) {}

    fn load_scene(
        engine: &mut Engine,
        document: &SceneDocument,
        files: &SceneFiles,
        _arrival: Arrival,
    ) -> Result<SceneInstance, Error> {
        document.instantiate_with(engine.app.world_mut(), files)
    }

    fn reset_camera(engine: &mut Engine, scene: &LoadedScene) {
        scene.place_camera(engine.app.world_mut());
    }

    fn content(_engine: &Engine, _scene: &LoadedScene) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    Opening,
    Reload,
}

#[derive(Default)]
pub struct LoadedScene {
    instance: Option<SceneInstance>,
    start: Option<(Transform, Camera)>,
}

impl LoadedScene {
    pub fn instance(&self) -> Option<&SceneInstance> {
        self.instance.as_ref()
    }

    pub fn start(&self) -> Option<(Transform, Camera)> {
        self.start
    }

    pub fn place_camera(&self, world: &mut World) {
        if let Some((at, lens)) = self.start
            && let Some(camera) = main_camera(world)
        {
            place_camera(world, camera, at, lens);
        }
    }

    pub(crate) fn replace(&mut self, world: &mut World, instance: SceneInstance) {
        if let Some(previous) = self.instance.replace(instance) {
            previous.despawn(world);
        }
        self.start = main_camera(world).and_then(|camera| {
            Some((
                *world.get::<Transform>(camera)?,
                *world.get::<Camera>(camera)?,
            ))
        });
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultScene {
    pub name: &'static str,
    pub path: &'static str,
    pub text: &'static str,
}

pub struct GameEntry<G> {
    pub title: &'static str,
    pub folder: &'static str,
    pub bricks: fn(BrickList) -> BrickList,
    pub build: fn(&mut Engine) -> crate::Result<G>,
    pub scene: Option<DefaultScene>,
}

impl<G> Clone for GameEntry<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G> Copy for GameEntry<G> {}

#[derive(Resource, Default)]
pub struct GameFiles {
    files: SceneFiles,
    root: String,
}

impl GameFiles {
    pub fn new(files: SceneFiles, root: impl Into<String>) -> Self {
        Self {
            files,
            root: root.into(),
        }
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.files.buffer(name, "$").ok()
    }

    pub fn root(&self) -> &str {
        &self.root
    }
}

impl From<SceneFiles> for GameFiles {
    fn from(files: SceneFiles) -> Self {
        Self::new(files, "")
    }
}

pub fn report_missing<G: Game>(files: &GameFiles) {
    for file in G::FILES {
        if files.bytes(file).is_some() {
            continue;
        }
        let record = serde_json::json!({
            "event": "optional_file_missing",
            "file": file,
            "level": "notice",
        });
        #[cfg(target_arch = "wasm32")]
        web_sys::console::info_1(&wasm_bindgen::JsValue::from_str(&record.to_string()));
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("{record}");
    }
}

pub fn default_build<G: Default>(_: &mut Engine) -> crate::Result<G> {
    Ok(G::default())
}

pub fn built<G: Game>(
    entry: &GameEntry<G>,
    engine: &mut Engine,
    files: impl Into<GameFiles>,
) -> crate::Result<G> {
    engine.app.world_mut().insert_resource(files.into());
    let game = (entry.build)(engine);
    engine.app.world_mut().remove_resource::<GameFiles>();
    game
}
