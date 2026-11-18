use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_core::{
    BrickList, Bricks, Camera, DeviceLimits, Engine, Error, Transform, main_camera, place_camera,
};
use engine_render::{RenderSettings, wgpu};
use engine_scene::{SceneDocument, SceneFiles, SceneInstance};

pub fn engine(bricks: impl FnOnce(BrickList) -> BrickList) -> crate::Result<Engine> {
    listed(bricks(BrickList::default()))
}

pub(crate) fn listed(list: BrickList) -> crate::Result<Engine> {
    let mut engine = Engine::listed(list)?;
    engine.settings::<RenderSettings>();
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

    #[cfg(feature = "dev")]
    pub(crate) fn instance_mut(&mut self) -> Option<&mut SceneInstance> {
        self.instance.as_mut()
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
    pub name: &'static str,
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

pub fn default_build<G: Default>(_: &mut Engine) -> crate::Result<G> {
    Ok(G::default())
}
