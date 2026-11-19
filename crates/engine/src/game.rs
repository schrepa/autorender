use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_core::{
    BrickList, Bricks, Camera, Engine, Error, Present, Show, Transform, main_camera, place_camera,
};
use engine_render::{GpuAsk, GpuRequest, RenderSettings, SurfaceForm, wgpu};
use engine_scene::{SceneDocument, SceneFiles, SceneInstance};

pub fn engine(bricks: BrickList, ticks_per_second: u32) -> crate::Result<Engine> {
    let mut engine = Engine::listed(bricks, ticks_per_second)?;
    engine.settings::<RenderSettings>();
    engine
        .app
        .world_mut()
        .insert_resource(engine_render::device_limits(&wgpu::Limits::default()));
    Ok(engine)
}

pub trait Game: Resource + Default {
    fn register(_bricks: &mut Bricks<'_>) {}

    fn start() -> Start {
        Start::default()
    }

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
}

pub struct Start {
    pub ticks_per_second: u32,
    pub gpu: GpuRequest,
    pub surface: SurfaceForm,
    pub present: Present,
    pub show: Show,
}

impl Default for Start {
    fn default() -> Self {
        let floor = wgpu::Limits::downlevel_defaults();
        Self {
            ticks_per_second: 60,
            gpu: GpuRequest {
                power: wgpu::PowerPreference::HighPerformance,
                need: GpuAsk {
                    features: wgpu::Features::empty(),
                    limits: floor.clone(),
                },
                want: GpuAsk {
                    features: wgpu::Features::TEXTURE_COMPRESSION_BC
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE,
                    limits: wgpu::Limits {
                        max_buffer_size: u64::MAX,
                        max_storage_buffer_binding_size: u64::MAX,
                        max_storage_buffers_per_shader_stage: u32::MAX,
                        max_texture_dimension_2d: u32::MAX,
                        max_texture_dimension_3d: u32::MAX,
                        max_texture_array_layers: u32::MAX,
                        max_color_attachments: u32::MAX,
                        max_color_attachment_bytes_per_sample: u32::MAX,
                        ..floor
                    },
                },
            },
            surface: SurfaceForm { srgb: false },
            present: Present {
                vsync: true,
                latency: 2,
            },
            show: Show::WhenReady,
        }
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

#[derive(Clone, Copy)]
pub struct GameEntry {
    pub name: &'static str,
    pub title: &'static str,
    pub folder: &'static str,
    pub bricks: fn(BrickList) -> BrickList,
    pub scene: DefaultScene,
}
