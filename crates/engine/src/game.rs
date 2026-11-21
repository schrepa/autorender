use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_core::{
    AssetKind, Assets, BrickList, Bricks, Camera, Engine, Error, MeshFile, Present,
    Show, Texture, Transform, main_camera, place_camera,
};
use engine_render::{GpuAsk, GpuRequest, RenderSettings, SurfaceForm, wgpu};
use engine_scene::SceneInstance;

pub fn engine(bricks: BrickList, start: &Start) -> crate::Result<Engine> {
    let mut engine = Engine::listed(bricks, start.ticks_per_second, start.catch_up_seconds)?;
    engine.settings::<RenderSettings>();
    let world = engine.app.world_mut();
    world.insert_resource(engine_render::device_limits(&wgpu::Limits::default()));
    let mut assets = world.resource_mut::<Assets>();
    assets.kind(AssetKind::declared::<MeshFile>("mesh", MeshFile::read))?;
    assets.kind(AssetKind::declared::<Texture>(
        "texture",
        engine_textures::read_texture,
    ))?;
    assets.name_groups(start.groups.iter().map(|name| (*name).to_owned()).collect());
    Ok(engine)
}

pub trait Game: Resource + Default {
    fn register(_bricks: &mut Bricks<'_>) {}

    fn start() -> Start {
        Start::default()
    }

    fn reset_camera(engine: &mut Engine, scene: &LoadedScene) {
        scene.place_camera(engine.app.world_mut());
    }
}

pub struct Start {
    pub ticks_per_second: u32,
    pub catch_up_seconds: f64,
    pub gpu: GpuRequest,
    pub surface: SurfaceForm,
    pub present: Present,
    pub show: Show,
    pub max_pixel_ratio: Option<f32>,
    pub groups: &'static [&'static str],
    pub variant: Option<fn(wgpu::Features) -> &'static str>,
}

impl Default for Start {
    fn default() -> Self {
        let floor = wgpu::Limits::downlevel_defaults();
        Self {
            ticks_per_second: 60,
            catch_up_seconds: 5.0 / 60.0,
            gpu: GpuRequest {
                power: wgpu::PowerPreference::HighPerformance,
                need: GpuAsk {
                    features: wgpu::Features::empty(),
                    limits: floor.clone(),
                },
                want: GpuAsk {
                    features: wgpu::Features::TEXTURE_COMPRESSION_BC
                        | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE
                        | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
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
            max_pixel_ratio: Some(2.0),
            groups: &[],
            variant: None,
        }
    }
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
        self.start = None;
    }

    pub(crate) fn land(&mut self, world: &mut World) -> Result<bool, Error> {
        let Some(instance) = self.instance.as_mut() else {
            return Ok(false);
        };
        let first = instance.land(world)?;
        if first {
            self.start = main_camera(world).and_then(|camera| {
                Some((
                    *world.get::<Transform>(camera)?,
                    *world.get::<Camera>(camera)?,
                ))
            });
        }
        Ok(first)
    }
}

#[derive(Clone, Copy)]
pub struct GameEntry {
    pub name: &'static str,
    pub title: &'static str,
    #[cfg(feature = "dev")]
    pub folder: &'static str,
    pub bricks: fn(BrickList) -> BrickList,
    pub scene: &'static str,
    pub linked: Option<&'static str>,
}
