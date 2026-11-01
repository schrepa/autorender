use bevy_ecs::resource::Resource;
use engine_core::{BrickList, DeviceLimits, Engine, Error, ErrorCode};
use engine_render::{Renderer, wgpu};
use engine_scene::{SceneFiles, SceneInstance};

pub fn engine(bricks: impl FnOnce(BrickList) -> BrickList) -> crate::Result<Engine> {
    listed(bricks(BrickList::default()))
}

pub(crate) fn listed(list: BrickList) -> crate::Result<Engine> {
    let mut engine = Engine::listed(list.with_core(Renderer::register_places))?;
    crate::resolution::register(&mut engine);
    engine
        .operation::<engine_scene::SceneViewControl>()
        .operation::<engine_scene::SceneLookControl>();
    let limits = device_limits(&wgpu::Limits::default(), wgpu::Features::empty(), 0);
    engine.app.world_mut().insert_resource(limits);
    Ok(engine)
}

pub(crate) fn device_limits(
    limits: &wgpu::Limits,
    features: wgpu::Features,
    cores: u32,
) -> DeviceLimits {
    DeviceLimits {
        storage_room: limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
        texture_layers: limits.max_texture_array_layers,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        indirect_first_instance: features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
        ..DeviceLimits::without_device(cores)
    }
}

pub trait Game: 'static {
    const ACTIONS: &'static [(&'static str, u32)] = &[("Space", 0)];

    fn before_frame(&mut self, _engine: &mut Engine) {}

    fn reset_camera(&mut self, _engine: &mut Engine) {}

    fn scene(&self) -> Option<&SceneInstance> {
        None
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        None
    }

    fn reload_scene(
        &mut self,
        _engine: &mut Engine,
        _source: &str,
        _files: &SceneFiles,
    ) -> Result<(), Error> {
        Err(Error::new(ErrorCode::UnsupportedCommand, "$.op").got("reload"))
    }

    fn inspectors(_inspectors: &mut Inspectors) {}
}

pub struct Inspectors<'a> {
    #[cfg(feature = "dev")]
    tools: &'a mut engine_dev::DevTools,
    #[cfg(not(feature = "dev"))]
    _tools: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> Inspectors<'a> {
    #[cfg(feature = "dev")]
    pub(crate) fn new(tools: &'a mut engine_dev::DevTools) -> Self {
        Self { tools }
    }

    pub fn resource<R: Resource>(
        &mut self,
        schemas: &str,
        name: &str,
        view: impl Fn(&R) -> serde_json::Value + Send + Sync + 'static,
    ) -> &mut Self {
        #[cfg(feature = "dev")]
        {
            let schemas: serde_json::Value =
                serde_json::from_str(schemas).expect("inspector_schema_json");
            self.tools
                .register_resource::<R>(
                    engine_dev::InspectorDescriptor::generated(&schemas, name),
                    view,
                )
                .unwrap_or_else(|error| panic!("inspector_valid {name} {error:?}"));
        }
        #[cfg(not(feature = "dev"))]
        let _ = (schemas, name, view);
        self
    }
}
