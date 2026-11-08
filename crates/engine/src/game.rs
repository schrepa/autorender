use bevy_ecs::resource::Resource;
use engine_core::{BrickList, DeviceLimits, Engine, Error, ErrorCode};
use engine_render::{Renderer, wgpu};
use engine_scene::{SceneFiles, SceneInstance, SceneLookControl, SceneMaterialControl};

use crate::host::{MENU_KEYS, MOVE_KEYS, MenuKey, MoveDirection};

pub fn engine(bricks: impl FnOnce(BrickList) -> BrickList) -> crate::Result<Engine> {
    listed(bricks(BrickList::default()))
}

pub(crate) fn listed(list: BrickList) -> crate::Result<Engine> {
    let mut engine = Engine::listed(list.with_core(Renderer::register_places))?;
    crate::resolution::register(&mut engine);
    engine
        .operation::<SceneLookControl>()
        .operation::<SceneMaterialControl>();
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

pub trait Game: 'static {
    const ACTIONS: &'static [(&'static str, u32)] = &[("Space", 0)];

    const AXES: &'static [(&'static str, u32, f32)] = &[];

    const MOVE_KEYS: &'static [(&'static str, MoveDirection)] = MOVE_KEYS;

    const MENU_KEYS: &'static [(&'static str, MenuKey)] = MENU_KEYS;

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
