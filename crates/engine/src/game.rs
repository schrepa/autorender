//! Games the page runs: the state beside an [`Engine`] that
//! [`crate::export_game!`] exports with the methods every game has.

use engine_core::{Engine, bevy_ecs::resource::Resource};
use engine_scene::{SceneError, SceneFiles, SceneInstance};

/// A game: what it keeps beside its engine and the hooks the engine calls.
/// The engine runs frames, jobs, loading, statistics, development tools and
/// profiling for every game.
pub trait Game: 'static {
    /// Runs before each frame the engine simulates or submits, development
    /// steps and renders included.
    fn before_frame(&mut self, _engine: &mut Engine) {}

    /// Puts the camera where the game starts.
    fn reset_camera(&mut self, _engine: &mut Engine) {}

    /// The authored scene running, which development tools inspect and
    /// reload.
    fn scene(&self) -> Option<&SceneInstance> {
        None
    }

    /// Replaces the authored scene with `source` and the `files` it
    /// references; a scene that fails keeps the current one. Development
    /// reloads call it once the source's scene ID and revision matched.
    fn reload_scene(
        &mut self,
        _engine: &mut Engine,
        _source: &str,
        _files: &SceneFiles,
    ) -> Result<(), SceneError> {
        Err(SceneError {
            code: "unsupported_command",
            path: "$.op".into(),
            message: "this game runs no authored scene".into(),
        })
    }

    /// Registers the game's own inspectors with the development tools.
    fn inspectors(_inspectors: &mut Inspectors) {}
}

/// The development tools' inspectors, which games register their own with;
/// outside development builds registration does nothing.
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

    /// Inspects resource `R` as `view` reads it, described by the schema
    /// named `name` in `schemas`, the JSON object of schemas the game
    /// generates. A missing schema or description panics.
    pub fn resource<R: Resource>(
        &mut self,
        schemas: &str,
        name: &str,
        view: impl Fn(&R) -> serde_json::Value + Send + Sync + 'static,
    ) -> &mut Self {
        #[cfg(feature = "dev")]
        {
            let schemas: serde_json::Value =
                serde_json::from_str(schemas).expect("inspector schemas are JSON");
            self.tools
                .register_resource::<R>(
                    engine_dev::InspectorDescriptor::generated(&schemas, name),
                    view,
                )
                .unwrap_or_else(|error| panic!("inspector {name} is invalid: {error:?}"));
        }
        #[cfg(not(feature = "dev"))]
        let _ = (schemas, name, view);
        self
    }
}
